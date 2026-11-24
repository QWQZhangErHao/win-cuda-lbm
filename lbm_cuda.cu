// ============================================================================
// lbm_cuda.cu —— Windows 原生 CUDA C++ 高性能 D3Q19 LBM 求解器
// 集成四大工程级优化（全 FP32 精度零损失）：
//   1. AA-Pattern 单数组原地演进（消除双缓冲，显存减半：76 B/体素，26.19M 仅需 1.99 GB）
//   2. Even 步 100% 本地合并读写 + Odd 步内部体素无取模/无分支快路径
//   3. CUDA Graph 批量打包提交（每次 20 步单次 Launch，大幅削减 WDDM 驱动开销）
//   4. Warp 32×4×2 合并访存 + __constant__ 常量展开 + --use_fast_math
// ============================================================================

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
#define LBM_API extern "C" __declspec(dllexport)
#else
#define LBM_API extern "C"
#endif

// D3Q19 离散速度集
__device__ __constant__ int c_ex[19] = {
    0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0
};
__device__ __constant__ int c_ey[19] = {
    0,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  1, -1,  1, -1
};
__device__ __constant__ int c_ez[19] = {
    0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1
};
__device__ __constant__ int c_opp[19] = {
    0,  2,  1,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17
};
__device__ __constant__ float c_w[19] = {
    1.0f / 3.0f,
    1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f, 1.0f / 18.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f,
    1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f
};

// D3Q7 温度场编译期常量表（cs_T^2 = 1/4）
__device__ __constant__ int ct_ex[7]  = {0,  1, -1,  0,  0,  0,  0};
__device__ __constant__ int ct_ey[7]  = {0,  0,  0,  1, -1,  0,  0};
__device__ __constant__ int ct_ez[7]  = {0,  0,  0,  0,  0,  1, -1};
__device__ __constant__ int ct_opp[7] = {0,  2,  1,  4,  3,  6,  5};
__device__ __constant__ float ct_w[7] = {
    0.25f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f, 0.125f
};

#define GRAPH_PAIR_STEPS 10   // 每个 CUDA Graph 包含 10 对 (Even+Odd) = 20 步

struct DeviceMetrics {
    double sum_uy;
    double sum_rho_in;
    double sum_rho_out;
    double sum_t_out;
    unsigned long long cnt_fluid;
    unsigned long long cnt_in;
    unsigned long long cnt_out;
    float max_t_solid;
};

struct LBMHandle {
    int nx, ny, nz;
    size_t ncells;
    float u_in;
    float tau0;
    float nu0;
    float cs2;
    float gy;
    int periodic_y;
    float tau_min;
    float tau_max;
    int parity;       // 0: 正常正向槽位 (d_f[i] = f_i); 1: 反向槽位 (d_f[opp[i]] = f_i)
    float* d_f;       // [19, nz, ny, nx] 单数组 AA-Pattern (FP32)
    uint8_t* d_solid; // [nz, ny, nx]
    // 可选 D3Q7 共轭传热温度场
    int thermal_enabled;
    float t_in;
    float t_hot;
    float tau_t_fluid;
    float tau_t_solid;
    float* d_g;             // [7, nz, ny, nx] 单数组 AA-Pattern (FP32)
    uint8_t* d_thermal_bc;  // 0=fluid, 1=solid(Al fin), 2=isothermal heat source, 3=copper pipe/base
    DeviceMetrics* d_metrics; // 64B 显存内归约缓存（零 3D 临时显存）
    cudaStream_t stream;
    cudaGraphExec_t graph_exec;
    int graph_ready;
};

#define CUDA_CHECK(expr)                                                          \
    do {                                                                          \
        cudaError_t _err = (expr);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "[lbm_cuda] CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(_err));                \
            return -1;                                                            \
        }                                                                         \
    } while (0)

// ----------------------------------------------------------------------------
// 公共碰撞内联函数（TRT 双松弛 + 壁面阻尼 Smagorinsky LES）
// ----------------------------------------------------------------------------
__device__ __forceinline__ void lbm_collide_fp32(
    float* ff,
    float tau0, float nu0, float cs2,
    float gy, float tau_min, float tau_max
) {
    float rho = 0.0f;
    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    float mxx = 0.0f, myy = 0.0f, mzz = 0.0f;
    float mxy = 0.0f, mxz = 0.0f, myz = 0.0f;

    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        float v = ff[i];
        float cx = (float)c_ex[i];
        float cy = (float)c_ey[i];
        float cz = (float)c_ez[i];
        rho += v;
        ux  += cx * v;
        uy  += cy * v;
        uz  += cz * v;
        mxx += cx * cx * v;
        myy += cy * cy * v;
        mzz += cz * cz * v;
        mxy += cx * cy * v;
        mxz += cx * cz * v;
        myz += cy * cz * v;
    }

    float inv_rho = 1.0f / rho;
    ux *= inv_rho;
    uy *= inv_rho;
    uz *= inv_rho;

    float tau_plus = tau0;
    if (cs2 > 0.0f) {
        constexpr float CS2_LBM = 1.0f / 3.0f;
        float pxx = mxx - rho * (CS2_LBM + ux * ux);
        float pyy = myy - rho * (CS2_LBM + uy * uy);
        float pzz = mzz - rho * (CS2_LBM + uz * uz);
        float pxy = mxy - rho * ux * uy;
        float pxz = mxz - rho * ux * uz;
        float pyz = myz - rho * uy * uz;
        float q2 = pxx * pxx + pyy * pyy + pzz * pzz +
                   2.0f * (pxy * pxy + pxz * pxz + pyz * pyz);
        if (q2 > 0.0f) {
            // 标准 Smagorinsky 一元二次方程解析解
            tau_plus = 0.5f * (tau0 + sqrtf(tau0 * tau0 + 18.0f * cs2 * sqrtf(2.0f * q2) * inv_rho));
            tau_plus = fminf(fmaxf(tau_plus, tau_min), tau_max);
        }
    }

    // TRT 魔术参数 Lambda = 3/16，锁定半步反弹精确壁面位置
    constexpr float TRT_LAMBDA = 3.0f / 16.0f;
    float tau_minus = 0.5f + TRT_LAMBDA / fmaxf(tau_plus - 0.5f, 1e-4f);
    float omega_plus  = 1.0f / tau_plus;
    float omega_minus = 1.0f / tau_minus;

    // 静止方向 i = 0 原地更新（省去 feq[19] 和 f_out[19]，立省 38 个寄存器）
    float u2_term = 1.5f * (ux * ux + uy * uy + uz * uz);
    float feq0 = c_w[0] * rho * (1.0f - u2_term);
    ff[0] = ff[0] - omega_plus * (ff[0] - feq0);

    // 其余 9 对相反方向 (1,2), (3,4), ..., (17,18) 原地成对更新
    #pragma unroll
    for (int i = 1; i < 19; i += 2) {
        int opp = i + 1;
        float fi = ff[i];
        float fo = ff[opp];

        float cu = 3.0f * ((float)c_ex[i] * ux + (float)c_ey[i] * uy + (float)c_ez[i] * uz);
        float feq_plus  = c_w[i] * rho * (1.0f + 0.5f * cu * cu - u2_term);
        float feq_minus = c_w[i] * rho * cu;

        float f_plus  = 0.5f * (fi + fo);
        float f_minus = 0.5f * (fi - fo);

        float d_plus  = omega_plus  * (f_plus  - feq_plus);
        float d_minus = omega_minus * (f_minus - feq_minus);
        float force   = 3.0f * c_w[i] * (float)c_ey[i] * gy;

        ff[i]   = fi - d_plus - d_minus + force;
        ff[opp] = fo - d_plus + d_minus - force;
    }
}

// ----------------------------------------------------------------------------
// 初始化 Kernel
// ----------------------------------------------------------------------------
__global__ void k_init_field(
    float* __restrict__ f,
    int nx, int ny, int nz, size_t ncells,
    float u_in
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
    float u2_term = 1.5f * u_in * u_in;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        float cu = 3.0f * (float)c_ey[i] * u_in;
        f[(size_t)i * ncells + idx] = c_w[i] * (1.0f + cu + 0.5f * cu * cu - u2_term);
    }
}

// ----------------------------------------------------------------------------
// Even Step Kernel：100% 本地合并读写（无邻居访存、无分支）
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void k_lbm_even(
    float* __restrict__ f,
    const uint8_t* __restrict__ solid,
    float tau0, float nu0, float cs2,
    int nx, int ny, int nz, size_t ncells,
    float gy, float tau_min, float tau_max
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
    if (__ldg(&solid[idx])) return;

    float ff[19];
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        ff[i] = f[(size_t)i * ncells + idx];
    }

    lbm_collide_fp32(ff, tau0, nu0, cs2, gy, tau_min, tau_max);

    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        f[(size_t)c_opp[i] * ncells + idx] = ff[i];
    }
}

// ----------------------------------------------------------------------------
// Odd Step Kernel：从邻居 opp[i] 拉取 -> 碰撞 -> 写回邻居 i（含内部无取模快路径）
// ----------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void k_lbm_odd(
    float* __restrict__ f,
    const uint8_t* __restrict__ solid,
    float tau0, float nu0, float cs2,
    int nx, int ny, int nz, size_t ncells,
    float u_in, float gy, int periodic_y, float tau_min, float tau_max
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t stride_z = (size_t)ny * (size_t)nx;
    size_t idx = (size_t)z * stride_z + (size_t)y * (size_t)nx + (size_t)x;
    if (__ldg(&solid[idx])) return;

    float ff[19];
    bool is_interior = (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1);

    if (is_interior) {
        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            size_t nb_idx = (size_t)(z - c_ez[i]) * stride_z
                          + (size_t)(y - c_ey[i]) * (size_t)nx
                          + (size_t)(x - c_ex[i]);
            if (__ldg(&solid[nb_idx])) {
                ff[i] = f[(size_t)i * ncells + idx];
            } else {
                ff[i] = f[(size_t)c_opp[i] * ncells + nb_idx];
            }
        }

        lbm_collide_fp32(ff, tau0, nu0, cs2, gy, tau_min, tau_max);

        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            size_t dst_idx = (size_t)(z + c_ez[i]) * stride_z
                           + (size_t)(y + c_ey[i]) * (size_t)nx
                           + (size_t)(x + c_ex[i]);
            if (__ldg(&solid[dst_idx])) {
                f[(size_t)c_opp[i] * ncells + idx] = ff[i];
            } else {
                f[(size_t)i * ncells + dst_idx] = ff[i];
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            int xs = x - c_ex[i];
            if (xs < 0) xs += nx; else if (xs >= nx) xs -= nx;
            int zs = z - c_ez[i];
            if (zs < 0) zs += nz; else if (zs >= nz) zs -= nz;
            int ys = y - c_ey[i];
            if (periodic_y) {
                if (ys < 0) ys += ny; else if (ys >= ny) ys -= ny;
            }
            if (ys < 0) {
                float cu = 3.0f * (float)c_ey[i] * u_in;
                ff[i] = c_w[i] * (1.0f + cu + 0.5f * cu * cu - 1.5f * u_in * u_in);
            } else if (ys >= ny) {
                ff[i] = c_w[i];
            } else {
                size_t nb_idx = (size_t)zs * stride_z + (size_t)ys * (size_t)nx + (size_t)xs;
                if (__ldg(&solid[nb_idx])) {
                    ff[i] = f[(size_t)i * ncells + idx];
                } else {
                    ff[i] = f[(size_t)c_opp[i] * ncells + nb_idx];
                }
            }
        }

        lbm_collide_fp32(ff, tau0, nu0, cs2, gy, tau_min, tau_max);

        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            int xd = x + c_ex[i];
            if (xd < 0) xd += nx; else if (xd >= nx) xd -= nx;
            int zd = z + c_ez[i];
            if (zd < 0) zd += nz; else if (zd >= nz) zd -= nz;
            int yd = y + c_ey[i];
            if (periodic_y) {
                if (yd < 0) yd += ny; else if (yd >= ny) yd -= ny;
            }
            if (yd >= 0 && yd < ny) {
                size_t dst_idx = (size_t)zd * stride_z + (size_t)yd * (size_t)nx + (size_t)xd;
                if (__ldg(&solid[dst_idx])) {
                    f[(size_t)c_opp[i] * ncells + idx] = ff[i];
                } else {
                    f[(size_t)i * ncells + dst_idx] = ff[i];
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 合并的入口 (y=0) + 出口 (y=ny-1) 二阶非平衡态外推边界 (Guo 格式，跳过固体掩码)
// ----------------------------------------------------------------------------
__global__ void k_apply_bc(
    float* __restrict__ f,
    const uint8_t* __restrict__ solid,
    float u_in,
    int nx, int ny, int nz, size_t ncells,
    int flip_slots
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || z >= nz) return;

    size_t stride_z = (size_t)ny * (size_t)nx;

    // 1. 速度入口 y = 0（二阶非平衡态外推）
    size_t idx0 = (size_t)z * stride_z + (size_t)x;
    size_t idx1 = idx0 + (size_t)nx;
    if (!__ldg(&solid[idx0]) && !__ldg(&solid[idx1])) {
        float rho1 = 0.0f, ux1 = 0.0f, uy1 = 0.0f, uz1 = 0.0f;
        float f1_buf[19];
        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            int slot = flip_slots ? c_opp[i] : i;
            float v = f[(size_t)slot * ncells + idx1];
            f1_buf[i] = v;
            rho1 += v;
            ux1  += (float)c_ex[i] * v;
            uy1  += (float)c_ey[i] * v;
            uz1  += (float)c_ez[i] * v;
        }
        if (rho1 < 1e-4f) rho1 = 1.0f;
        float inv_rho1 = 1.0f / rho1;
        ux1 *= inv_rho1; uy1 *= inv_rho1; uz1 *= inv_rho1;
        float u2_1  = 1.5f * (ux1 * ux1 + uy1 * uy1 + uz1 * uz1);
        float u2_in = 1.5f * u_in * u_in;

        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            int slot = flip_slots ? c_opp[i] : i;
            float cu1   = 3.0f * ((float)c_ex[i] * ux1 + (float)c_ey[i] * uy1 + (float)c_ez[i] * uz1);
            float feq1  = c_w[i] * rho1 * (1.0f + cu1 + 0.5f * cu1 * cu1 - u2_1);
            float cu_in = 3.0f * (float)c_ey[i] * u_in;
            float feq0  = c_w[i] * rho1 * (1.0f + cu_in + 0.5f * cu_in * cu_in - u2_in);
            f[(size_t)slot * ncells + idx0] = feq0 + (f1_buf[i] - feq1);
        }
    }

    // 2. 压力出口 y = ny - 1（固定参考密度 rho = 1.0 + 非平衡态外推）
    size_t idx_up  = (size_t)z * stride_z + (size_t)(ny - 2) * (size_t)nx + (size_t)x;
    size_t idx_out = (size_t)z * stride_z + (size_t)(ny - 1) * (size_t)nx + (size_t)x;
    if (!__ldg(&solid[idx_out]) && !__ldg(&solid[idx_up])) {
        float rho_up = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
        float fup_buf[19];
        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            int slot = flip_slots ? c_opp[i] : i;
            float v = f[(size_t)slot * ncells + idx_up];
            fup_buf[i] = v;
            rho_up += v;
            ux += (float)c_ex[i] * v;
            uy += (float)c_ey[i] * v;
            uz += (float)c_ez[i] * v;
        }
        if (rho_up < 1e-4f) rho_up = 1.0f;
        float inv_rho = 1.0f / rho_up;
        ux *= inv_rho; uy *= inv_rho; uz *= inv_rho;
        float u2_out = 1.5f * (ux * ux + uy * uy + uz * uz);

        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            int slot = flip_slots ? c_opp[i] : i;
            float cu      = 3.0f * ((float)c_ex[i] * ux + (float)c_ey[i] * uy + (float)c_ez[i] * uz);
            float feq_up  = c_w[i] * rho_up * (1.0f + cu + 0.5f * cu * cu - u2_out);
            float feq_out = c_w[i] * 1.0f   * (1.0f + cu + 0.5f * cu * cu - u2_out);
            f[(size_t)slot * ncells + idx_out] = feq_out + (fup_buf[i] - feq_up);
        }
    }
}

// ----------------------------------------------------------------------------
// 切片提取 Kernel
// ----------------------------------------------------------------------------
__global__ void k_extract_y_plane(
    const float* __restrict__ f,
    const uint8_t* __restrict__ solid,
    float* __restrict__ out_rho,
    float* __restrict__ out_uy,
    int y, int nx, int ny, int nz, size_t ncells, int flip_slots
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || z >= nz) return;

    size_t plane_idx = (size_t)z * (size_t)nx + (size_t)x;
    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
    if (__ldg(&solid[idx])) {
        out_rho[plane_idx] = 0.0f;
        out_uy[plane_idx] = 0.0f;
        return;
    }
    float rho = 0.0f, uy = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        int slot = flip_slots ? c_opp[i] : i;
        float v = __ldg(&f[(size_t)slot * ncells + idx]);
        rho += v;
        uy += (float)c_ey[i] * v;
    }
    out_rho[plane_idx] = rho;
    out_uy[plane_idx] = (rho > 1e-8f) ? (uy / rho) : 0.0f;
}

__global__ void k_extract_z_slice(
    const float* __restrict__ f,
    const uint8_t* __restrict__ solid,
    float* __restrict__ out_rho,
    float* __restrict__ out_uy,
    int z, int nx, int ny, int nz, size_t ncells, int flip_slots
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    size_t slice_idx = (size_t)y * (size_t)nx + (size_t)x;
    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + slice_idx;
    if (__ldg(&solid[idx])) {
        out_rho[slice_idx] = 0.0f;
        out_uy[slice_idx] = 0.0f;
        return;
    }
    float rho = 0.0f, uy = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        int slot = flip_slots ? c_opp[i] : i;
        float v = __ldg(&f[(size_t)slot * ncells + idx]);
        rho += v;
        uy += (float)c_ey[i] * v;
    }
    out_rho[slice_idx] = rho;
    out_uy[slice_idx] = (rho > 1e-8f) ? (uy / rho) : 0.0f;
}

__global__ void k_extract_x_slice(
    const float* __restrict__ f,
    const uint8_t* __restrict__ solid,
    float* __restrict__ out_rho,
    float* __restrict__ out_uy,
    int x, int nx, int ny, int nz, size_t ncells, int flip_slots
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= ny || z >= nz) return;

    size_t slice_idx = (size_t)z * (size_t)ny + (size_t)y;
    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
    if (__ldg(&solid[idx])) {
        out_rho[slice_idx] = 0.0f;
        out_uy[slice_idx] = 0.0f;
        return;
    }
    float rho = 0.0f, uy = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        int slot = flip_slots ? c_opp[i] : i;
        float v = __ldg(&f[(size_t)slot * ncells + idx]);
        rho += v;
        uy += (float)c_ey[i] * v;
    }
    out_rho[slice_idx] = rho;
    out_uy[slice_idx] = (rho > 1e-8f) ? (uy / rho) : 0.0f;
}

// ============================================================================
// D3Q7 共轭传热温度场 Kernel（AA-Pattern 单数组：流体对流扩散 + 翅片导热 + 恒温热源）
// ============================================================================
__global__ void k_init_thermal(
    float* __restrict__ g,
    const uint8_t* __restrict__ thermal_bc,
    int nx, int ny, int nz, size_t ncells,
    float t_in, float t_hot
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
    float T = (thermal_bc[idx] == 2) ? t_hot : t_in;
    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        g[(size_t)i * ncells + idx] = ct_w[i] * T;
    }
}

__device__ __forceinline__ void thermal_collide_d3q7(
    float* gg, uint8_t bc,
    float ux, float uy, float uz,
    float t_hot, float tau_t_fluid, float tau_t_solid
) {
    if (bc == 2) {
        #pragma unroll
        for (int i = 0; i < 7; ++i) gg[i] = ct_w[i] * t_hot;
        return;
    }
    float T = 0.0f;
    #pragma unroll
    for (int i = 0; i < 7; ++i) T += gg[i];

    // bc=0: 空气流体 (tau >= 0.55), bc=1: 铝翅片 (tau=1.25 -> alpha~0.75), bc=3: 铜热管/底座 (tau=1.98 -> alpha~1.48)
    // 铜/铝保持真实 ~2:1 相对导热比，铝/空气保持 15:1 导热比且温度严格有界
    float tau_t = (bc == 0) ? fmaxf(tau_t_fluid, 0.55f)
                            : ((bc == 3) ? 1.98f : fminf(tau_t_solid, 1.25f));
    float inv_tau = 1.0f / tau_t;

    if (bc != 0) {
        ux = 0.0f; uy = 0.0f; uz = 0.0f;
    } else {
        ux = fminf(fmaxf(ux, -0.18f), 0.18f);
        uy = fminf(fmaxf(uy, -0.18f), 0.18f);
        uz = fminf(fmaxf(uz, -0.18f), 0.18f);
    }

    #pragma unroll
    for (int i = 0; i < 7; ++i) {
        float cu = 4.0f * ((float)ct_ex[i] * ux + (float)ct_ey[i] * uy + (float)ct_ez[i] * uz);
        float geq = ct_w[i] * T * (1.0f + cu);
        gg[i] = gg[i] - (gg[i] - geq) * inv_tau;
    }
}

__device__ __forceinline__ void load_vel_from_f(
    const float* __restrict__ f,
    size_t idx, size_t ncells, int flip_f,
    float& ux, float& uy, float& uz
) {
    float rho = 0.0f, vx = 0.0f, vy = 0.0f, vz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        int slot = flip_f ? c_opp[i] : i;
        float v = __ldg(&f[(size_t)slot * ncells + idx]);
        rho += v;
        vx  += (float)c_ex[i] * v;
        vy  += (float)c_ey[i] * v;
        vz  += (float)c_ez[i] * v;
    }
    float inv_rho = (rho > 1e-8f) ? (1.0f / rho) : 1.0f;
    ux = vx * inv_rho;
    uy = vy * inv_rho;
    uz = vz * inv_rho;
}

__global__ __launch_bounds__(256, 2)
void k_thermal_even(
    float* __restrict__ g,
    const float* __restrict__ f,
    const uint8_t* __restrict__ thermal_bc,
    int nx, int ny, int nz, size_t ncells,
    float t_hot, float tau_t_fluid, float tau_t_solid, int flip_f
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
    uint8_t bc = __ldg(&thermal_bc[idx]);

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (bc == 0) load_vel_from_f(f, idx, ncells, flip_f, ux, uy, uz);

    float gg[7];
    #pragma unroll
    for (int i = 0; i < 7; ++i) gg[i] = g[(size_t)i * ncells + idx];

    thermal_collide_d3q7(gg, bc, ux, uy, uz, t_hot, tau_t_fluid, tau_t_solid);

    #pragma unroll
    for (int i = 0; i < 7; ++i) g[(size_t)ct_opp[i] * ncells + idx] = gg[i];
}

__global__ __launch_bounds__(256, 2)
void k_thermal_odd(
    float* __restrict__ g,
    const float* __restrict__ f,
    const uint8_t* __restrict__ thermal_bc,
    int nx, int ny, int nz, size_t ncells,
    int periodic_y, float t_in, float t_hot, float tau_t_fluid, float tau_t_solid, int flip_f
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    size_t stride_z = (size_t)ny * (size_t)nx;
    size_t idx = (size_t)z * stride_z + (size_t)y * (size_t)nx + (size_t)x;
    uint8_t bc = __ldg(&thermal_bc[idx]);

    float ux = 0.0f, uy = 0.0f, uz = 0.0f;
    if (bc == 0) load_vel_from_f(f, idx, ncells, flip_f, ux, uy, uz);

    float gg[7];
    bool is_interior = (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1);
    if (is_interior) {
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            size_t nb_idx = (size_t)(z - ct_ez[i]) * stride_z
                          + (size_t)(y - ct_ey[i]) * (size_t)nx
                          + (size_t)(x - ct_ex[i]);
            gg[i] = g[(size_t)ct_opp[i] * ncells + nb_idx];
        }
        thermal_collide_d3q7(gg, bc, ux, uy, uz, t_hot, tau_t_fluid, tau_t_solid);
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            size_t dst_idx = (size_t)(z + ct_ez[i]) * stride_z
                           + (size_t)(y + ct_ey[i]) * (size_t)nx
                           + (size_t)(x + ct_ex[i]);
            g[(size_t)i * ncells + dst_idx] = gg[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            int xs = x - ct_ex[i];
            if (xs < 0) xs += nx; else if (xs >= nx) xs -= nx;
            int zs = z - ct_ez[i];
            if (zs < 0) zs += nz; else if (zs >= nz) zs -= nz;
            int ys = y - ct_ey[i];
            if (periodic_y) {
                if (ys < 0) ys += ny; else if (ys >= ny) ys -= ny;
            }
            if (ys < 0) {
                gg[i] = ct_w[i] * t_in;
            } else if (ys >= ny) {
                // 无竞争零梯度出口外推：用本地静止粒子 g[0, idx] 推算 T_local
                gg[i] = ct_w[i] * (4.0f * g[idx]);
            } else {
                size_t nb_idx = (size_t)zs * stride_z + (size_t)ys * (size_t)nx + (size_t)xs;
                gg[i] = g[(size_t)ct_opp[i] * ncells + nb_idx];
            }
        }
        thermal_collide_d3q7(gg, bc, ux, uy, uz, t_hot, tau_t_fluid, tau_t_solid);
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            int xd = x + ct_ex[i];
            if (xd < 0) xd += nx; else if (xd >= nx) xd -= nx;
            int zd = z + ct_ez[i];
            if (zd < 0) zd += nz; else if (zd >= nz) zd -= nz;
            int yd = y + ct_ey[i];
            if (periodic_y) {
                if (yd < 0) yd += ny; else if (yd >= ny) yd -= ny;
            }
            if (yd >= 0 && yd < ny) {
                size_t dst_idx = (size_t)zd * stride_z + (size_t)yd * (size_t)nx + (size_t)xd;
                g[(size_t)i * ncells + dst_idx] = gg[i];
            }
        }
    }
}

__global__ void k_apply_thermal_bc(
    float* __restrict__ g,
    const uint8_t* __restrict__ thermal_bc,
    float t_in,
    int nx, int ny, int nz, size_t ncells, int flip_slots
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || z >= nz) return;

    size_t stride_z = (size_t)ny * (size_t)nx;
    // y = 0: 入口冷风恒温 t_in（跳过恒温热源）
    size_t idx0 = (size_t)z * stride_z + (size_t)x;
    if (__ldg(&thermal_bc[idx0]) == 0) {
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            int slot = flip_slots ? ct_opp[i] : i;
            g[(size_t)slot * ncells + idx0] = ct_w[i] * t_in;
        }
    }
    // y = ny - 1: 出口绝热零梯度外推
    size_t idx_up  = (size_t)z * stride_z + (size_t)(ny - 2) * (size_t)nx + (size_t)x;
    size_t idx_out = (size_t)z * stride_z + (size_t)(ny - 1) * (size_t)nx + (size_t)x;
    if (__ldg(&thermal_bc[idx_out]) == 0) {
        float T_up = 0.0f;
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            int slot = flip_slots ? ct_opp[i] : i;
            T_up += g[(size_t)slot * ncells + idx_up];
        }
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            int slot = flip_slots ? ct_opp[i] : i;
            g[(size_t)slot * ncells + idx_out] = ct_w[i] * T_up;
        }
    }
}

__global__ void k_extract_y_temp_plane(
    const float* __restrict__ g,
    float* __restrict__ out_temp,
    int y, int nx, int ny, int nz, size_t ncells
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || z >= nz) return;
    size_t plane_idx = (size_t)z * (size_t)nx + (size_t)x;
    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
    float T = 0.0f;
    #pragma unroll
    for (int i = 0; i < 7; ++i) T += __ldg(&g[(size_t)i * ncells + idx]);
    out_temp[plane_idx] = T;
}

__global__ void k_extract_z_temp_slice(
    const float* __restrict__ g,
    float* __restrict__ out_temp,
    int z, int nx, int ny, int nz, size_t ncells
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    size_t slice_idx = (size_t)y * (size_t)nx + (size_t)x;
    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + slice_idx;
    float T = 0.0f;
    #pragma unroll
    for (int i = 0; i < 7; ++i) T += __ldg(&g[(size_t)i * ncells + idx]);
    out_temp[slice_idx] = T;
}

__global__ void k_extract_x_temp_slice(
    const float* __restrict__ g,
    float* __restrict__ out_temp,
    int x, int nx, int ny, int nz, size_t ncells
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= ny || z >= nz) return;
    size_t slice_idx = (size_t)z * (size_t)ny + (size_t)y;
    size_t idx = (size_t)z * (size_t)ny * (size_t)nx + (size_t)y * (size_t)nx + (size_t)x;
    float T = 0.0f;
    #pragma unroll
    for (int i = 0; i < 7; ++i) T += __ldg(&g[(size_t)i * ncells + idx]);
    out_temp[slice_idx] = T;
}

// ============================================================================
// 辅助函数：单步发射 & CUDA Graph 构建
// ============================================================================
static void enqueue_one_step(LBMHandle* h, int step_parity) {
    dim3 blk3(32, 4, 2);
    dim3 grd3((h->nx + blk3.x - 1) / blk3.x,
              (h->ny + blk3.y - 1) / blk3.y,
              (h->nz + blk3.z - 1) / blk3.z);
    dim3 blk2(32, 8);
    dim3 grd2((h->nx + blk2.x - 1) / blk2.x,
              (h->nz + blk2.y - 1) / blk2.y);

    if (step_parity == 0) {
        k_lbm_even<<<grd3, blk3, 0, h->stream>>>(
            h->d_f, h->d_solid, h->tau0, h->nu0, h->cs2,
            h->nx, h->ny, h->nz, h->ncells, h->gy, h->tau_min, h->tau_max);
        if (!h->periodic_y) {
            k_apply_bc<<<grd2, blk2, 0, h->stream>>>(
                h->d_f, h->d_solid, h->u_in, h->nx, h->ny, h->nz, h->ncells, 1);
        }
        if (h->thermal_enabled) {
            k_thermal_even<<<grd3, blk3, 0, h->stream>>>(
                h->d_g, h->d_f, h->d_thermal_bc,
                h->nx, h->ny, h->nz, h->ncells,
                h->t_hot, h->tau_t_fluid, h->tau_t_solid, 1);
            if (!h->periodic_y) {
                k_apply_thermal_bc<<<grd2, blk2, 0, h->stream>>>(
                    h->d_g, h->d_thermal_bc, h->t_in, h->nx, h->ny, h->nz, h->ncells, 1);
            }
        }
    } else {
        k_lbm_odd<<<grd3, blk3, 0, h->stream>>>(
            h->d_f, h->d_solid, h->tau0, h->nu0, h->cs2,
            h->nx, h->ny, h->nz, h->ncells, h->u_in, h->gy, h->periodic_y, h->tau_min, h->tau_max);
        if (!h->periodic_y) {
            k_apply_bc<<<grd2, blk2, 0, h->stream>>>(
                h->d_f, h->d_solid, h->u_in, h->nx, h->ny, h->nz, h->ncells, 0);
        }
        if (h->thermal_enabled) {
            k_thermal_odd<<<grd3, blk3, 0, h->stream>>>(
                h->d_g, h->d_f, h->d_thermal_bc,
                h->nx, h->ny, h->nz, h->ncells,
                h->periodic_y, h->t_in, h->t_hot, h->tau_t_fluid, h->tau_t_solid, 0);
            if (!h->periodic_y) {
                k_apply_thermal_bc<<<grd2, blk2, 0, h->stream>>>(
                    h->d_g, h->d_thermal_bc, h->t_in, h->nx, h->ny, h->nz, h->ncells, 0);
            }
        }
    }
}

static int ensure_cuda_graph(LBMHandle* h) {
    if (h->graph_ready) return 0;
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(h->stream, cudaStreamCaptureModeGlobal));
    for (int p = 0; p < GRAPH_PAIR_STEPS; ++p) {
        enqueue_one_step(h, 0);
        enqueue_one_step(h, 1);
    }
    CUDA_CHECK(cudaStreamEndCapture(h->stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&h->graph_exec, graph, nullptr, nullptr, 0));
    cudaGraphDestroy(graph);
    h->graph_ready = 1;
    return 0;
}

// ============================================================================
// C API 导出接口
// ============================================================================
LBM_API int lbm_device_info(char* name_buf, int buf_len, int* cc_major, int* cc_minor, size_t* total_mem_mb) {
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    if (name_buf && buf_len > 0) snprintf(name_buf, (size_t)buf_len, "%s", prop.name);
    if (cc_major) *cc_major = prop.major;
    if (cc_minor) *cc_minor = prop.minor;
    if (total_mem_mb) *total_mem_mb = prop.totalGlobalMem / (1024 * 1024);
    return 0;
}

LBM_API void* lbm_create(
    const uint8_t* h_solid,
    int nz, int ny, int nx,
    float u_in, float tau0, float cs,
    float body_force, int periodic_y,
    float tau_min, float tau_max
) {
    LBMHandle* h = (LBMHandle*)calloc(1, sizeof(LBMHandle));
    if (!h) return nullptr;

    h->nx = nx;
    h->ny = ny;
    h->nz = nz;
    h->ncells = (size_t)nx * (size_t)ny * (size_t)nz;
    h->u_in = u_in;
    h->tau0 = tau0;
    h->nu0 = (tau0 - 0.5f) / 3.0f;
    h->cs2 = cs * cs;
    h->gy = body_force;
    h->periodic_y = periodic_y ? 1 : 0;
    h->tau_min = tau_min;
    h->tau_max = tau_max;
    h->parity = 0;
    h->graph_ready = 0;

    size_t f_bytes = 19ULL * h->ncells * sizeof(float);
    size_t s_bytes = h->ncells * sizeof(uint8_t);

    if (cudaStreamCreate(&h->stream) != cudaSuccess ||
        cudaMalloc(&h->d_f, f_bytes) != cudaSuccess ||
        cudaMalloc(&h->d_solid, s_bytes) != cudaSuccess) {
        fprintf(stderr, "[lbm_cuda] GPU 显存分配失败 (需求 %.1f MB)\n",
                (double)(f_bytes + s_bytes) / 1e6);
        if (h->d_f) cudaFree(h->d_f);
        if (h->d_solid) cudaFree(h->d_solid);
        if (h->stream) cudaStreamDestroy(h->stream);
        free(h);
        return nullptr;
    }

    cudaMemcpyAsync(h->d_solid, h_solid, s_bytes, cudaMemcpyHostToDevice, h->stream);

    dim3 blk(32, 4, 2);
    dim3 grd((nx + blk.x - 1) / blk.x,
             (ny + blk.y - 1) / blk.y,
             (nz + blk.z - 1) / blk.z);
    k_init_field<<<grd, blk, 0, h->stream>>>(h->d_f, nx, ny, nz, h->ncells, u_in);
    cudaStreamSynchronize(h->stream);
    return (void*)h;
}

LBM_API int lbm_run_steps(void* handle, int nsteps) {
    if (!handle || nsteps <= 0) return -1;
    LBMHandle* h = (LBMHandle*)handle;

    int rem = nsteps;
    if (h->parity == 1 && rem > 0) {
        enqueue_one_step(h, 1);
        h->parity = 0;
        rem--;
    }

    int batch = 2 * GRAPH_PAIR_STEPS;
    if (rem >= batch) {
        if (ensure_cuda_graph(h) != 0) return -1;
        while (rem >= batch) {
            CUDA_CHECK(cudaGraphLaunch(h->graph_exec, h->stream));
            rem -= batch;
        }
    }

    while (rem > 0) {
        enqueue_one_step(h, h->parity);
        h->parity ^= 1;
        rem--;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    return 0;
}

LBM_API int lbm_get_y_plane(void* handle, int y, float* h_rho, float* h_uy) {
    if (!handle) return -1;
    LBMHandle* h = (LBMHandle*)handle;
    if (y < 0 || y >= h->ny) return -1;

    size_t plane_bytes = (size_t)h->nz * (size_t)h->nx * sizeof(float);
    float *d_rho = nullptr, *d_uy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rho, plane_bytes));
    CUDA_CHECK(cudaMalloc(&d_uy, plane_bytes));

    dim3 blk(32, 8);
    dim3 grd((h->nx + blk.x - 1) / blk.x, (h->nz + blk.y - 1) / blk.y);
    k_extract_y_plane<<<grd, blk, 0, h->stream>>>(
        h->d_f, h->d_solid, d_rho, d_uy, y, h->nx, h->ny, h->nz, h->ncells, h->parity);

    CUDA_CHECK(cudaMemcpyAsync(h_rho, d_rho, plane_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(h_uy, d_uy, plane_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));

    cudaFree(d_rho);
    cudaFree(d_uy);
    return 0;
}

LBM_API int lbm_get_z_slice(void* handle, int z, float* h_rho, float* h_uy) {
    if (!handle) return -1;
    LBMHandle* h = (LBMHandle*)handle;
    if (z < 0 || z >= h->nz) return -1;

    size_t slice_bytes = (size_t)h->ny * (size_t)h->nx * sizeof(float);
    float *d_rho = nullptr, *d_uy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rho, slice_bytes));
    CUDA_CHECK(cudaMalloc(&d_uy, slice_bytes));

    dim3 blk(32, 8);
    dim3 grd((h->nx + blk.x - 1) / blk.x, (h->ny + blk.y - 1) / blk.y);
    k_extract_z_slice<<<grd, blk, 0, h->stream>>>(
        h->d_f, h->d_solid, d_rho, d_uy, z, h->nx, h->ny, h->nz, h->ncells, h->parity);

    CUDA_CHECK(cudaMemcpyAsync(h_rho, d_rho, slice_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(h_uy, d_uy, slice_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));

    cudaFree(d_rho);
    cudaFree(d_uy);
    return 0;
}

LBM_API int lbm_check_health(void* handle, int* out_nan_count, float* out_rho_min, float* out_rho_max) {
    if (!handle) return -1;
    LBMHandle* h = (LBMHandle*)handle;
    size_t plane_n = (size_t)h->nz * (size_t)h->nx;
    float* h_rho = (float*)malloc(plane_n * sizeof(float));
    float* h_uy  = (float*)malloc(plane_n * sizeof(float));
    if (!h_rho || !h_uy) {
        free(h_rho);
        free(h_uy);
        return -1;
    }

    int nans = 0;
    float rmin = 1e30f, rmax = -1e30f;
    for (int y = 0; y < h->ny; ++y) {
        if (lbm_get_y_plane(handle, y, h_rho, h_uy) != 0) {
            free(h_rho);
            free(h_uy);
            return -1;
        }
        for (size_t k = 0; k < plane_n; ++k) {
            float r = h_rho[k];
            if (isnan(r) || isinf(r)) {
                nans++;
            } else if (r > 0.0f) {
                if (r < rmin) rmin = r;
                if (r > rmax) rmax = r;
            }
        }
    }
    free(h_rho);
    free(h_uy);
    if (out_nan_count) *out_nan_count = nans;
    if (out_rho_min) *out_rho_min = rmin;
    if (out_rho_max) *out_rho_max = rmax;
    return 0;
}

LBM_API int lbm_enable_thermal(
    void* handle,
    const uint8_t* h_thermal_bc,
    float t_in, float t_hot,
    float alpha_fluid, float alpha_solid
) {
    if (!handle || !h_thermal_bc) return -1;
    LBMHandle* h = (LBMHandle*)handle;

    // 若先前录制过纯流场 Graph，失效重建
    if (h->graph_ready && h->graph_exec) {
        cudaGraphExecDestroy(h->graph_exec);
        h->graph_exec = nullptr;
        h->graph_ready = 0;
    }

    size_t g_bytes = 7ULL * h->ncells * sizeof(float);
    size_t bc_bytes = h->ncells * sizeof(uint8_t);
    if (!h->d_g) CUDA_CHECK(cudaMalloc(&h->d_g, g_bytes));
    if (!h->d_thermal_bc) CUDA_CHECK(cudaMalloc(&h->d_thermal_bc, bc_bytes));

    CUDA_CHECK(cudaMemcpyAsync(h->d_thermal_bc, h_thermal_bc, bc_bytes, cudaMemcpyHostToDevice, h->stream));

    h->t_in = t_in;
    h->t_hot = t_hot;
    h->tau_t_fluid = fmaxf(0.502f, 0.5f + 4.0f * alpha_fluid);
    h->tau_t_solid = fminf(1.98f, fmaxf(0.505f, 0.5f + 4.0f * alpha_solid));
    h->thermal_enabled = 1;

    dim3 blk(32, 4, 2);
    dim3 grd((h->nx + blk.x - 1) / blk.x,
             (h->ny + blk.y - 1) / blk.y,
             (h->nz + blk.z - 1) / blk.z);
    k_init_thermal<<<grd, blk, 0, h->stream>>>(
        h->d_g, h->d_thermal_bc, h->nx, h->ny, h->nz, h->ncells, t_in, t_hot);
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    return 0;
}

LBM_API int lbm_get_y_temp_plane(void* handle, int y, float* h_temp) {
    if (!handle || !h_temp) return -1;
    LBMHandle* h = (LBMHandle*)handle;
    if (!h->thermal_enabled || y < 0 || y >= h->ny) return -1;

    size_t plane_bytes = (size_t)h->nz * (size_t)h->nx * sizeof(float);
    float* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, plane_bytes));

    dim3 blk(32, 8);
    dim3 grd((h->nx + blk.x - 1) / blk.x, (h->nz + blk.y - 1) / blk.y);
    k_extract_y_temp_plane<<<grd, blk, 0, h->stream>>>(
        h->d_g, d_temp, y, h->nx, h->ny, h->nz, h->ncells);
    CUDA_CHECK(cudaMemcpyAsync(h_temp, d_temp, plane_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    cudaFree(d_temp);
    return 0;
}

LBM_API int lbm_get_z_temp_slice(void* handle, int z, float* h_temp) {
    if (!handle || !h_temp) return -1;
    LBMHandle* h = (LBMHandle*)handle;
    if (!h->thermal_enabled || z < 0 || z >= h->nz) return -1;

    size_t slice_bytes = (size_t)h->ny * (size_t)h->nx * sizeof(float);
    float* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, slice_bytes));

    dim3 blk(32, 8);
    dim3 grd((h->nx + blk.x - 1) / blk.x, (h->ny + blk.y - 1) / blk.y);
    k_extract_z_temp_slice<<<grd, blk, 0, h->stream>>>(
        h->d_g, d_temp, z, h->nx, h->ny, h->nz, h->ncells);
    CUDA_CHECK(cudaMemcpyAsync(h_temp, d_temp, slice_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    cudaFree(d_temp);
    return 0;
}

LBM_API int lbm_get_x_slice(void* handle, int x, float* h_rho, float* h_uy) {
    if (!handle) return -1;
    LBMHandle* h = (LBMHandle*)handle;
    if (x < 0 || x >= h->nx) return -1;

    size_t slice_bytes = (size_t)h->nz * (size_t)h->ny * sizeof(float);
    float *d_rho = nullptr, *d_uy = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rho, slice_bytes));
    CUDA_CHECK(cudaMalloc(&d_uy, slice_bytes));

    dim3 blk(32, 8);
    dim3 grd((h->ny + blk.x - 1) / blk.x, (h->nz + blk.y - 1) / blk.y);
    k_extract_x_slice<<<grd, blk, 0, h->stream>>>(
        h->d_f, h->d_solid, d_rho, d_uy, x, h->nx, h->ny, h->nz, h->ncells, h->parity);

    CUDA_CHECK(cudaMemcpyAsync(h_rho, d_rho, slice_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaMemcpyAsync(h_uy, d_uy, slice_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    cudaFree(d_rho);
    cudaFree(d_uy);
    return 0;
}

LBM_API int lbm_get_x_temp_slice(void* handle, int x, float* h_temp) {
    if (!handle || !h_temp) return -1;
    LBMHandle* h = (LBMHandle*)handle;
    if (!h->thermal_enabled || x < 0 || x >= h->nx) return -1;

    size_t slice_bytes = (size_t)h->nz * (size_t)h->ny * sizeof(float);
    float* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_temp, slice_bytes));

    dim3 blk(32, 8);
    dim3 grd((h->ny + blk.x - 1) / blk.x, (h->nz + blk.y - 1) / blk.y);
    k_extract_x_temp_slice<<<grd, blk, 0, h->stream>>>(
        h->d_g, d_temp, x, h->nx, h->ny, h->nz, h->ncells);
    CUDA_CHECK(cudaMemcpyAsync(h_temp, d_temp, slice_bytes, cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));
    cudaFree(d_temp);
    return 0;
}

// ----------------------------------------------------------------------------
// CUB BlockReduce + double 精度零临时显存归约 & Z 分块三维场导出 Kernel
// ----------------------------------------------------------------------------
struct MaxFloatOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
        if (assumed == old) break;
    }
}

__global__ void k_reduce_metrics_fused(
    const float* __restrict__ f,
    const uint8_t* __restrict__ solid,
    const float* __restrict__ g,
    const uint8_t* __restrict__ thermal_bc,
    int nx, int ny, size_t ncells,
    int y_in, int y_out, int flip_slots, int thermal_enabled,
    DeviceMetrics* __restrict__ out
) {
    constexpr int BLK = 256;
    using BlockReduceD = cub::BlockReduce<double, BLK>;
    using BlockReduceU = cub::BlockReduce<unsigned long long, BLK>;
    using BlockReduceF = cub::BlockReduce<float, BLK>;

    __shared__ union {
        typename BlockReduceD::TempStorage d;
        typename BlockReduceU::TempStorage u;
        typename BlockReduceF::TempStorage f;
    } temp_storage;

    double loc_uy = 0.0, loc_ri = 0.0, loc_ro = 0.0, loc_to = 0.0;
    unsigned long long loc_cf = 0, loc_ci = 0, loc_co = 0;
    float loc_ts = -1e30f;

    for (size_t idx = (size_t)blockIdx.x * BLK + threadIdx.x;
         idx < ncells;
         idx += (size_t)gridDim.x * BLK) {
        int y = (int)((idx / (size_t)nx) % (size_t)ny);
        uint8_t s = __ldg(&solid[idx]);
        if (!s) {
            float rho = 0.0f, uy = 0.0f;
            #pragma unroll
            for (int i = 0; i < 19; ++i) {
                int slot = flip_slots ? c_opp[i] : i;
                float v = __ldg(&f[(size_t)slot * ncells + idx]);
                rho += v;
                uy += (float)c_ey[i] * v;
            }
            loc_uy += (rho > 1e-8f) ? (double)(uy / rho) : 0.0;
            loc_cf += 1;
            if (y == y_in)  { loc_ri += (double)rho; loc_ci += 1; }
            if (y == y_out) { loc_ro += (double)rho; loc_co += 1; }
        }
        if (thermal_enabled) {
            float T = 0.0f;
            #pragma unroll
            for (int i = 0; i < 7; ++i) T += __ldg(&g[(size_t)i * ncells + idx]);
            if (!s && y == y_out) loc_to += (double)T;
            uint8_t bc = __ldg(&thermal_bc[idx]);
            if (bc == 1 || bc == 3) loc_ts = fmaxf(loc_ts, T);
        }
    }

    double b_uy = BlockReduceD(temp_storage.d).Sum(loc_uy); __syncthreads();
    double b_ri = BlockReduceD(temp_storage.d).Sum(loc_ri); __syncthreads();
    double b_ro = BlockReduceD(temp_storage.d).Sum(loc_ro); __syncthreads();
    double b_to = BlockReduceD(temp_storage.d).Sum(loc_to); __syncthreads();
    unsigned long long b_cf = BlockReduceU(temp_storage.u).Sum(loc_cf); __syncthreads();
    unsigned long long b_ci = BlockReduceU(temp_storage.u).Sum(loc_ci); __syncthreads();
    unsigned long long b_co = BlockReduceU(temp_storage.u).Sum(loc_co); __syncthreads();
    float b_ts = BlockReduceF(temp_storage.f).Reduce(loc_ts, MaxFloatOp());

    if (threadIdx.x == 0) {
        atomicAdd(&out->sum_uy, b_uy);
        atomicAdd(&out->sum_rho_in, b_ri);
        atomicAdd(&out->sum_rho_out, b_ro);
        atomicAdd(&out->sum_t_out, b_to);
        atomicAdd(&out->cnt_fluid, b_cf);
        atomicAdd(&out->cnt_in, b_ci);
        atomicAdd(&out->cnt_out, b_co);
        if (thermal_enabled && b_ts > -1e20f) atomicMaxFloat(&out->max_t_solid, b_ts);
    }
}

__global__ void k_extract_fields_chunk(
    const float* __restrict__ f,
    const uint8_t* __restrict__ solid,
    const float* __restrict__ g,
    size_t ncells, size_t offset, size_t count,
    int flip_slots, int thermal_enabled,
    float* __restrict__ out_ux,
    float* __restrict__ out_uy,
    float* __restrict__ out_uz,
    float* __restrict__ out_rho,
    float* __restrict__ out_temp
) {
    size_t local_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= count) return;
    size_t idx = offset + local_idx;

    if (__ldg(&solid[idx])) {
        if (out_ux)  out_ux[local_idx]  = 0.0f;
        if (out_uy)  out_uy[local_idx]  = 0.0f;
        if (out_uz)  out_uz[local_idx]  = 0.0f;
        if (out_rho) out_rho[local_idx] = 1.0f;
    } else {
        float rho = 0.0f, vx = 0.0f, vy = 0.0f, vz = 0.0f;
        #pragma unroll
        for (int i = 0; i < 19; ++i) {
            int slot = flip_slots ? c_opp[i] : i;
            float v = __ldg(&f[(size_t)slot * ncells + idx]);
            rho += v;
            vx += (float)c_ex[i] * v;
            vy += (float)c_ey[i] * v;
            vz += (float)c_ez[i] * v;
        }
        float inv_rho = (rho > 1e-8f) ? (1.0f / rho) : 1.0f;
        if (out_ux)  out_ux[local_idx]  = vx * inv_rho;
        if (out_uy)  out_uy[local_idx]  = vy * inv_rho;
        if (out_uz)  out_uz[local_idx]  = vz * inv_rho;
        if (out_rho) out_rho[local_idx] = rho;
    }
    if (thermal_enabled && out_temp) {
        float T = 0.0f;
        #pragma unroll
        for (int i = 0; i < 7; ++i) T += __ldg(&g[(size_t)i * ncells + idx]);
        out_temp[local_idx] = T;
    }
}

LBM_API int lbm_compute_metrics(
    void* handle, int y_in, int y_out,
    float* out_mean_uy, float* out_rho_in, float* out_rho_out,
    float* out_t_out, float* out_t_max_solid
) {
    if (!handle) return -1;
    LBMHandle* h = (LBMHandle*)handle;

    if (!h->d_metrics) {
        CUDA_CHECK(cudaMalloc(&h->d_metrics, sizeof(DeviceMetrics)));
    }

    DeviceMetrics h_init = {0.0, 0.0, 0.0, 0.0, 0ULL, 0ULL, 0ULL, -1e30f};
    CUDA_CHECK(cudaMemcpyAsync(h->d_metrics, &h_init, sizeof(DeviceMetrics), cudaMemcpyHostToDevice, h->stream));

    int grid = (int)((h->ncells + 255) / 256);
    if (grid > 1024) grid = 1024;
    k_reduce_metrics_fused<<<grid, 256, 0, h->stream>>>(
        h->d_f, h->d_solid, h->d_g, h->d_thermal_bc,
        h->nx, h->ny, h->ncells, y_in, y_out, h->parity, h->thermal_enabled,
        h->d_metrics
    );
    CUDA_CHECK(cudaGetLastError());

    DeviceMetrics h_res;
    CUDA_CHECK(cudaMemcpyAsync(&h_res, h->d_metrics, sizeof(DeviceMetrics), cudaMemcpyDeviceToHost, h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));

    if (out_mean_uy)     *out_mean_uy     = h_res.cnt_fluid ? (float)(h_res.sum_uy / (double)h_res.cnt_fluid) : 0.0f;
    if (out_rho_in)      *out_rho_in      = h_res.cnt_in    ? (float)(h_res.sum_rho_in / (double)h_res.cnt_in) : 1.0f;
    if (out_rho_out)     *out_rho_out     = h_res.cnt_out   ? (float)(h_res.sum_rho_out / (double)h_res.cnt_out) : 1.0f;
    if (out_t_out)       *out_t_out       = h_res.cnt_out   ? (float)(h_res.sum_t_out / (double)h_res.cnt_out) : 0.0f;
    if (out_t_max_solid) *out_t_max_solid = (h_res.max_t_solid > -1e20f) ? h_res.max_t_solid : h->t_in;
    return 0;
}

LBM_API int lbm_extract_fields_3d(
    void* handle,
    float* h_ux, float* h_uy, float* h_uz, float* h_rho, float* h_temp
) {
    if (!handle) return -1;
    LBMHandle* h = (LBMHandle*)handle;

    // 按 Z 切片分块提取（每次 16 层，1 亿网格仅占 < 20 MB 额外显存），统一 cleanup 防止泄漏
    const int chunk_z = 16;
    size_t max_chunk = (size_t)chunk_z * (size_t)h->ny * (size_t)h->nx;
    if (max_chunk > h->ncells) max_chunk = h->ncells;
    size_t chunk_bytes = max_chunk * sizeof(float);

    float *d_ux = nullptr, *d_uy = nullptr, *d_uz = nullptr, *d_rho = nullptr, *d_temp = nullptr;
    int rc = 0;

    if (h_ux  && cudaMalloc(&d_ux, chunk_bytes) != cudaSuccess) { rc = -1; goto cleanup; }
    if (h_uy  && cudaMalloc(&d_uy, chunk_bytes) != cudaSuccess) { rc = -1; goto cleanup; }
    if (h_uz  && cudaMalloc(&d_uz, chunk_bytes) != cudaSuccess) { rc = -1; goto cleanup; }
    if (h_rho && cudaMalloc(&d_rho, chunk_bytes) != cudaSuccess) { rc = -1; goto cleanup; }
    if (h->thermal_enabled && h_temp && cudaMalloc(&d_temp, chunk_bytes) != cudaSuccess) { rc = -1; goto cleanup; }

    for (size_t offset = 0; offset < h->ncells; offset += max_chunk) {
        size_t count = h->ncells - offset;
        if (count > max_chunk) count = max_chunk;
        size_t cbytes = count * sizeof(float);

        int blk = 256;
        int grd = (int)((count + blk - 1) / blk);
        k_extract_fields_chunk<<<grd, blk, 0, h->stream>>>(
            h->d_f, h->d_solid, h->d_g, h->ncells, offset, count,
            h->parity, h->thermal_enabled, d_ux, d_uy, d_uz, d_rho, d_temp
        );
        if (cudaGetLastError() != cudaSuccess) { rc = -1; goto cleanup; }

        if (d_ux   && cudaMemcpyAsync(h_ux + offset, d_ux, cbytes, cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) { rc = -1; goto cleanup; }
        if (d_uy   && cudaMemcpyAsync(h_uy + offset, d_uy, cbytes, cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) { rc = -1; goto cleanup; }
        if (d_uz   && cudaMemcpyAsync(h_uz + offset, d_uz, cbytes, cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) { rc = -1; goto cleanup; }
        if (d_rho  && cudaMemcpyAsync(h_rho + offset, d_rho, cbytes, cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) { rc = -1; goto cleanup; }
        if (d_temp && cudaMemcpyAsync(h_temp + offset, d_temp, cbytes, cudaMemcpyDeviceToHost, h->stream) != cudaSuccess) { rc = -1; goto cleanup; }
    }
    if (cudaStreamSynchronize(h->stream) != cudaSuccess) rc = -1;

cleanup:
    if (d_ux)   cudaFree(d_ux);
    if (d_uy)   cudaFree(d_uy);
    if (d_uz)   cudaFree(d_uz);
    if (d_rho)  cudaFree(d_rho);
    if (d_temp) cudaFree(d_temp);
    return rc;
}

LBM_API void lbm_destroy(void* handle) {
    if (!handle) return;
    LBMHandle* h = (LBMHandle*)handle;
    if (h->graph_ready && h->graph_exec) cudaGraphExecDestroy(h->graph_exec);
    if (h->d_f) cudaFree(h->d_f);
    if (h->d_solid) cudaFree(h->d_solid);
    if (h->d_g) cudaFree(h->d_g);
    if (h->d_thermal_bc) cudaFree(h->d_thermal_bc);
    if (h->d_metrics) cudaFree(h->d_metrics);
    if (h->stream) cudaStreamDestroy(h->stream);
    free(h);
}
