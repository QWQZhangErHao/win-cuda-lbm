// ============================================================================
// lbm_cuda.cu —— Windows 原生 CUDA C++ 高性能 D3Q19 LBM 求解器
// 集成四大工程级优化（全 FP32 精度零损失）：
//   1. AA-Pattern 单数组原地演进（消除双缓冲，显存减半：76 B/体素，26.19M 仅需 1.99 GB）
//   2. Even 步 100% 本地合并读写 + Odd 步内部体素无取模/无分支快路径
//   3. CUDA Graph 批量打包提交（每次 20 步单次 Launch，大幅削减 WDDM 驱动开销）
//   4. Warp 32×4×2 合并访存 + __constant__ 常量展开 + --use_fast_math
// ============================================================================

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
    uint8_t* d_thermal_bc;  // 0=fluid, 1=solid(conduction), 2=isothermal wall
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
// 公共碰撞内联函数（Smagorinsky LES + BGK）
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

    float tau = tau0;
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
            float s_mag = 1.5f * sqrtf(2.0f * q2) * inv_rho / tau0;
            float nu_t = cs2 * s_mag;
            tau = 0.5f + 3.0f * (nu0 + nu_t);
            tau = fminf(fmaxf(tau, tau_min), tau_max);
        }
    }

    float inv_tau = 1.0f / tau;
    float u2_term = 1.5f * (ux * ux + uy * uy + uz * uz);
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        float cu = 3.0f * ((float)c_ex[i] * ux + (float)c_ey[i] * uy + (float)c_ez[i] * uz);
        float feq = c_w[i] * rho * (1.0f + cu + 0.5f * cu * cu - u2_term);
        ff[i] = ff[i] - (ff[i] - feq) * inv_tau + 3.0f * c_w[i] * (float)c_ey[i] * gy;
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
// 合并的入口 (y=0) + 出口 (y=ny-1) 边界 Kernel
// ----------------------------------------------------------------------------
__global__ void k_apply_bc(
    float* __restrict__ f,
    float u_in,
    int nx, int ny, int nz, size_t ncells,
    int flip_slots
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || z >= nz) return;

    size_t stride_z = (size_t)ny * (size_t)nx;

    // 1. 速度入口 y = 0
    size_t idx0 = (size_t)z * stride_z + (size_t)x;
    size_t idx1 = idx0 + (size_t)nx;
    float rho_in = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        rho_in += f[(size_t)i * ncells + idx1];
    }
    float u2_in = 1.5f * u_in * u_in;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        int slot = flip_slots ? c_opp[i] : i;
        float cu = 3.0f * (float)c_ey[i] * u_in;
        f[(size_t)slot * ncells + idx0] = c_w[i] * rho_in * (1.0f + cu + 0.5f * cu * cu - u2_in);
    }

    // 2. 压力出口 y = ny - 1
    size_t idx_up  = (size_t)z * stride_z + (size_t)(ny - 2) * (size_t)nx + (size_t)x;
    size_t idx_out = (size_t)z * stride_z + (size_t)(ny - 1) * (size_t)nx + (size_t)x;
    float rho_up = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        int slot = flip_slots ? c_opp[i] : i;
        float v = f[(size_t)slot * ncells + idx_up];
        rho_up += v;
        ux += (float)c_ex[i] * v;
        uy += (float)c_ey[i] * v;
        uz += (float)c_ez[i] * v;
    }
    float inv_rho = 1.0f / rho_up;
    ux *= inv_rho;
    uy *= inv_rho;
    uz *= inv_rho;
    float u2_out = 1.5f * (ux * ux + uy * uy + uz * uz);
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        int slot = flip_slots ? c_opp[i] : i;
        float cu = 3.0f * ((float)c_ex[i] * ux + (float)c_ey[i] * uy + (float)c_ez[i] * uz);
        f[(size_t)slot * ncells + idx_out] = c_w[i] * (1.0f + cu + 0.5f * cu * cu - u2_out);
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
        // 恒温壁面节点
        #pragma unroll
        for (int i = 0; i < 7; ++i) {
            gg[i] = ct_w[i] * t_hot;
        }
        return;
    }
    float T = 0.0f;
    #pragma unroll
    for (int i = 0; i < 7; ++i) T += gg[i];

    float inv_tau = (bc == 1) ? (1.0f / tau_t_solid) : (1.0f / fmaxf(tau_t_fluid, 0.85f));
    if (bc == 1) {
        ux = 0.0f; uy = 0.0f; uz = 0.0f;  // 固体导热区域
    } else {
        // D3Q7 正定性速度限制
        ux = fminf(fmaxf(ux, -0.20f), 0.20f);
        uy = fminf(fmaxf(uy, -0.20f), 0.20f);
        uz = fminf(fmaxf(uz, -0.20f), 0.20f);
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
                gg[i] = ct_w[i] * t_in;
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
                h->d_f, h->u_in, h->nx, h->ny, h->nz, h->ncells, 1);
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
                h->d_f, h->u_in, h->nx, h->ny, h->nz, h->ncells, 0);
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

LBM_API void lbm_destroy(void* handle) {
    if (!handle) return;
    LBMHandle* h = (LBMHandle*)handle;
    if (h->graph_ready && h->graph_exec) cudaGraphExecDestroy(h->graph_exec);
    if (h->d_f) cudaFree(h->d_f);
    if (h->d_solid) cudaFree(h->d_solid);
    if (h->d_g) cudaFree(h->d_g);
    if (h->d_thermal_bc) cudaFree(h->d_thermal_bc);
    if (h->stream) cudaStreamDestroy(h->stream);
    free(h);
}
