# win-cuda-lbm

Windows native CUDA C++ D3Q19/D3Q7 Lattice Boltzmann solver (BGK + Smagorinsky LES + Conjugate Heat Transfer), compiled directly with `nvcc` + MSVC into a standalone DLL (`lbm_cuda.dll`) with zero Docker / WSL / Linux dependency.

## Features

- **D3Q19 Fluid**: BGK + Smagorinsky SGS turbulence (`Cs`), half-way bounce-back, periodic x/z, velocity inlet / pressure outlet along y.
- **D3Q7 Thermal (Optional)**: Conjugate heat transfer (fluid advection-diffusion, solid conduction, isothermal walls).
- **AA-Pattern In-Place**: Single distribution array (`76 B/voxel` for flow), fitting 120M+ voxels on a 12 GB GPU.
- **CUDA Graph Batching**: 20-step graph launch to minimize Windows WDDM kernel dispatch overhead.
- **Zero-Dependency Python Wrapper**: `ctypes` binding in `lbm_cuda.py` (requires `numpy` only).

## Requirements

- Windows 10/11 x64
- NVIDIA CUDA Toolkit (CUDA 12.8+ for RTX 50-series `sm_120`)
- Visual Studio 2022 / C++ Build Tools
- Python 3.10+ (`numpy`)

## Build & Verify

```powershell
powershell -ExecutionPolicy Bypass -File .\build.ps1
python .\lbm_cuda.py
```

## Benchmark (RTX 5070 Ti Laptop 12GB, `sm_120`)

| Case | Grid | Throughput | Notes |
|---|---|---|---|
| Poiseuille channel | `4 x 32 x 32` | ~348 MLUPS | Error `-0.09%` vs analytical |
| 3D finned heat sink (flow) | `300 x 225 x 388` (26.19M) | ~2311 MLUPS | `1990 MB` VRAM |
| 3D finned heat sink (flow + D3Q7 CHT) | `300 x 225 x 388` (26.19M) | ~1302 MLUPS | Coupled convection + conduction |

## License

MIT
