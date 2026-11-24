# -*- coding: utf-8 -*-
"""
3D Finned Heat Sink with Staggered Heat Pipes (D3Q19 Flow + D3Q7 Conjugate Heat Transfer)

Usage:
    python examples/finned_heatsink.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lbm_cuda import LBMCudaSolver


def build_heatsink(pitch_mm=0.4, pad_y=20.0):
    base_w, base_d, base_h = 40.0, 40.0, 10.0
    tower_w, tower_d, total_h = 120.0, 50.0, 155.0
    fin_n, fin_t, fin_pitch = 58, 0.3, 2.488
    fin_z0 = base_h + 4.0
    pipe_n, pipe_d = 6, 6.0

    lx, ly, lz = tower_w, tower_d + 2.0 * pad_y, total_h
    nx = int(round(lx / pitch_mm))
    ny = int(round(ly / pitch_mm))
    nz = int(round(lz / pitch_mm))

    xc = -lx / 2 + (np.arange(nx) + 0.5) * pitch_mm
    yc = -ly / 2 + (np.arange(ny) + 0.5) * pitch_mm
    zc = (np.arange(nz) + 0.5) * pitch_mm
    X = xc[None, None, :]
    Y = yc[None, :, None]
    Z = zc[:, None, None]

    solid = np.zeros((nz, ny, nx), dtype=bool)
    thermal_bc = np.zeros((nz, ny, nx), dtype=np.uint8)

    base = (np.abs(X) <= base_w / 2) & (np.abs(Y) <= base_d / 2) & (Z >= 0) & (Z <= base_h)
    solid |= base
    thermal_bc[base] = 2

    plate = (np.abs(X) <= tower_w / 2) & (np.abs(Y) <= tower_d / 2)
    for i in range(fin_n):
        zf = fin_z0 + i * fin_pitch
        fin = (Z >= zf) & (Z <= zf + fin_t) & plate
        solid |= fin
        thermal_bc[fin] = 1

    zcov = total_h - 4.5
    cov = (Z >= zcov) & (Z <= zcov + 1.5) & plate
    solid |= cov
    thermal_bc[cov] = 1

    x_rise = tower_w * 0.34
    span_y = tower_d * 0.62
    for i in range(pipe_n):
        dx = x_rise + (1 if i % 2 == 0 else -1) * 2.0
        yr = -span_y / 2 + i * (span_y / (pipe_n - 1))
        pipe = (((X - dx) ** 2 + (Y - yr) ** 2) <= (pipe_d / 2) ** 2) & (Z >= base_h) & (Z <= total_h)
        solid |= pipe
        thermal_bc[pipe] = 2

    solid[0, :, :] = True
    solid[-1, :, :] = True
    thermal_bc[0, :, :] = 1
    thermal_bc[-1, :, :] = 1
    return solid, thermal_bc


def main():
    pitch_mm, u_phys, steps = 0.4, 2.5, 6000
    t_in, t_hot = 25.0, 65.0
    u_lat = 0.05
    gap_mm = 2.488 - 0.3
    re = u_phys * gap_mm * 1e-3 / 1.5e-5
    nu_lat = u_lat * (gap_mm / pitch_mm) / re
    tau = 0.5 + 3.0 * nu_lat

    solid, thermal_bc = build_heatsink(pitch_mm)
    nz, ny, nx = solid.shape
    sim = LBMCudaSolver(solid, u_in=u_lat, tau=tau, periodic_y=False, cs=0.16)
    sim.enable_thermal(thermal_bc, t_in=t_in, t_hot=t_hot, alpha_fluid=nu_lat / 0.71, alpha_solid=0.25)

    t0 = time.perf_counter()
    sim.run_steps(steps)
    dt = time.perf_counter() - t0
    mlups = (solid.size * steps) / dt / 1e6
    print(f"Grid {nx}x{ny}x{nz} ({solid.size/1e6:.2f}M) | {steps} steps in {dt:.2f}s ({mlups:.1f} MLUPS)")

    mid = nz // 2
    uy, _ = sim.z_slice(mid)
    temp = sim.z_temp_slice(mid)

    out_dir = os.path.join(ROOT, "assets")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "heatsink_thermal_demo.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), dpi=160, constrained_layout=True)
    # 剔除 y=ny-1 出口边界层并采用真实物理坐标 (mm) 与 1:1 等比例显示
    u_plot = np.ma.masked_where(solid[mid, :-1, :].T, uy[:-1, :].T)
    t_plot = np.ma.masked_where(solid[mid, :-1, :].T, temp[:-1, :].T)
    extent = [0.0, (ny - 1) * pitch_mm, 0.0, nx * pitch_mm]

    cmap_u = plt.get_cmap("RdYlBu_r").copy()
    cmap_u.set_bad(color="#2b2b2b")
    im0 = axes[0].imshow(u_plot, origin="lower", cmap=cmap_u, vmin=-0.02, vmax=0.09,
                         aspect="equal", extent=extent)
    axes[0].set_title(f"Velocity u_y (lattice) | {solid.size/1e6:.2f}M voxels, {steps} steps", fontsize=10.5, fontweight="bold")
    axes[0].set_xlabel("Airflow Y (mm)")
    axes[0].set_ylabel("Width X (mm)")
    fig.colorbar(im0, ax=axes[0], pad=0.02, fraction=0.045)

    cmap_t = plt.get_cmap("inferno").copy()
    cmap_t.set_bad(color="#d9d9d9")
    im1 = axes[1].imshow(t_plot, origin="lower", cmap=cmap_t, vmin=t_in, vmax=t_hot,
                         aspect="equal", extent=extent)
    axes[1].set_title(f"D3Q7 Temperature (°C) | T_in={t_in}°C, T_hot={t_hot}°C", fontsize=10.5, fontweight="bold")
    axes[1].set_xlabel("Airflow Y (mm)")
    axes[1].set_ylabel("Width X (mm)")
    fig.colorbar(im1, ax=axes[1], pad=0.02, fraction=0.045)

    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
