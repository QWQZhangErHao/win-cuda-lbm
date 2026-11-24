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
    fin_n, fin_t, fin_pitch = 58, 0.4, 2.488  # fin_t >= pitch_mm 确保体素化不漏片
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
    for side in (-1.0, 1.0):
        for i in range(pipe_n):
            dx = side * x_rise + (1 if i % 2 == 0 else -1) * 2.0
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
    u_lat = 0.035
    gap_mm = 2.488 - 0.4
    re = u_phys * gap_mm * 1e-3 / 1.5e-5
    nu_lat = u_lat * (gap_mm / pitch_mm) / re
    tau = max(0.515, 0.5 + 3.0 * nu_lat)

    solid, thermal_bc = build_heatsink(pitch_mm)
    nz, ny, nx = solid.shape
    sim = LBMCudaSolver(solid, u_in=u_lat, tau=tau, periodic_y=False, cs=0.16)
    sim.enable_thermal(thermal_bc, t_in=t_in, t_hot=t_hot, alpha_fluid=nu_lat / 0.71, alpha_solid=0.37)

    t0 = time.perf_counter()
    sim.run_steps(steps)
    dt = time.perf_counter() - t0
    mlups = (solid.size * steps) / dt / 1e6
    print(f"Grid {nx}x{ny}x{nz} ({solid.size/1e6:.2f}M) | {steps} steps in {dt:.2f}s ({mlups:.1f} MLUPS)")

    out_dir = os.path.join(ROOT, "assets")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 纯翅片通道局部放大（YZ 纵截面, X=+33mm）
    x_fin = int(round((nx * 0.5) + 33.0 / pitch_mm))
    uy_yz, _ = sim.x_slice(x_fin)
    temp_yz = sim.x_temp_slice(x_fin)
    y0, y1 = int(12.0 / pitch_mm), int(82.0 / pitch_mm)
    z0, z1 = int(60.0 / pitch_mm), int(82.0 / pitch_mm)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), dpi=180, constrained_layout=True)
    solid_zoom = solid[z0:z1, y0:y1, x_fin]
    uy_raw = uy_yz[z0:z1, y0:y1].copy()
    uy_vis = uy_raw.copy()
    uy_vis[:, 1:-1] = 0.25 * uy_raw[:, :-2] + 0.5 * uy_raw[:, 1:-1] + 0.25 * uy_raw[:, 2:]
    u_plot = np.ma.masked_where(solid_zoom, uy_vis)
    t_plot = np.ma.masked_where(solid_zoom, temp_yz[z0:z1, y0:y1])
    extent_yz = [y0 * pitch_mm, y1 * pitch_mm, z0 * pitch_mm, z1 * pitch_mm]

    cmap_u = plt.get_cmap("RdYlBu_r").copy()
    cmap_u.set_bad(color="#2b2b2b")
    im0 = axes[0].imshow(u_plot, origin="lower", cmap=cmap_u, vmin=0.0, vmax=0.06,
                         aspect="equal", extent=extent_yz)
    axes[0].set_title(f"Fin-Channel Velocity u_y (Zoom-in, {solid.size/1e6:.2f}M voxels)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Airflow Direction Y (mm)")
    axes[0].set_ylabel("Fin Stack Height Z (mm)")
    fig.colorbar(im0, ax=axes[0], pad=0.02, fraction=0.025)

    cmap_t = plt.get_cmap("inferno").copy()
    cmap_t.set_bad(color="#e0e0e0")
    im1 = axes[1].imshow(t_plot, origin="lower", cmap=cmap_t, vmin=t_in, vmax=55.0,
                         aspect="equal", extent=extent_yz)
    axes[1].set_title("D3Q7 Thermal Boundary Layer (Zoom-in, °C)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Airflow Direction Y (mm)")
    axes[1].set_ylabel("Fin Stack Height Z (mm)")
    fig.colorbar(im1, ax=axes[1], pad=0.02, fraction=0.025)

    fig.savefig(os.path.join(out_dir, "heatsink_thermal_demo.png"), dpi=160)
    plt.close(fig)

    # 2. 热管绕流与热尾迹（XY 水平中截面, Z=H/2）
    mid_z = nz // 2
    uy_xy, _ = sim.z_slice(mid_z)
    temp_xy = sim.z_temp_slice(mid_z)
    solid_xy = solid[mid_z, :-1, :].T
    uy_xy_vis = uy_xy[:-1, :].T.copy()
    uy_xy_vis[:, 1:-1] = 0.25 * uy_xy_vis[:, :-2] + 0.5 * uy_xy_vis[:, 1:-1] + 0.25 * uy_xy_vis[:, 2:]
    extent_xy = [0.0, (ny - 1) * pitch_mm, 0.0, nx * pitch_mm]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.6), dpi=160, constrained_layout=True)
    im0 = axes[0].imshow(np.ma.masked_where(solid_xy, uy_xy_vis), origin="lower", cmap=cmap_u,
                         vmin=-0.015, vmax=0.06, aspect="equal", extent=extent_xy)
    axes[0].set_title("Staggered Heat-Pipe Wake u_y (XY Plane)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Airflow Y (mm)")
    axes[0].set_ylabel("Width X (mm)")
    fig.colorbar(im0, ax=axes[0], pad=0.02, fraction=0.045)

    im1 = axes[1].imshow(np.ma.masked_where(solid_xy, temp_xy[:-1, :].T), origin="lower", cmap=cmap_t,
                         vmin=t_in, vmax=t_hot, aspect="equal", extent=extent_xy)
    axes[1].set_title("Heat-Pipe Thermal Wake (°C, XY Plane)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Airflow Y (mm)")
    axes[1].set_ylabel("Width X (mm)")
    fig.colorbar(im1, ax=axes[1], pad=0.02, fraction=0.045)

    fig.savefig(os.path.join(out_dir, "heatsink_pipe_wake.png"), dpi=160)
    plt.close(fig)

    # 3. 迎风正面全视图（XZ 截面，Y 处于散热器后半段 Y=55mm）
    y_front = int(55.0 / pitch_mm)
    _, uy_xz, _ = sim.y_plane(y_front)
    t_xz = np.empty((nz, nx), dtype=np.float32)
    sim._lib.lbm_get_y_temp_plane(sim._h, y_front, t_xz.ctypes.data_as(sim._lib.lbm_get_y_temp_plane.argtypes[2]))
    solid_xz = solid[:, y_front, :]
    extent_xz = [0.0, nx * pitch_mm, 0.0, nz * pitch_mm]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 6.2), dpi=160, constrained_layout=True)
    im0 = axes[0].imshow(np.ma.masked_where(solid_xz, uy_xz), origin="lower", cmap=cmap_u,
                         vmin=0.0, vmax=0.06, aspect="equal", extent=extent_xz)
    axes[0].set_title("Front Cross-Section Velocity u_y (XZ Plane)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Width X (mm)")
    axes[0].set_ylabel("Height Z (mm)")
    fig.colorbar(im0, ax=axes[0], pad=0.02, fraction=0.05)

    im1 = axes[1].imshow(np.ma.masked_where(solid_xz, t_xz), origin="lower", cmap=cmap_t,
                         vmin=t_in, vmax=t_hot, aspect="equal", extent=extent_xz)
    axes[1].set_title("Front Cross-Section Temperature (°C, XZ Plane)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Width X (mm)")
    axes[1].set_ylabel("Height Z (mm)")
    fig.colorbar(im1, ax=axes[1], pad=0.02, fraction=0.05)

    fig.savefig(os.path.join(out_dir, "heatsink_front_xz.png"), dpi=160)
    plt.close(fig)
    print(f"Saved 3 views in {out_dir}")


if __name__ == "__main__":
    main()
