# -*- coding: utf-8 -*-
"""
Standalone CLI runner: Physical Units -> GPU LBM (CUB Convergence) -> ParaView .vti
Usage:
    python cli/run_case.py --dx 0.8 --u 2.5 --steps 3000 --vti result.vti
"""

import argparse
import os
import sys
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lbm_cuda import LBMCudaSolver
from cli.units import map_air_25c
from cli.post_vti import write_vti
from examples.finned_heatsink import build_heatsink


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=0.8, help="Voxel size (mm)")
    ap.add_argument("--u", type=float, default=2.5, help="Inlet velocity (m/s)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--check-every", type=int, default=1000)
    ap.add_argument("--vti", type=str, default="cooler_result.vti")
    a = ap.parse_args()

    solid, bc = build_heatsink(pitch_mm=a.dx)
    bc[bc == 2] = 3
    bc[0:max(2, int(2.0 / a.dx)), :, :] = 2
    nz, ny, nx = solid.shape

    m = map_air_25c(dx_mm=a.dx, u_phys_ms=a.u, char_len_mm=2.088, u_lat_target=0.02)
    print(f"[CLI] Grid {nx}x{ny}x{nz} ({solid.size/1e6:.2f}M) | dx={a.dx}mm | Ma={m.ma:.4f} | tau={m.tau0:.4f}")

    sim = LBMCudaSolver(solid, u_in=m.u_lat, tau=m.tau0, periodic_y=False, cs=0.16)
    sim.enable_thermal(bc, t_in=25.0, t_hot=65.0, alpha_fluid=m.alpha_lat, alpha_solid=0.35)

    y_in = max(2, int(5.0 / a.dx))
    y_out = min(ny - 3, ny - max(3, int(5.0 / a.dx)))

    t0 = time.perf_counter()
    done = 0
    while done < a.steps:
        chunk = min(a.check_every, a.steps - done)
        sim.run_steps(chunk)
        done += chunk
        met = sim.compute_metrics(y_in, y_out)
        dp_pa = (met["rho_in"] - met["rho_out"]) * m.press_scale
        print(f"  step {done:5d} | dp={dp_pa:6.2f} Pa | T_max_solid={met['t_max_solid']:.2f} C")

    dt = time.perf_counter() - t0
    print(f"Finished {a.steps} steps in {dt:.2f}s ({solid.size*a.steps/dt/1e6:.1f} MLUPS)")

    if a.vti:
        ux, uy, uz, rho, temp = sim.extract_fields_3d(with_thermal=True)
        out_path = os.path.join(ROOT, a.vti)
        write_vti(out_path, a.dx, solid, ux, uy, uz, rho, temp, m.vel_scale, m.press_scale)
        print(f"Exported ParaView VTI: {out_path} ({os.path.getsize(out_path)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
