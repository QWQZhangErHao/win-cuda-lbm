# -*- coding: utf-8 -*-
"""Physical-to-lattice unit converter with Ma < 0.05 and tau stability bounds."""

from dataclasses import dataclass


@dataclass
class LatticeMapping:
    dx_m: float
    dt_s: float
    u_lat: float
    tau0: float
    nu_lat: float
    alpha_lat: float
    re: float
    ma: float
    vel_scale: float
    press_scale: float


def map_air_25c(
    dx_mm: float,
    u_phys_ms: float,
    char_len_mm: float,
    u_lat_target: float = 0.02,
    rho_air: float = 1.184,
    nu_air: float = 1.56e-5,
    pr_air: float = 0.71,
) -> LatticeMapping:
    u_lat = min(0.024, max(0.005, float(u_lat_target)))
    dx = float(dx_mm) * 1e-3
    dt = dx * u_lat / max(1e-6, float(u_phys_ms))
    vel_scale = dx / dt
    press_scale = rho_air * (1.0 / 3.0) * (vel_scale ** 2)

    re = float(u_phys_ms) * (float(char_len_mm) * 1e-3) / nu_air
    nu_lat = u_lat * (char_len_mm / dx_mm) / max(1.0, re)
    tau0 = max(0.51, min(1.2, 0.5 + 3.0 * nu_lat))
    alpha_lat = nu_lat / pr_air
    ma = u_lat * (3.0 ** 0.5)
    return LatticeMapping(
        dx_m=dx, dt_s=dt, u_lat=u_lat, tau0=tau0,
        nu_lat=nu_lat, alpha_lat=alpha_lat, re=re, ma=ma,
        vel_scale=vel_scale, press_scale=press_scale,
    )
