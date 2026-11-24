# -*- coding: utf-8 -*-
import ctypes
import math
import os
import time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DLL = os.path.join(_HERE, "lbm_cuda.dll")
_LIB = None


def _add_cuda_dirs():
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    dirs = [_HERE]
    if os.environ.get("CUDA_PATH"):
        dirs += [os.path.join(os.environ["CUDA_PATH"], "bin"),
                 os.path.join(os.environ["CUDA_PATH"], "bin", "x64")]
    root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(root):
        for v in sorted(os.listdir(root), reverse=True):
            dirs += [os.path.join(root, v, "bin"), os.path.join(root, v, "bin", "x64")]
    for d in dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def load_lib(dll_path: str = _DLL) -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB
    if not os.path.isfile(dll_path):
        raise FileNotFoundError(f"{dll_path} not found. Run build.ps1 first.")
    _add_cuda_dirs()
    lib = ctypes.CDLL(dll_path)
    cf_p = ctypes.POINTER(ctypes.c_float)
    cu8_p = ctypes.POINTER(ctypes.c_uint8)
    ci_p = ctypes.POINTER(ctypes.c_int)
    cs_p = ctypes.POINTER(ctypes.c_size_t)

    lib.lbm_device_info.argtypes = [ctypes.c_char_p, ctypes.c_int, ci_p, ci_p, cs_p]
    lib.lbm_device_info.restype = ctypes.c_int
    lib.lbm_create.argtypes = [
        cu8_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float,
    ]
    lib.lbm_create.restype = ctypes.c_void_p
    lib.lbm_run_steps.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.lbm_run_steps.restype = ctypes.c_int
    lib.lbm_get_y_plane.argtypes = [ctypes.c_void_p, ctypes.c_int, cf_p, cf_p]
    lib.lbm_get_y_plane.restype = ctypes.c_int
    lib.lbm_get_z_slice.argtypes = [ctypes.c_void_p, ctypes.c_int, cf_p, cf_p]
    lib.lbm_get_z_slice.restype = ctypes.c_int
    lib.lbm_check_health.argtypes = [ctypes.c_void_p, ci_p, cf_p, cf_p]
    lib.lbm_check_health.restype = ctypes.c_int
    lib.lbm_enable_thermal.argtypes = [
        ctypes.c_void_p, cu8_p,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    ]
    lib.lbm_enable_thermal.restype = ctypes.c_int
    lib.lbm_get_y_temp_plane.argtypes = [ctypes.c_void_p, ctypes.c_int, cf_p]
    lib.lbm_get_y_temp_plane.restype = ctypes.c_int
    lib.lbm_get_z_temp_slice.argtypes = [ctypes.c_void_p, ctypes.c_int, cf_p]
    lib.lbm_get_z_temp_slice.restype = ctypes.c_int
    lib.lbm_destroy.argtypes = [ctypes.c_void_p]
    lib.lbm_destroy.restype = None
    _LIB = lib
    return lib


class LBMCudaSolver:
    def __init__(self, solid: np.ndarray, u_in=0.02, tau=0.6,
                 body_force=0.0, periodic_y=False, cs=0.16):
        self._lib = load_lib()
        nz, ny, nx = solid.shape
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.solid_host = np.ascontiguousarray(solid, dtype=bool)
        s_u8 = np.ascontiguousarray(solid, dtype=np.uint8)
        self._h = self._lib.lbm_create(
            s_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self.nz, self.ny, self.nx,
            float(u_in), float(tau), float(cs),
            float(body_force), 1 if periodic_y else 0, 0.5005, 1.98,
        )
        if not self._h:
            raise RuntimeError("lbm_create failed")

    def run_steps(self, nsteps: int):
        if self._lib.lbm_run_steps(self._h, int(nsteps)) != 0:
            raise RuntimeError("lbm_run_steps failed")

    def enable_thermal(self, thermal_bc: np.ndarray, t_in=25.0, t_hot=65.0,
                       alpha_fluid=0.01, alpha_solid=0.25):
        bc = np.ascontiguousarray(thermal_bc, dtype=np.uint8)
        if self._lib.lbm_enable_thermal(
            self._h, bc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            float(t_in), float(t_hot), float(alpha_fluid), float(alpha_solid)
        ) != 0:
            raise RuntimeError("lbm_enable_thermal failed")

    def y_plane(self, y: int):
        rho = np.empty((self.nz, self.nx), dtype=np.float32)
        uy = np.empty((self.nz, self.nx), dtype=np.float32)
        self._lib.lbm_get_y_plane(
            self._h, int(y),
            rho.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            uy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        m = (~self.solid_host[:, y, :]) & (rho > 1e-8)
        return rho, uy, m

    def plane_u(self, y: int) -> float:
        _, uy, m = self.y_plane(y)
        return float(uy[m].mean(dtype=np.float64)) if m.any() else float("nan")

    def plane_temp(self, y: int) -> float:
        t = np.empty((self.nz, self.nx), dtype=np.float32)
        self._lib.lbm_get_y_temp_plane(
            self._h, int(y), t.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        m = ~self.solid_host[:, y, :]
        return float(t[m].mean(dtype=np.float64)) if m.any() else float("nan")

    def z_slice(self, z: int):
        rho = np.empty((self.ny, self.nx), dtype=np.float32)
        uy = np.empty((self.ny, self.nx), dtype=np.float32)
        self._lib.lbm_get_z_slice(
            self._h, int(z),
            rho.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            uy.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return uy, rho

    def z_temp_slice(self, z: int):
        temp = np.empty((self.ny, self.nx), dtype=np.float32)
        self._lib.lbm_get_z_temp_slice(
            self._h, int(z),
            temp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return temp

    def close(self):
        if getattr(self, "_h", None):
            self._lib.lbm_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


if __name__ == "__main__":
    # Poiseuille channel verification
    H, L, tau, u_target, steps = 32, 32, 1.0, 0.02, 6000
    solid = np.zeros((H, L, 4), dtype=bool)
    solid[0, :, :] = True
    solid[-1, :, :] = True
    g = u_target * 12.0 * ((tau - 0.5) / 3.0) / ((H - 2.0) ** 2)
    sim = LBMCudaSolver(solid, u_in=0.0, tau=tau, body_force=g, periodic_y=True)
    t0 = time.perf_counter()
    sim.run_steps(steps)
    dt = time.perf_counter() - t0
    u_mean = float(np.mean([sim.plane_u(y) for y in range(2, H - 2, 2)]))
    err = (u_mean - u_target) / u_target * 100.0
    print(f"Poiseuille {steps} steps in {dt:.3f}s | u_mean={u_mean:.6f} (err {err:+.2f}%)")
