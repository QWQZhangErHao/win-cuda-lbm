# -*- coding: utf-8 -*-
"""Zero-dependency binary VTK ImageData (.vti) writer for ParaView."""

import struct
import numpy as np


def write_vti(
    path: str,
    dx_mm: float,
    solid: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    rho: np.ndarray,
    temp: np.ndarray = None,
    vel_scale: float = 1.0,
    press_scale: float = 1.0,
) -> None:
    nz, ny, nx = solid.shape
    u_mag = np.sqrt(ux * ux + uy * uy + uz * uz) * np.float32(vel_scale)
    p_pa = (rho - np.float32(1.0)) * np.float32(press_scale)

    arrays = [
        ("solid", "UInt8", np.ascontiguousarray(solid, dtype=np.uint8)),
        ("U_mag_ms", "Float32", np.ascontiguousarray(u_mag, dtype=np.float32)),
        ("p_Pa", "Float32", np.ascontiguousarray(p_pa, dtype=np.float32)),
    ]
    if temp is not None:
        arrays.append(("T_C", "Float32", np.ascontiguousarray(temp, dtype=np.float32)))

    offset = 0
    point_data_xml = []
    blobs = []
    for name, vtk_type, arr in arrays:
        raw = arr.tobytes(order="C")
        point_data_xml.append(
            f'        <DataArray type="{vtk_type}" Name="{name}" format="appended" offset="{offset}"/>'
        )
        blobs.append(struct.pack("<Q", len(raw)) + raw)
        offset += 8 + len(raw)

    header = (
        f'<?xml version="1.0"?>\n'
        f'<VTKFile type="ImageData" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n'
        f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" Origin="0 0 0" Spacing="{dx_mm} {dx_mm} {dx_mm}">\n'
        f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">\n'
        f'      <PointData>\n'
        + "\n".join(point_data_xml) + "\n"
        f'      </PointData>\n'
        f'    </Piece>\n'
        f'  </ImageData>\n'
        f'  <AppendedData encoding="raw">\n_'
    ).encode("ascii")

    footer = b"\n  </AppendedData>\n</VTKFile>\n"
    with open(path, "wb") as fp:
        fp.write(header)
        for b in blobs:
            fp.write(b)
        fp.write(footer)
