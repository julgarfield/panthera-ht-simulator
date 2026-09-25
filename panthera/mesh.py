"""Lossless binary-STL to OBJ conversion; no simplification or rescaling."""
from pathlib import Path
import struct
import numpy as np


def stl_to_obj(source: Path, destination: Path) -> None:
    raw = source.read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError(f'Expected binary STL: {source}')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attr', '<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, offset=84, count=count)['vertices']
    # Nine significant digits round-trip every float32 from the official STL.
    vertices, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    with destination.open('w', encoding='ascii', newline='\n') as out:
        out.write('# Exact official STL triangles; metres; no mesh reduction.\n')
        np.savetxt(out, vertices, fmt='v %.9g %.9g %.9g')
        np.savetxt(out, inverse.reshape(-1, 3) + 1, fmt='f %d %d %d')
