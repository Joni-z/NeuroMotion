#!/usr/bin/env python3
"""Dump the structure of a WAY-EEG-GAL .mat file.

The dataset ships MATLAB structs whose exact field names are documented only
loosely in the paper, so we look before we write a loader.
"""
import sys
import numpy as np


def walk(obj, name="", depth=0, maxdepth=4):
    pad = "  " * depth
    if depth > maxdepth:
        print(f"{pad}{name}: ...")
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.names:  # struct array
            print(f"{pad}{name}: struct{obj.shape} fields={list(obj.dtype.names)}")
            flat = obj.ravel()
            if flat.size:
                for f in obj.dtype.names:
                    walk(flat[0][f], f, depth + 1, maxdepth)
            return
        if obj.dtype == object:
            print(f"{pad}{name}: cell{obj.shape}")
            flat = obj.ravel()
            for i, x in enumerate(flat[:3]):
                walk(x, f"[{i}]", depth + 1, maxdepth)
            if flat.size > 3:
                print(f"{'  ' * (depth + 1)}... {flat.size} entries")
            return
        desc = f"{obj.dtype} {obj.shape}"
        if obj.size and obj.size <= 8 and np.issubdtype(obj.dtype, np.number):
            desc += f" = {obj.ravel()}"
        elif obj.size and np.issubdtype(obj.dtype, np.number):
            with np.errstate(all="ignore"):
                desc += f" range=[{np.nanmin(obj):.4g}, {np.nanmax(obj):.4g}]"
        print(f"{pad}{name}: {desc}")
        return
    if isinstance(obj, (str, bytes)):
        print(f"{pad}{name}: str {obj!r}")
        return
    print(f"{pad}{name}: {type(obj).__name__} {obj!r}"[:200])


def main(path):
    try:
        from scipy.io import loadmat
        d = loadmat(path, squeeze_me=False, struct_as_record=True)
        print(f"# {path}\n# loaded with scipy.io.loadmat (MATLAB <= v7.2)")
        for k, v in d.items():
            if k.startswith("__"):
                continue
            walk(v, k)
    except NotImplementedError:
        import h5py
        print(f"# {path}\n# MATLAB v7.3 (HDF5), using h5py")
        with h5py.File(path, "r") as f:
            f.visit(lambda n: print("  " + n))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
