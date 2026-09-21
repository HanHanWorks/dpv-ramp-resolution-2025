#!/usr/bin/env python3
"""Convert Tianji gridded weather to normalized PV output."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--u0", type=float, default=26.9)
    parser.add_argument("--u1", type=float, default=6.2)
    parser.add_argument("--gamma", type=float, default=0.0035)
    parser.add_argument("--eta-inv", type=float, default=0.96)
    args = parser.parse_args()

    root = Path(args.input_root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(root / "DSWRFsfc.nc") as ghi_ds, xr.open_dataset(root / "t2mz.nc") as t_ds, xr.open_dataset(root / "UGRD10m.nc") as u_ds, xr.open_dataset(root / "VGRD10m.nc") as v_ds:
        ghi = ghi_ds["DSWRFsfc"].clip(min=0)
        tamb_c = t_ds["t2mz"] - 273.15
        ws10m = np.sqrt(u_ds["UGRD10m"] ** 2 + v_ds["VGRD10m"] ** 2)
        tcell = tamb_c + ghi / (args.u0 + args.u1 * ws10m)
        pv_pu = (ghi / 1000.0) * (1 - args.gamma * (tcell - 25.0)) * args.eta_inv
        pv_pu = pv_pu.clip(min=0, max=1.2).rename("pv_pu")
        tcell = tcell.rename("tcell_C")
        ws10m = ws10m.rename("ws10m")

        out = xr.Dataset(
            {
                "pv_pu": pv_pu.astype("float32"),
                "tcell_C": tcell.astype("float32"),
                "ws10m": ws10m.astype("float32"),
            },
            attrs={
                "description": "Simplified normalized PV output from Tianji gridded weather",
                "formula": "clip(GHI/1000*(1-0.0035*(Tcell-25))*0.96, 0, 1.2)",
            },
        )
        out.to_netcdf(output)

    print(f"[OK] wrote {output}")


if __name__ == "__main__":
    main()
