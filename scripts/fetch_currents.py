"""Download ocean surface currents from Copernicus Marine (CMEMS).

    copernicusmarine login          # once - see CREDENTIALS below
    python scripts/fetch_currents.py --config configs/fetch_mc20.yaml

Currents are the last synthetic input in the pipeline. Backward drift moves a
slick from where it was seen to where it started, and with an invented current
field that estimate is close to arbitrary - scripts/validate_elsa3.py prints
its own caveat saying exactly that. Real currents are what turn "the drift
moved toward the wreck" from a coin flip into evidence.

CREDENTIALS. CMEMS has no API key, which is the usual point of confusion. The
toolbox authenticates with the username and password from the account
registration and caches them:

    copernicusmarine login

The username is NOT the email address - it is the separate username Copernicus
issues, shown on the profile page at marine.copernicus.eu. Credentials land in
~/.copernicusmarine/; nothing here reads or stores them.

DATASETS. Which product covers a date depends on how old the date is:

  * analysis/forecast      - recent dates plus a rolling window of past ones
  * multi-year reanalysis  - older dates, published with a lag of a year or more

Both are tried in order and whichever answers is printed and recorded, because
which product a number came from is part of the number.

The subset is bounded to the scene bbox and the drift window, so a file is a
few megabytes rather than the whole globe. The deployed container never reads
these: drift runs at precompute time and only its result is served.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Surface velocity components. These are drift/readers.py NetCDFField's
# defaults (u_var="uo", v_var="vo"); changing one means changing both.
VARIABLES = ["uo", "vo"]

# Tried in order. The analysis/forecast product carries recent dates; the
# reanalyses carry older ones and lag real time by a year or more.
DATASETS = [
    ("cmems_mod_glo_phy_anfc_0.083deg_PT1H-m", "analysis/forecast, hourly"),
    ("cmems_mod_glo_phy_myint_0.083deg_P1D-m", "interim reanalysis, daily"),
    ("cmems_mod_glo_phy_my_0.083deg_P1D-m", "multi-year reanalysis, daily"),
]


def credentials_present() -> bool:
    """Whether the toolbox has cached credentials to work with."""
    for candidate in (
        Path.home() / ".copernicusmarine" / ".copernicusmarine-credentials",
        Path.home() / ".copernicusmarine-credentials",
    ):
        if candidate.exists():
            return True
    import os

    return bool(os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"))


def fetch(
    dataset_id: str,
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    out_path: Path,
) -> bool:
    """One attempt against one product. True if a non-empty file was written."""
    import copernicusmarine

    min_lon, min_lat, max_lon, max_lat = bbox
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=VARIABLES,
        minimum_longitude=min_lon, maximum_longitude=max_lon,
        minimum_latitude=min_lat, maximum_latitude=max_lat,
        start_datetime=start.strftime("%Y-%m-%dT%H:%M:%S"),
        end_datetime=end.strftime("%Y-%m-%dT%H:%M:%S"),
        minimum_depth=0.0, maximum_depth=1.0,
        output_filename=out_path.name,
        output_directory=str(out_path.parent),
        overwrite=True,
    )
    return out_path.exists() and out_path.stat().st_size > 0


def describe(path: Path) -> str:
    """What actually arrived.

    An empty subset is the documented failure mode for these services: a
    slightly-wrong bbox returns a valid file with a zero-length dimension
    rather than an error, and a drift run reading it would produce a confident
    origin from no data at all.
    """
    import numpy as np
    import xarray as xr

    with xr.open_dataset(path) as ds:
        dims = {k: int(v) for k, v in ds.sizes.items()}
        missing = [v for v in VARIABLES if v not in ds]
        if missing:
            raise ValueError(f"{path.name} lacks {missing}; has {list(ds.data_vars)}")
        if any(n == 0 for n in dims.values()):
            raise ValueError(
                f"{path.name} has an empty dimension {dims}. The bbox or time "
                f"window falls outside this product's coverage."
            )
        finite = int(np.isfinite(ds[VARIABLES[0]].values).sum())
        if finite == 0:
            raise ValueError(
                f"{path.name} contains no finite {VARIABLES[0]} values - the "
                f"box may be entirely on land."
            )
        return f"dims {dims}, {finite:,} finite {VARIABLES[0]} samples"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pad-deg", type=float, default=1.0,
                    help="widen the bbox so drifting particles stay inside it")
    ap.add_argument("--hours-before", type=float, default=36.0)
    ap.add_argument("--hours-after", type=float, default=12.0)
    args = ap.parse_args()

    from core.config import load_config, resolve_path

    config = load_config(args.config)
    section = config.section("fetch") or {}
    if "bbox" not in section or "date" not in section:
        print("config needs fetch.bbox and fetch.date", file=sys.stderr)
        return 2

    bbox = tuple(float(v) for v in section["bbox"])
    centre = datetime.fromisoformat(str(section["date"]))
    if centre.tzinfo is None:
        centre = centre.replace(tzinfo=timezone.utc)

    # Particles leave the scene during a backward run, and a field that stops at
    # the scene edge pins them to the boundary - a confident, wrong origin.
    # Padding costs a few megabytes.
    padded = (bbox[0] - args.pad_deg, bbox[1] - args.pad_deg,
              bbox[2] + args.pad_deg, bbox[3] + args.pad_deg)
    start = centre - timedelta(hours=args.hours_before)
    end = centre + timedelta(hours=args.hours_after)

    out_path = resolve_path(
        section.get("currents_path")
        or f"data/currents/currents_{centre:%Y%m%dT%H%M}.nc"
    )

    # Preflight. The toolbox does not raise when credentials are missing - it
    # prompts, and with no terminal attached it prints "Abort" and returns
    # normally. Every dataset then "fails" and the run ends complaining about
    # coverage, which sends you looking in entirely the wrong place.
    if not credentials_present():
        for line in (
            "Not logged in to Copernicus Marine.",
            "",
            "There is no API key - that is why you cannot find one. The",
            "toolbox uses your account username and password directly. Run:",
            "",
            "    .venv/Scripts/copernicusmarine login",
            "",
            "The USERNAME is not your email address: it is the separate",
            "username Copernicus issued, shown on your profile page at",
            "marine.copernicus.eu. It is cached in ~/.copernicusmarine/ and",
            "only the toolbox ever reads it.",
        ):
            print(line, file=sys.stderr)
        return 1

    print("Copernicus Marine surface currents")
    print(f"  bbox   : {padded}  (scene bbox padded by {args.pad_deg} deg)")
    print(f"  window : {start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} UTC")

    last_error: Exception | None = None
    for dataset_id, note in DATASETS:
        print(f"  trying {dataset_id}  ({note})", flush=True)
        try:
            if fetch(dataset_id, padded, start, end, out_path):
                summary = describe(out_path)
                size_mb = out_path.stat().st_size / 1e6
                print(f"\nOK  {dataset_id}")
                print(f"    {summary}")
                print(f"-> {out_path}  ({size_mb:.1f} MB)")
                print("\nPoint the scene config at it:")
                print("  drift:")
                print("    currents_source: cmems")
                print(f"    currents_path: {out_path.relative_to(REPO_ROOT).as_posix()}")
                return 0
        except Exception as exc:                       # noqa: BLE001
            last_error = exc
            message = str(exc)
            print(f"    no: {type(exc).__name__}: {message[:160]}")
            lowered = message.lower()
            if "credential" in lowered or "unauthor" in lowered or "login" in lowered:
                print(
                    "\nNot logged in. CMEMS has no API key - run:\n"
                    "    copernicusmarine login\n"
                    "and give the USERNAME from marine.copernicus.eu (not the "
                    "email address) with its password.",
                    file=sys.stderr,
                )
                return 1

    print(
        f"\nFAILED: no CMEMS product covered {padded} for "
        f"{start:%Y-%m-%d} to {end:%Y-%m-%d}.\n"
        f"Last error: {last_error}\n"
        f"Reanalyses lag real time by a year or more and the analysis/forecast "
        f"product keeps only a rolling window of past dates, so a date can fall "
        f"between the two.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
