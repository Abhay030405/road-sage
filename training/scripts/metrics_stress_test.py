"""
training.scripts.metrics_stress_test
======================================

Stress-tests the RoadSage API by sending a continuous stream of real camera
frames to ``POST /api/v1/predict`` for a configurable duration.

The script cycles through all ``rgb_image_*.png`` frames found in the source
directory, sends each as a multipart upload, and records latency and error
statistics.  Progress is printed every 30 seconds; a summary is printed at
the end with a reminder to check Grafana.

Purpose
-------
After running this script you should see:

* Latency percentiles trending upward in the **Inference Latency** panel.
* Command distribution filling out in the **Driving Command Distribution**
  pie chart.
* Safety gate and ML fallback counters incrementing in the stat panels.
* Mean confidence tracking in the time-series panel.

Usage::

    # Default: 5 minutes against local dev server
    python training/scripts/metrics_stress_test.py

    # Custom duration and API URL
    python training/scripts/metrics_stress_test.py \\
        --duration-minutes 10 \\
        --api-url http://localhost:8000 \\
        --source data/mnnit/rgb

Requirements
------------
``httpx`` must be installed::

    pip install httpx

"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)

_PROGRESS_INTERVAL_S = 30      # print a progress line every 30 seconds
_PREDICT_ENDPOINT = "/api/v1/predict"
_GRAFANA_URL = "http://localhost:3001"


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_stress_test(
    api_url: str,
    source_dir: str,
    duration_minutes: float,
) -> None:
    """Send a continuous stream of camera frames to the predict endpoint.

    Cycles through all ``rgb_image_*.png`` images in ``source_dir``
    indefinitely until ``duration_minutes`` has elapsed.

    Parameters
    ----------
    api_url:
        Base URL of the running RoadSage API, e.g. ``http://localhost:8000``.
    source_dir:
        Directory containing ``rgb_image_*.png`` frames.
    duration_minutes:
        Total run duration in minutes.
    """
    try:
        import httpx
    except ImportError:
        print(
            "ERROR: 'httpx' is not installed. Run: pip install httpx",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Discover images                                                     #
    # ------------------------------------------------------------------ #
    image_root = Path(source_dir)
    image_paths = sorted(image_root.glob("rgb_image_*.png"))
    if not image_paths:
        print(
            f"ERROR: No rgb_image_*.png files found in {source_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) in {source_dir}.")
    print(f"Stress-testing {api_url}{_PREDICT_ENDPOINT} "
          f"for {duration_minutes:.1f} minute(s) ...")
    print(f"Progress update every {_PROGRESS_INTERVAL_S}s. Press Ctrl-C to stop early.\n")

    # ------------------------------------------------------------------ #
    # Run loop                                                            #
    # ------------------------------------------------------------------ #
    deadline = time.time() + duration_minutes * 60
    next_progress_at = time.time() + _PROGRESS_INTERVAL_S

    total_requests = 0
    total_errors = 0
    latencies: List[float] = []

    url = f"{api_url.rstrip('/')}{_PREDICT_ENDPOINT}"

    with httpx.Client(timeout=30.0) as client:
        idx = 0
        try:
            while time.time() < deadline:
                img_path = image_paths[idx % len(image_paths)]
                idx += 1

                t0 = time.time()
                try:
                    with open(img_path, "rb") as fh:
                        response = client.post(
                            url,
                            files={"file": (img_path.name, fh, "image/png")},
                        )

                    elapsed_ms = (time.time() - t0) * 1000
                    total_requests += 1
                    latencies.append(elapsed_ms)

                    if response.status_code != 200:
                        total_errors += 1
                        log.debug(
                            "Non-200 response: %d — %s",
                            response.status_code,
                            response.text[:120],
                        )

                except httpx.RequestError as exc:
                    total_errors += 1
                    log.warning("Request error: %s", exc)

                # Progress report
                if time.time() >= next_progress_at:
                    _print_progress(total_requests, total_errors, latencies, deadline)
                    next_progress_at += _PROGRESS_INTERVAL_S

        except KeyboardInterrupt:
            print("\nInterrupted by user.")

    # ------------------------------------------------------------------ #
    # Final summary                                                       #
    # ------------------------------------------------------------------ #
    _print_summary(total_requests, total_errors, latencies, api_url)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_progress(
    total: int,
    errors: int,
    latencies: List[float],
    deadline: float,
) -> None:
    """Print a one-line progress update."""
    remaining = max(0.0, deadline - time.time())
    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    error_pct = errors / total * 100 if total else 0.0
    print(
        f"  [{int(remaining):>4}s left] "
        f"requests={total:>6}  errors={errors:>4} ({error_pct:.1f}%)  "
        f"avg_latency={avg_ms:.0f}ms"
    )


def _print_summary(
    total: int,
    errors: int,
    latencies: List[float],
    api_url: str,
) -> None:
    """Print the final summary table."""
    import numpy as np

    divider = "=" * 56
    print()
    print(divider)
    print("  Stress Test — Final Summary")
    print(divider)
    print(f"  Total requests    : {total}")
    print(f"  Errors            : {errors}  ({errors/total*100:.1f}%)" if total else "  Errors            : 0")

    if latencies:
        arr = latencies
        print(f"  Avg latency       : {sum(arr)/len(arr):.1f} ms")
        print(f"  P50 latency       : {float(sorted(arr)[len(arr)//2]):.1f} ms")
        p95_idx = int(len(arr) * 0.95)
        print(f"  P95 latency       : {float(sorted(arr)[p95_idx]):.1f} ms")
        print(f"  Max latency       : {max(arr):.1f} ms")

    print(divider)
    print()
    print(f"  Metrics are now visible in Grafana:")
    print(f"  {_GRAFANA_URL}")
    print()
    print("  Panels to check:")
    print("    * Inference Latency — should show a populated time series")
    print("    * Driving Command Distribution — pie slices should be filled")
    print("    * Safety Gate Triggers — counter incremented on hazard frames")
    print("    * ML Fallback Activations — shows fallback rate")
    print("    * Mean Prediction Confidence — confidence time series")
    print(divider)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stress-test the RoadSage API by cycling camera frames through "
            "POST /api/v1/predict and populating Grafana metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--duration-minutes",
        type=float,
        default=5.0,
        metavar="MINUTES",
        help="Total run time in minutes.",
    )
    p.add_argument(
        "--api-url",
        default="http://localhost:8000",
        metavar="URL",
        help="Base URL of the RoadSage API.",
    )
    p.add_argument(
        "--source",
        default="rgb",
        metavar="DIR",
        help="Directory containing rgb_image_*.png frames to send.",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for the metrics stress test.

    Parameters
    ----------
    argv:
        Optional argument list (defaults to ``sys.argv``).

    Returns
    -------
    int
        ``0`` on success, ``1`` if no images were found or httpx is missing.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    run_stress_test(
        api_url=args.api_url,
        source_dir=args.source,
        duration_minutes=args.duration_minutes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
