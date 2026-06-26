"""Data validation for marine local MTGNN."""

from pathlib import Path
from typing import Any
import json
import pandas as pd
import logging

from .loader import (
    load_raw_csv,
    validate_columns,
    compute_cadence,
    check_monotonicity,
    check_value_ranges,
    summarize_data,
)
from ..constants import RAW_CSV_COLUMNS
from ..config import Config

logger = logging.getLogger(__name__)


# Expected value ranges for marine parameters
DEFAULT_RANGES = {
    "air_temp_c": (-50, 60),
    "air_pressure_hpa": (850, 1100),
    "wind_direction_deg": (0, 360),
    "wind_speed_ms": (0, 50),
    "wind_u_east_ms": (-50, 50),  # Will be computed
    "wind_v_north_ms": (-50, 50),  # Will be computed
    "significant_wave_height_m": (0, 20),
    "significant_wave_period_s": (0, 30),
    "zero_crossing_period_s": (0, 30),
    "peak_wave_period_s": (0, 50),
    "peak_wave_direction_deg": (0, 360),
    "water_temp_c": (-5, 50),
    "tidal_level_m": (-10, 10),
    "dew_point_c": (-50, 50),
    "global_radiation_wm2": (0, 1500),
    "salinity_psu": (0, 50),
    "current_direction_deg": (0, 360),
    "current_speed_ms": (0, 5),
    "current_u_east_ms": (-5, 5),  # Will be computed
    "current_v_north_ms": (-5, 5),  # Will be computed
}


class DataValidator:
    """Orchestrate data validation."""

    def __init__(self, config: Config):
        """
        Initialize validator.

        Parameters
        ----------
        config : Config
            Configuration object
        """
        self.config = config
        self.report = {}

    def validate(self) -> dict[str, Any]:
        """
        Validate raw data and produce quality report.

        Returns
        -------
        dict
            Validation report with all checks

        Raises
        ------
        ValueError
            If critical issues found (missing columns, non-monotonic timestamps)
        """
        logger.info("Starting data validation...")

        # Load data
        df = load_raw_csv(
            self.config.data.raw_csv,
            timezone=self.config.site.timezone,
        )
        logger.info(f"Loaded {len(df)} rows")

        # Check columns
        expected_cols = [col for col in RAW_CSV_COLUMNS if col != "timestamp"]
        is_valid, missing = validate_columns(df, expected_cols)
        if not is_valid:
            raise ValueError(f"Missing required columns: {missing}")
        logger.info(f"✓ All {len(expected_cols)} required columns present")

        # Check monotonicity
        is_monotonic, dups = check_monotonicity(df.index)
        if not is_monotonic:
            if dups:
                raise ValueError(f"Found {len(dups)} duplicate timestamps")
            raise ValueError("Timestamps are not monotonically increasing")
        logger.info("✓ Timestamps are monotonically increasing with no duplicates")

        # Compute cadence
        cadence, missing_intervals = compute_cadence(df.index)
        logger.info(f"✓ Nominal cadence: {cadence}")
        if missing_intervals:
            logger.warning(f"  Found {len(missing_intervals)} gaps")

        # Check value ranges
        range_warnings = check_value_ranges(df, DEFAULT_RANGES)
        if range_warnings:
            logger.warning(f"  Found {len(range_warnings)} parameters with out-of-range values")

        # Generate summary
        summary = summarize_data(df)

        # Assemble report
        self.report = {
            "validation_status": "pass" if is_monotonic and not missing else "pass_with_warnings",
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            "data_file": str(self.config.data.raw_csv),
            "summary": summary,
            "cadence": {
                "nominal_minutes": float(cadence.total_seconds() / 60),
                "nominal_pandas": str(cadence),
            },
            "monotonicity": {
                "is_monotonic": bool(is_monotonic),
                "duplicate_count": len(dups),
                "duplicates": dups,
            },
            "missing_intervals": {
                "total_gaps": len(missing_intervals),
                "total_missing_rows": sum(g["missing_rows"] for g in missing_intervals),
                "intervals": missing_intervals[:10],  # Limit output
            } if missing_intervals else None,
            "value_range_warnings": {
                "total_parameters_with_warnings": len(range_warnings),
                "warnings": range_warnings[:20],  # Limit output
            } if range_warnings else None,
            "columns_validated": expected_cols,
        }

        logger.info(f"Validation complete: {self.report['validation_status']}")
        return self.report

    def save_reports(self, output_dir: str | Path = "outputs") -> None:
        """
        Save JSON and CSV reports.

        Parameters
        ----------
        output_dir : str | Path
            Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_path = output_dir / "data_quality_report.json"
        with open(json_path, "w") as f:
            json.dump(self.report, f, indent=2)
        logger.info(f"Saved JSON report: {json_path}")

        # Save CSV report (flattened)
        csv_path = output_dir / "data_quality_report.csv"
        csv_data = self._flatten_report()
        csv_data.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV report: {csv_path}")

    def _flatten_report(self) -> pd.DataFrame:
        """Flatten report to CSV format."""
        rows = []

        # Summary section
        summary = self.report.get("summary", {})
        rows.append(["section", "key", "value"])
        rows.append(["SUMMARY", "total_rows", summary.get("total_rows")])
        rows.append(["SUMMARY", "total_columns", summary.get("total_columns")])
        rows.append(["SUMMARY", "time_range_start", summary.get("time_range", {}).get("start")])
        rows.append(["SUMMARY", "time_range_end", summary.get("time_range", {}).get("end")])

        # Monotonicity section
        mono = self.report.get("monotonicity", {})
        rows.append(["MONOTONICITY", "is_monotonic", mono.get("is_monotonic")])
        rows.append(["MONOTONICITY", "duplicate_count", mono.get("duplicate_count")])

        # Cadence section
        cadence = self.report.get("cadence", {})
        rows.append(["CADENCE", "nominal_minutes", cadence.get("nominal_minutes")])

        # Missing intervals summary
        missing = self.report.get("missing_intervals", {})
        if missing:
            rows.append(["MISSING_INTERVALS", "total_gaps", missing.get("total_gaps")])
            rows.append(["MISSING_INTERVALS", "total_missing_rows", missing.get("total_missing_rows")])

        # Range warnings summary
        warnings = self.report.get("value_range_warnings", {})
        if warnings:
            rows.append(["VALUE_RANGES", "total_parameters_with_warnings", warnings.get("total_parameters_with_warnings")])

        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df
