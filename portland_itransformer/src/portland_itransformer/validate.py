"""Data validation and quality reporting."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate Portland Harbor data."""

    def __init__(self, config):
        """Initialize validator."""
        self.config = config
        self.report = {}

    def validate(self) -> dict:
        """Run full validation."""
        logger.info("Starting data validation...")

        # Load raw CSV
        raw_csv = self.config.paths.raw_csv
        logger.info(f"Loading: {raw_csv}")

        df = pd.read_csv(raw_csv)

        # Remove unnamed index column if present
        if df.columns[0].startswith("Unnamed"):
            df = df.iloc[:, 1:]
            logger.info("Removed unnamed index column")

        # Check schema
        self._validate_schema(df)

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Check cadence
        self._validate_cadence(df)

        # Check duplicates
        self._validate_duplicates(df)

        # Check values
        self._validate_values(df)

        # Check degrees
        self._validate_degrees(df)

        self.report['status'] = 'pass'
        self.report['timestamp'] = datetime.utcnow().isoformat()
        self.report['num_rows'] = len(df)
        self.report['date_range'] = {
            'start': str(df['timestamp'].iloc[0]),
            'end': str(df['timestamp'].iloc[-1]),
        }

        logger.info(f"Validation complete: {len(df)} rows, {self.report['status']}")
        return self.report

    def _validate_schema(self, df: pd.DataFrame):
        """Check that all required columns exist."""
        from .constants import RAW_COLUMNS

        required = set(RAW_COLUMNS)
        actual = set(df.columns)
        missing = required - actual
        extra = actual - required

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        if extra:
            logger.warning(f"Extra columns (ignored): {extra}")

        logger.info("✓ Schema validation passed")
        self.report['schema'] = 'pass'

    def _validate_cadence(self, df: pd.DataFrame):
        """Check 15-minute cadence."""
        diffs = df['timestamp'].diff().dt.total_seconds() / 60  # minutes
        expected_cadence = self.config.data.cadence_minutes

        # Check first non-NaN diff
        diffs_valid = diffs[1:]  # Skip first NaN
        cadence_mode = diffs_valid.mode()

        if len(cadence_mode) > 0:
            actual_cadence = cadence_mode.iloc[0]
        else:
            actual_cadence = expected_cadence

        bad_cadence = diffs_valid[diffs_valid != expected_cadence]

        if len(bad_cadence) > 0:
            logger.warning(f"Found {len(bad_cadence)} non-standard cadence gaps")
            self.report['cadence_issues'] = len(bad_cadence)
        else:
            logger.info("✓ Cadence validation passed")

        self.report['cadence'] = {
            'expected_minutes': expected_cadence,
            'actual_mode_minutes': float(actual_cadence),
            'violations': len(bad_cadence),
        }

    def _validate_duplicates(self, df: pd.DataFrame):
        """Check for duplicate timestamps."""
        dup_mask = df['timestamp'].duplicated()
        n_dup = dup_mask.sum()

        if n_dup > 0:
            logger.warning(f"Found {n_dup} duplicate timestamps")
            self.report['duplicates'] = n_dup
        else:
            logger.info("✓ No duplicate timestamps")

        self.report['duplicates'] = n_dup

    def _validate_values(self, df: pd.DataFrame):
        """Check for non-finite values."""
        non_finite = {}

        for col in df.columns:
            if col == 'timestamp':
                continue
            mask = ~np.isfinite(df[col])
            count = mask.sum()
            if count > 0:
                non_finite[col] = int(count)

        if non_finite:
            logger.warning(f"Non-finite values: {non_finite}")
            self.report['non_finite'] = non_finite
        else:
            logger.info("✓ All values finite")

        self.report['non_finite'] = non_finite

    def _validate_degrees(self, df: pd.DataFrame):
        """Check degree fields are in [0, 360)."""
        degree_cols = [
            'wind_direction_deg',
            'current_direction_deg',
            'compass_deg',
        ]

        degree_issues = {}

        for col in degree_cols:
            if col not in df.columns:
                continue

            bad_mask = (df[col] < 0) | (df[col] >= 360)
            bad_count = bad_mask.sum()

            if bad_count > 0:
                degree_issues[col] = int(bad_count)

        if degree_issues:
            logger.warning(f"Out-of-range degree values: {degree_issues}")
            # Normalize them
            for col in degree_issues:
                df[col] = np.mod(df[col], 360.0)
                logger.info(f"Normalized {col}")

        self.report['degree_issues'] = degree_issues

    def save_reports(self, output_dir: Path):
        """Save validation report as JSON and CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = output_dir / 'data_quality_report.json'
        with open(json_path, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        logger.info(f"Saved: {json_path}")

        # CSV summary
        csv_path = output_dir / 'data_quality_report.csv'
        summary = pd.DataFrame([{
            'metric': k,
            'value': str(v),
        } for k, v in self.report.items()])
        summary.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")
