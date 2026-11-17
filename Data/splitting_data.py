"""
Create a reproducible 20% "reporting" subset of the Telco Customer Churn data.

- Uses a fixed random seed (42) so the same rows are selected every run.
- Stratifies on Churn so the reporting set keeps the same Yes/No churn ratio
  as the full dataset.
- Also writes the remaining 80% to a separate file, so the two sets never overlap.

Usage:
    python make_reporting_set.py
    python make_reporting_set.py --input other.csv --frac 0.2 --seed 42
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Split off a reproducible reporting set.")
    parser.add_argument("--input", default="WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--reporting-out", default="telco_churn_reporting_20pct.csv")
    parser.add_argument("--remaining-out", default="telco_churn_remaining_80pct.csv")
    parser.add_argument("--frac", type=float, default=0.20, help="Fraction for the reporting set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-stratify", action="store_true", help="Sample purely at random")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    stratify_col = None if args.no_stratify else df["Churn"]

    remaining, reporting = train_test_split(
        df,
        test_size=args.frac,
        random_state=args.seed,
        stratify=stratify_col,
        shuffle=True,
    )

    reporting.to_csv(args.reporting_out, index=False)
    remaining.to_csv(args.remaining_out, index=False)

    print(f"Full dataset:   {len(df):>5} rows")
    print(f"Reporting set:  {len(reporting):>5} rows -> {args.reporting_out}")
    print(f"Remaining set:  {len(remaining):>5} rows -> {args.remaining_out}")
    print("\nChurn rate (full vs reporting):")
    print(f"  full:      {(df['Churn'] == 'Yes').mean():.3%}")
    print(f"  reporting: {(reporting['Churn'] == 'Yes').mean():.3%}")


if __name__ == "__main__":
    main()