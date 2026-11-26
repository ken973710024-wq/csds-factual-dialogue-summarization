"""Recalculate reported aggregate ratios (not claim annotations or bootstrap CIs)."""
import csv
from pathlib import Path


def main():
    path = Path(__file__).resolve().parents[1] / "results/handoff_dev60_counts.csv"
    values = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            coverage = int(row["covered_key_facts"]) / int(row["total_key_facts"])
            support = int(row["supported_claims"]) / int(row["judged_claims"])
            values[row["arm"]] = coverage, support
            print(f'{row["arm"]:10s} coverage={coverage:.2%}  supported claims={support:.2%}')
    a, b = values["original"], values["handoff"]
    print(f"Handoff − Original: coverage={(b[0]-a[0])*100:+.2f} pp; supported claims={(b[1]-a[1])*100:+.2f} pp")


if __name__ == "__main__":
    main()
