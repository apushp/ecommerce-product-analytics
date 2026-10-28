"""
Pulls the thelook_ecommerce tables from BigQuery's public datasets and saves
them as parquet in data/raw/thelook/.

thelook is regenerated every day, so each pull is a slightly different
snapshot. The pull date gets written to data/raw/thelook/_snapshot.txt so the
numbers in the report can be traced back to it.

Needs a Google Cloud project (the free BigQuery sandbox works) and
`gcloud auth application-default login` run once.

usage: python scripts/01_extract_thelook.py --project <your-gcp-project-id>
"""
import argparse
import datetime as dt
from pathlib import Path

from google.cloud import bigquery

TABLES = [
    "users",
    "orders",
    "order_items",
    "products",
    "inventory_items",
    "distribution_centers",
    "events",
]

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "thelook"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="GCP project to bill the read to")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in TABLES:
        table = f"bigquery-public-data.thelook_ecommerce.{name}"
        df = client.list_rows(table).to_dataframe(create_bqstorage_client=True)
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        print(f"{name:22s} {len(df):>10,} rows")

    (OUT_DIR / "_snapshot.txt").write_text(dt.date.today().isoformat() + "\n")


if __name__ == "__main__":
    main()
