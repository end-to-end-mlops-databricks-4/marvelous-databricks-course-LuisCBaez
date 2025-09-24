from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


CONFIG_PATH = Path(__file__).resolve().parents[2] / "project_config.yaml"


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


raw = load_yaml(CONFIG_PATH)


@dataclass(frozen=True)
class Config:
    catalog: str = raw["dev"]["catalog_name"]
    schema: str = raw["dev"]["schema_name"]

    volume: str = raw["volume"]["name"]
    subdir: str = raw["volume"]["subdir"]

    sales_file: str = raw["raw_files"]["sales"]
    calendar_file: str = raw["raw_files"]["calendar"]
    prices_file: str = raw["raw_files"]["prices"]

    tbl_prepared: str = raw["processed"]["prepared_sales"]
    tbl_train: str = raw["processed"]["train"]
    tbl_test: str = raw["processed"]["test"]

    holdout_months: int = int(raw.get("holdout_months", 4))

    def volume_file(self, filename: str) -> str:
        """
        Use dbfs:/ path so both CLI and Spark (Databricks Connect).
        Example: dbfs:/Volumes/mlops_dev/lcbaezlo/m5/raw/sales.csv
        """
        return f"dbfs:/Volumes/{self.catalog}/{self.schema}/{self.volume}/{self.subdir}/{filename}"


CFG = Config()
