import json
from pathlib import Path

import pandas as pd

from phishguard.data_audit import audit_dataset


def test_dataset_audit_records_mapping_and_conflicts(tmp_path: Path):
    dataset = tmp_path / "data.csv"
    output = tmp_path / "audit.json"
    pd.DataFrame(
        {
            "URL": ["https://example.com/a", "https://example.com/b", "https://other.org"],
            "label": [1, 0, 1],
            "ignored_page_feature": [1, 2, 3],
        }
    ).to_csv(dataset, index=False)

    report = audit_dataset(dataset, output)

    assert report["domains_with_conflicting_labels"] == 1
    assert report["excluded_column_count"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["label_mapping"]["raw_0"] == "phishing"
