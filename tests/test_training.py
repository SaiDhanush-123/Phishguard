import numpy as np

from phishguard.training import assert_no_group_leakage, grouped_partitions


def test_grouped_partitions_have_no_domain_leakage():
    groups = np.repeat(np.array([f"domain-{index}.test" for index in range(80)]), 3)
    labels = np.repeat(np.array([index % 2 for index in range(80)]), 3)

    partitions = grouped_partitions(labels, groups, seed=42)
    counts = assert_no_group_leakage(partitions, groups)

    assert set(partitions) == {"train", "calibration", "validation", "test"}
    assert sum(len(indices) for indices in partitions.values()) == len(labels)
    assert sum(counts.values()) == len(set(groups))
