# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
import json
import subprocess
import sys

import numpy as np
import pytest

from gift.data import ITEMS_PATH, PACKAGES_PATH, ROOT, DataError, build, load, summary


@pytest.fixture(scope="module")
def data():
    return load()


def test_real_data_shapes(data):
    assert data.A.shape == (1000, 60)
    assert data.b.shape == (1000,)
    assert data.prices.shape == (60,)
    assert set(np.unique(data.A)) == {0.0, 1.0}


def test_names_and_prices_match_items_json(data):
    items = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
    assert data.names == [item["name"] for item in items]
    assert data.prices.dtype.kind == "i"
    assert data.prices.tolist() == [item["price"] for item in items]


def test_every_row_matches_packages_json(data):
    packages = json.loads(PACKAGES_PATH.read_text(encoding="utf-8"))
    for j, package in enumerate(packages):
        assert sorted(data.names[i] for i in np.flatnonzero(data.A[j])) == sorted(package["items"])
        assert data.b[j] == package["total_volume"]


def test_first_package(data):
    assert {data.names[i] for i in np.flatnonzero(data.A[0])} == {"A28", "A3", "A33", "A59"}
    assert data.b[0] == 36.04


def test_negative_volumes_are_kept_and_reported(data):
    negative = summary(data)["negative_volumes"]
    assert [(n["package"], n["volume"], n["items"]) for n in negative] == [
        (663, -0.85, ["A39"]),
        (790, -0.34, ["A6"]),
    ]


def test_repeated_combinations_are_kept(data):
    s = summary(data)
    assert s["n_packages"] == 1000
    assert sorted(r["times"] for r in s["repeated_combinations"]) == [2] * 13 + [3] * 2


def test_summary_facts(data):
    s = summary(data)
    assert s["items_per_package"] == (1, 4.0, 10)
    assert s["packages_per_item"] == (52, 69.0, 85)
    assert s["price_range"] == (21, 119)
    assert s["least_frequent_items"] == ["A6", "A10"]
    assert s["most_frequent_items"] == ["A47"]
    assert s["rank"] == 60


def test_summary_command():
    result = subprocess.run(
        [sys.executable, "-m", "gift.data"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "package #663: -0.85 L, items A39" in result.stdout
    assert "package #790: -0.34 L, items A6" in result.stdout
    assert result.stdout.rstrip().endswith("All checks passed.")


def valid_example():
    items = [{"name": "A", "price": 10}, {"name": "B", "price": 20}, {"name": "C", "price": 30}]
    packages = [
        {"total_volume": 1.0, "items": ["A"]},
        {"total_volume": 2.0, "items": ["B"]},
        {"total_volume": 3.0, "items": ["C"]},
        {"total_volume": 3.1, "items": ["A", "B"]},
    ]
    return items, packages


def test_valid_example_builds():
    d = build(*valid_example())
    assert d.A.shape == (4, 3)
    assert d.A[3].tolist() == [1.0, 1.0, 0.0]


def duplicate_name(items, packages):
    items.append({"name": "A", "price": 5})


def unknown_item(items, packages):
    packages[0]["items"].append("Z")


def unknown_item_with_space(items, packages):
    packages[0]["items"] = ["A "]


def item_twice_in_package(items, packages):
    packages[3]["items"].append("A")


def empty_package(items, packages):
    packages[0]["items"].clear()


def items_not_a_list(items, packages):
    packages[0]["items"] = "A"


def item_in_no_package(items, packages):
    items.append({"name": "D", "price": 40})


def items_always_together(items, packages):
    # B and C only ever appear together, so their volumes cannot be separated.
    packages[:] = [
        {"total_volume": 1.0, "items": ["A"]},
        {"total_volume": 5.0, "items": ["B", "C"]},
        {"total_volume": 6.0, "items": ["A", "B", "C"]},
    ]


def price(value):
    def set_price(items, packages):
        items[0]["price"] = value
    return set_price


def volume(value):
    def set_volume(items, packages):
        packages[0]["total_volume"] = value
    return set_volume


@pytest.mark.parametrize(
    "break_data, message",
    [
        (duplicate_name, "more than once in items.json: 'A'"),
        (unknown_item, "not in items.json: 'Z'"),
        (unknown_item_with_space, "not in items.json: 'A '"),
        (item_twice_in_package, "same item more than once: 'A'"),
        (empty_package, "has no items"),
        (items_not_a_list, "must be a list of item names"),
        (item_in_no_package, "appear in no package.*: 'D'"),
        (items_always_together, "rank 2 < 3"),
        (price(10.5), "positive whole numbers"),
        (price(10.0), "positive whole numbers"),
        (price(0), "positive whole numbers"),
        (price(-5), "positive whole numbers"),
        (price(True), "positive whole numbers"),
        (price(10**30), "positive whole numbers"),
        (volume("1.0"), "finite number"),
        (volume(float("nan")), "finite number"),
        (volume(float("inf")), "finite number"),
        (volume(True), "finite number"),
        (volume(10**400), "finite number"),
    ],
)
def test_broken_data_stops_with_clear_error(break_data, message):
    items, packages = valid_example()
    break_data(items, packages)
    with pytest.raises(DataError, match=message):
        build(items, packages)
