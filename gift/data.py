# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
"""Load and check the gift-problem data.

Builds the arrays that every later step uses:
    names   item names, in items.json order (A1..A60)
    prices  item prices (whole numbers)
    A       0/1 table: A[j, i] = 1 if package j contains item i
    b       measured volume of each package, in litres

Decisions applied here (DECISIONS.md records who made each one):
    #16  Item combinations that appear more than once are kept as separate
         measurements. (AI proposal, approved by the author.)
    #17  Any failed check stops with a DataError. (AI proposal, approved by
         the author.)
    #18  Negative measured volumes are kept as measurement noise. (AI proposal,
         approved by the author.) A negative volume cannot happen in reality and
         would not appear in a real data set. (The author's addition.)
    #20  Extra checks beyond the agreed list: prices are positive whole numbers,
         volumes are finite numbers, and both files have the expected structure.
         (AI addition, approved by the author.)

Run `python -m gift.data` to print a summary of the data.
"""

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ITEMS_PATH = ROOT / "items.json"
PACKAGES_PATH = ROOT / "packages.json"
MAX_PRICE = np.iinfo(np.int64).max


class DataError(ValueError):
    """The input data failed a check."""


@dataclass(frozen=True)
class GiftData:
    names: list[str]
    prices: np.ndarray
    A: np.ndarray
    b: np.ndarray


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _as_finite_float(x):
    """Return x as a float, or None if it is not a finite number (bools and huge ints included)."""
    if not isinstance(x, (int, float)) or isinstance(x, bool):
        return None
    try:
        x = float(x)
    except OverflowError:
        return None
    return x if math.isfinite(x) else None


def _names(names):
    return ", ".join(map(repr, names))


def build(items, packages):
    """Check the parsed JSON and build GiftData. Raises DataError on the first failed check."""
    if not isinstance(items, list) or not items:
        raise DataError("items.json must be a non-empty list of items")
    names, prices = [], []
    for k, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise DataError(f"item #{k} in items.json is not an object: {item!r}")
        name, price = item.get("name"), item.get("price")
        if not isinstance(name, str) or not name:
            raise DataError(f"item #{k} in items.json has no valid name: {item!r}")
        if not _is_int(price) or not 0 < price <= MAX_PRICE:
            raise DataError(
                f"item {name!r} has price {price!r}; prices must be positive whole numbers, "
                "written as integers (10, not 10.0)"
            )
        names.append(name)
        prices.append(price)
    duplicates = sorted(n for n, c in Counter(names).items() if c > 1)
    if duplicates:
        raise DataError(f"item names appear more than once in items.json: {_names(duplicates)}")
    column = {name: i for i, name in enumerate(names)}

    if not isinstance(packages, list) or not packages:
        raise DataError("packages.json must be a non-empty list of packages")
    A = np.zeros((len(packages), len(names)))
    b = np.empty(len(packages))
    for j, package in enumerate(packages):
        label = f"package #{j + 1}"
        if not isinstance(package, dict):
            raise DataError(f"{label} is not an object: {package!r}")
        volume, contents = _as_finite_float(package.get("total_volume")), package.get("items")
        if volume is None:
            raise DataError(
                f"{label} has total_volume {package.get('total_volume')!r}; it must be a finite number"
            )
        if not isinstance(contents, list):
            raise DataError(f"{label} has items {contents!r}; it must be a list of item names")
        if not contents:
            raise DataError(f"{label} has no items")
        if not all(isinstance(n, str) for n in contents):
            raise DataError(f"{label} has an entry that is not an item name: {contents!r}")
        unknown = sorted(set(contents) - column.keys())
        if unknown:
            raise DataError(f"{label} contains items not in items.json: {_names(unknown)}")
        repeated = sorted(n for n, c in Counter(contents).items() if c > 1)
        if repeated:
            raise DataError(f"{label} lists the same item more than once: {_names(repeated)}")
        A[j, [column[n] for n in contents]] = 1
        b[j] = volume

    never = [names[i] for i in np.flatnonzero(A.sum(axis=0) == 0)]
    if never:
        raise DataError(
            f"these items appear in no package, so their volume cannot be estimated: {_names(never)}"
        )
    rank = np.linalg.matrix_rank(A)
    if rank < len(names):
        raise DataError(
            f"the package table has rank {rank} < {len(names)} items, so some item volumes "
            "cannot be told apart (for example, items that always appear together)"
        )
    return GiftData(names, np.array(prices, dtype=np.int64), A, b)


def load(items_path=ITEMS_PATH, packages_path=PACKAGES_PATH):
    """Read both JSON files and return checked GiftData."""
    with open(items_path, encoding="utf-8") as f:
        items = json.load(f)
    with open(packages_path, encoding="utf-8") as f:
        packages = json.load(f)
    return build(items, packages)


def summary(data):
    """Facts about the data for the report. Package numbers are 1-based, as in packages.json."""
    sizes = data.A.sum(axis=1).astype(int)
    counts = data.A.sum(axis=0).astype(int)
    combos = Counter(tuple(np.flatnonzero(row)) for row in data.A)
    return {
        "n_items": len(data.names),
        "n_packages": len(data.b),
        "price_range": (int(data.prices.min()), int(data.prices.max())),
        "items_per_package": (int(sizes.min()), float(np.median(sizes)), int(sizes.max())),
        "packages_per_item": (int(counts.min()), float(np.median(counts)), int(counts.max())),
        "least_frequent_items": [data.names[i] for i in np.flatnonzero(counts == counts.min())],
        "most_frequent_items": [data.names[i] for i in np.flatnonzero(counts == counts.max())],
        "rank": int(np.linalg.matrix_rank(data.A)),
        "repeated_combinations": [
            {"items": [data.names[i] for i in combo], "times": times}
            for combo, times in combos.items()
            if times > 1
        ],
        "negative_volumes": [
            {"package": int(j) + 1, "volume": float(data.b[j]),
             "items": [data.names[i] for i in np.flatnonzero(data.A[j])]}
            for j in np.flatnonzero(data.b < 0)
        ],
    }


def main():
    data = load()
    s = summary(data)
    lo, med, hi = s["items_per_package"]
    plo, pmed, phi = s["packages_per_item"]
    print(f"Items: {s['n_items']}, prices {s['price_range'][0]} to {s['price_range'][1]}")
    print(f"Packages: {s['n_packages']}")
    print(f"Items per package: min {lo}, median {med:g}, max {hi}")
    print(f"Packages per item: min {plo} ({', '.join(s['least_frequent_items'])}), "
          f"median {pmed:g}, max {phi} ({', '.join(s['most_frequent_items'])})")
    print(f"Package table: {data.A.shape[0]} x {data.A.shape[1]}, rank {s['rank']}")
    print(f"Item combinations appearing more than once: {len(s['repeated_combinations'])} "
          "(kept as separate measurements, #16)")
    for r in s["repeated_combinations"]:
        print(f"  {' '.join(r['items'])}: {r['times']} times")
    print(f"Negative measured volumes: {len(s['negative_volumes'])} "
          "(kept as measurement noise, #18; impossible in reality, would not appear in a real data set)")
    for n in s["negative_volumes"]:
        print(f"  package #{n['package']}: {n['volume']} L, items {' '.join(n['items'])}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
