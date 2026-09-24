# AI-drafted (Claude Code, Claude Opus 5.5). Author review: see DECISIONS.md.
import json

import numpy as np
import pytest

from gift.data import load
from gift.estimate import estimate
from gift.knapsack import (
    alternative_readings,
    exact_table,
    fractional_bound,
    greedy,
    integer_program,
    main,
    near_optimal,
    solve,
)


def all_subsets(prices, volumes, capacity):
    """Brute force: price and volume of every subset that fits, as (mask, price, volume)."""
    n = len(prices)
    masks = np.arange(2 ** n)
    bits = (masks[:, None] >> np.arange(n)) & 1
    price = bits @ np.asarray(prices)
    volume = bits @ np.asarray(volumes, dtype=float)
    fits = volume <= capacity + 1e-9
    return masks[fits], price[fits], volume[fits]


def random_instances(count, seed):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        n = int(rng.integers(1, 15))
        prices = rng.integers(1, 121, size=n).tolist()
        volumes = rng.uniform(0.1, 30, size=n).tolist()
        yield prices, volumes, float(rng.uniform(0, 60))


def best_brute_force(prices, volumes, capacity):
    _, price, _ = all_subsets(prices, volumes, capacity)
    return int(price.max())


@pytest.mark.parametrize("seed", range(3))
def test_exact_table_matches_brute_force(seed):
    for prices, volumes, capacity in random_instances(100, seed):
        selection, _ = exact_table(prices, volumes, capacity)
        assert selection.price == best_brute_force(prices, volumes, capacity)
        assert selection.volume <= capacity + 1e-9
        assert selection.price == sum(prices[i] for i in selection.items)


def test_integer_program_matches_brute_force():
    for prices, volumes, capacity in random_instances(100, seed=10):
        selection, _ = integer_program(prices, volumes, capacity)
        assert selection.price == best_brute_force(prices, volumes, capacity)
        assert selection.volume <= capacity + 1e-9


def test_greedy_and_bound_bracket_the_optimum():
    for prices, volumes, capacity in random_instances(200, seed=20):
        best = best_brute_force(prices, volumes, capacity)
        heuristic, _ = greedy(prices, volumes, capacity)
        assert heuristic.volume <= capacity + 1e-9
        assert heuristic.price <= best <= fractional_bound(prices, volumes, capacity) + 1e-9


def test_greedy_can_be_far_off():
    # X: 1 L at price 2 (2 per litre). Y: 40 L at price 40 (1 per litre).
    # Greedy takes X first, and then Y no longer fits.
    prices, volumes = [2, 40], [1.0, 40.0]
    assert greedy(prices, volumes)[0].price == 2
    assert exact_table(prices, volumes)[0].price == 40


def test_near_optimal_matches_brute_force():
    for prices, volumes, capacity in random_instances(100, seed=30):
        best = best_brute_force(prices, volumes, capacity)
        masks, price, _ = all_subsets(prices, volumes, capacity)
        expected = sorted(
            tuple(i for i in range(len(prices)) if m >> i & 1) for m, p in zip(masks, price) if p >= best - 40
        )
        found = sorted(s.items for s in near_optimal(prices, volumes, best, window=40, capacity=capacity))
        assert found == expected


def test_edge_cases():
    # Nothing fits.
    assert exact_table([5, 6], [50.0, 60.0])[0].items == ()
    # Everything fits.
    assert exact_table([5, 6], [10.0, 20.0])[0].items == (0, 1)
    # A set that fills exactly 40 L is allowed (#35).
    assert exact_table([10, 20, 30], [10.0, 20.0, 10.0])[0].items == (0, 1, 2)
    # Ties on price: the set with the smaller volume is chosen.
    selection, _ = exact_table([10, 10], [30.0, 20.0], capacity=30.0)
    assert selection.items == (1,)


def test_alternative_readings():
    # Two items; with repeats the best is 4 copies of the first (price 4 x 3 = 12 in 40 L).
    alt = alternative_readings([3, 5], [10.0, 25.0])
    assert alt["repeats_allowed"] == {0: 4}
    assert alt["single_item"] == [1]
    # The single item must fit; a tie lists every tied item, smallest volume first.
    assert alternative_readings([3, 5], [10.0, 45.0])["single_item"] == [0]
    assert alternative_readings([119, 119], [29.67, 11.18])["single_item"] == [1, 0]
    assert alternative_readings([7, 8], [41.0, 50.0])["single_item"] == []
    # 100 copies of a 0.4 L item fill 40 L exactly, although 40 // 0.4 == 99 in float.
    assert alternative_readings([1], [0.4])["repeats_allowed"] == {0: 100}
    # 0.1 + 0.2 is 0.30000000000000004 in float, and must still fit in 0.3.
    alt = alternative_readings([1, 2], [0.1, 0.2], capacity=0.3)
    assert sum([1, 2][k] * c for k, c in alt["repeats_allowed"].items()) == 3


# --- Float sums just over the capacity (TOLERANCE) and other boundaries -------------------------

def test_float_sum_just_over_capacity_still_fits():
    # 0.1 + 0.2 = 0.30000000000000004 in float.
    assert exact_table([1, 1], [0.1, 0.2], capacity=0.3)[0].items == (0, 1)
    assert greedy([1, 1], [0.1, 0.2], capacity=0.3)[0].items == (0, 1)
    assert [s.items for s in near_optimal([1, 1], [0.1, 0.2], 2, window=0, capacity=0.3)] == [(0, 1)]
    # A volume just inside the tolerance fits.
    assert exact_table([1], [40 + 1e-9])[0].items == (0,)


def test_near_optimal_does_not_cut_a_set_at_the_threshold():
    # The root bound here is 96.99999999999999, so a cut without the 1e-9 slack loses the set.
    found = near_optimal([37, 60], [4.83, 16.8], 97, window=0, capacity=21.63)
    assert [s.items for s in found] == [(0, 1)]
    # A set exactly at the threshold whose volume is just over 40 L (within TOLERANCE).
    found = near_optimal([100, 100, 1, 240], [20.0, 20.0000000005, 0.25, 39.9], 240, window=40)
    assert (0, 1) in [s.items for s in found]


def test_near_optimal_order():
    found = near_optimal([10, 10], [30.0, 20.0], 10, window=0, capacity=30.0)
    assert [s.items for s in found] == [(1,), (0,)]


def test_integer_program_matches_exact_table_on_larger_cases():
    # With 15 to 30 items the solver sometimes leaves values like 5e-15 for unused items.
    prices = [20, 53, 112, 114, 24, 18, 74, 112, 41]
    volumes = [13.188, 8.736, 22.405, 25.951, 27.963, 24.107, 4.903, 15.541, 5.531]
    assert integer_program(prices, volumes, 55.776)[0] == exact_table(prices, volumes, 55.776)[0]
    rng = np.random.default_rng(40)
    for _ in range(100):
        n = int(rng.integers(15, 31))
        prices = rng.integers(1, 121, size=n).tolist()
        volumes = rng.uniform(0.1, 30, size=n).tolist()
        capacity = float(rng.uniform(0, 100))
        assert integer_program(prices, volumes, capacity)[0].price == exact_table(prices, volumes, capacity)[0].price


def test_integer_program_rejects_a_set_over_the_capacity():
    # The solver's own tolerance (about 1e-7) would accept 40.00000005 L.
    with pytest.raises(RuntimeError, match="over the capacity"):
        integer_program([10, 1], [40.00000005, 1.0])


def test_non_positive_volumes_are_rejected():
    for method in [exact_table, integer_program, greedy, fractional_bound, alternative_readings]:
        with pytest.raises(ValueError, match="positive"):
            method([10, 5], [30.0, -1.0])
    with pytest.raises(ValueError, match="positive"):
        near_optimal([10, 5], [30.0, 0.0], 10)


# --- The real data ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def report():
    data = load()
    return solve(data, estimate(data).volumes)


def test_real_exact_methods(report):
    for method in ["exact_table", "integer_program"]:
        r = report["methods"][method]
        assert r["price"] == 757
        assert r["items"] == ["A6", "A8", "A9", "A23", "A32", "A35", "A38", "A44", "A48"]
        assert r["volume_l"] == pytest.approx(39.972, abs=1e-3)
        assert r["slack_l"] == pytest.approx(0.028, abs=1e-3)
    assert report["exact_methods_agree"]
    assert report["exact_methods_same_items"]


def test_real_greedy_and_bound(report):
    g = report["methods"]["greedy"]
    assert g["price"] == 751
    assert g["items"] == ["A6", "A8", "A9", "A23", "A32", "A35", "A38", "A39", "A44", "A53"]
    assert report["greedy_gap"] == 6
    assert report["fractional_upper_bound"] == pytest.approx(770.8, abs=0.05)


def test_real_near_optimal(report):
    near = report["near_optimal"]
    assert len(near) == 51
    assert report["sets_at_best_price"] == 1
    assert [s["price"] for s in near[:5]] == [757, 751, 741, 737, 736]
    assert near[0]["items"] == report["methods"]["exact_table"]["items"]
    assert near == sorted(near, key=lambda s: (-s["price"], s["volume_l"]))


def test_real_alternative_readings(report):
    alt = report["alternative_readings"]
    assert alt["repeats_allowed"] == {"items": {"A6": 106}, "price": 11554}
    assert alt["single_item"] == {"items": ["A49", "A14"], "price": 119}


def test_main_writes_report(tmp_path, capsys):
    main(tmp_path)
    saved = json.loads((tmp_path / "knapsack.json").read_text(encoding="utf-8"))
    assert saved["methods"]["exact_table"]["price"] == 757
    assert "Exact methods agree: True" in capsys.readouterr().out
