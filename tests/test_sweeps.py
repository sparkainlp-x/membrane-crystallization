import math

from membrane_crystallization.__main__ import main
from membrane_crystallization.sweeps import (entropy_curve, forgetting,
                                             recreation, whiteness_controls)


def test_forgetting_matches_exponential_window():
    for r in forgetting(alphas=(0.05,), seeds=10):
        assert r["tracked"] == 10
        assert abs(r["measured"][0] - r["predicted"]) < 3
        assert abs(r["predicted"] - math.log(2) / 0.05) < 0.5


def test_whiteness_does_not_separate_patterns_from_pure_noise():
    w = whiteness_controls(seeds=3)
    iso = w["3 patterns, isotropic noise"]
    noise = w["noise"]
    assert noise[3][0] >= iso[3][0] - 0.01        # pure noise looks as white
    assert iso[1][0] < iso[3][0]                  # k too small is lower
    assert w["3 patterns, coloured noise"][3][0] < 0.9   # coloured noise is not white


def test_entropy_curve_saturates_early():
    c = entropy_curve(seeds=3)
    rows = {r["t"]: r for r in c["rows"]}
    assert rows[1]["entropy"][0] < 1e-9
    assert abs(rows[6000]["entropy"][0] - rows[200]["entropy"][0]) < 0.1
    assert rows[6000]["entropy"][0] < c["max_entropy"] - 0.1


def test_recreation_decays_with_sigma():
    r = recreation(seeds=3, sigmas=(0.1, 0.5), generations=3, steps=1500)
    lo, hi = r["by_sigma"][0.1], r["by_sigma"][0.5]
    assert lo[-1]["match"][0] > hi[-1]["match"][0]
    assert hi[-1]["match"][0] < hi[0]["match"][0]


def test_cli_quick(capsys):
    assert main(["--quick"]) == 0
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "Verified" not in out
