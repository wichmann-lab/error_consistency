"""
Tests for utility methods.
These are only unit tests designed to prevent software development errors.
We do not base our trust in the methods on these tests.
"""

import numpy as np
from pytest import approx, raises

from utils import *


def test_is_binary():
    assert is_binary(np.array([0, 1, 0, 1, 1, 0]))
    assert is_binary(np.array([0.0, 1.0, -0.0]))
    assert not is_binary(np.array([0, 1, 2, 4]))
    assert is_binary(np.array([]))


def test_calculate_kappa():
    with raises(AssertionError):
        calculate_kappa(-0.3, 0.3)
        calculate_kappa(1.3, 0.3)
        calculate_kappa(0.3, -0.3)
        calculate_kappa(0.3, 1.3)

    assert np.isnan(calculate_kappa(0.3, 1.0))
    assert np.isnan(calculate_kappa(1, 1))
    assert calculate_kappa(0.8, 0.5) == approx(0.6)
    assert calculate_kappa(0.3, 0.3) == approx(0)
    assert calculate_kappa(0.4, 0.6) == approx(-0.5)


def test_fast_cohen():
    with raises(AssertionError):
        fast_cohen([0, 1, 0], np.array([1, 0, 0]))
        fast_cohen(np.array([0, 1, 0]), [1, 0, 0])
        fast_cohen(np.array([0, 1, 0]), np.array([0, 1, 1, 0]))
        fast_cohen(np.array([[0, 1, 0]]), np.array([0, 1, 0]))

    a = np.ones(5)
    b = np.ones_like(a)
    c = np.array([0, 0, 1, 1])
    d = np.array([1, 1, 0, 0])
    e = np.array([0, 1, 0, 1])

    assert np.isnan(fast_cohen(a, b))
    assert np.isnan(fast_cohen(np.zeros_like(a), np.zeros_like(b)))
    assert fast_cohen(c, d) == approx(-1)
    assert fast_cohen(c, c) == approx(1)
    assert fast_cohen(c, e) == approx(0)

    assert fast_cohen(
        np.array([0, 1, 0, 1, 1, 1, 1]), np.array([0, 1, 1, 0, 0, 0, 0]), True
    ) == approx(-0.25)


def test_kappa_max():
    with raises(AssertionError):
        kappa_max(-0.1, 0.5)
        kappa_max(0.5, -0.1)
        kappa_max(1.4, 0.5)
        kappa_max(0.5, 1.5)

    assert kappa_max(0.8, 0.6) == approx(0.54545454545454)


def test_simulate_trials_from_copy_model():
    with raises(AssertionError):
        simulate_trials_from_copy_model(0, -0.5, 0.5, 100)
        simulate_trials_from_copy_model(0, 0.5, -0.5, 100)
        simulate_trials_from_copy_model(0, 1.5, 0.5, 100)
        simulate_trials_from_copy_model(0, 0.5, 1.5, 100)
        simulate_trials_from_copy_model(0, 0.5, 0.5, 0)
        simulate_trials_from_copy_model(0, 0.5, 0.5, -100)
        simulate_trials_from_copy_model(0.9, 0.9, 0.1, 1000)
        simulate_trials_from_copy_model(-0.9, 0.8, 0.8, 1000)

    def _test(kappa, acc1, acc2, n_trials):
        ecs = []
        t1s = []
        t2s = []
        for i in range(1000):
            t1, t2 = simulate_trials_from_copy_model(kappa, acc1, acc2, n_trials)
            assert len(t1) == len(t2) == n_trials
            ec = fast_cohen_core(t1, t2)
            ecs.append(ec)
            t1s.append(np.mean(t1))
            t2s.append(np.mean(t2))
        assert np.mean(t1s) == approx(acc1, abs=0.01)
        assert np.mean(t2s) == approx(acc2, abs=0.01)
        assert np.mean(ecs) == approx(kappa, abs=0.01)

    _test(0.0, 0.5, 0.5, 1000)
    _test(0.2, 0.6, 0.7, 1000)
    _test(-0.6, 0.5, 0.5, 1000)


def test_simulate_trials_exact():
    with raises(AssertionError):
        simulate_trials_exact(0.0, 1.5, 0.8, 10)
        simulate_trials_exact(0.0, -0.5, 0.8, 10)
        simulate_trials_exact(0.0, 0.8, 1.8, 10)
        simulate_trials_exact(0.0, 0.5, -0.8, 10)
        simulate_trials_exact(0.0, 0.5, 0.8, 0)
        simulate_trials_exact(0.0, 0.5, 0.8, -100)
        simulate_trials_exact(0.9, 0.1, 0.8, 100)  # impossible kappa

    def _test(kappa, acc1, acc2, n_trials, shuffle):
        ecs = []
        t1s = []
        t2s = []
        for i in range(1000):
            _, t1, t2 = simulate_trials_exact(kappa, acc1, acc2, n_trials, shuffle)
            assert len(t1) == len(t2) == n_trials
            ec = fast_cohen_core(t1, t2)
            ecs.append(ec)
            t1s.append(np.mean(t1))
            t2s.append(np.mean(t2))
        assert np.mean(t1s) == approx(acc1, abs=0.01)
        assert np.mean(t2s) == approx(acc2, abs=0.01)
        assert np.mean(ecs) == approx(kappa, abs=0.01)

    _test(0.5, 0.75, 0.65, 1000, False)
    _test(0.5, 0.75, 0.65, 1000, True)
    _test(-0.2, 0.75, 0.65, 1000, False)


def test_calculate_ec_pvalue():
    with raises(AssertionError):
        calculate_ec_pvalue(1.5, 50, 50, 100, 1000)
        calculate_ec_pvalue(-1.5, 50, 50, 100, 1000)
        calculate_ec_pvalue(0.5, 100, 10, 80, 1000)
        calculate_ec_pvalue(0.5, 10, 100, 80, 1000)
        calculate_ec_pvalue(0.5, 50, 50, 100, 0)
        calculate_ec_pvalue(0.5, 50, 50, 100, -100)

    assert calculate_ec_pvalue(0.5, 50, 50, 100, 1000) < 1e-5
    assert calculate_ec_pvalue(0.001, 70, 70, 100, 1000) > 0.95


def test_error_consistency():
    with raises(AssertionError):
        error_consistency(np.array([0, 1, 0]), [1, 0, 1], 100)
        error_consistency([0, 1, 0], np.array([1, 0, 1]), 100)
        error_consistency(np.array([1, 0, 1]), np.array([1, 0, 0, 1]), 100)
        error_consistency(np.array([1, 0, 2]), np.array([1, 0, 1]), 100)
        error_consistency(np.array([1, 0, 1]), np.array([1, 1, 0]), 0)
        error_consistency(np.array([1, 0, 1]), np.array([1, 1, 0]), -10)

    ec, p = error_consistency(
        np.array([1, 1, 1, 0, 0, 0]), np.array([1, 1, 0, 0, 0, 0]), 1000
    )
    assert ec == approx(0.6666666)
    assert p < 0.05
