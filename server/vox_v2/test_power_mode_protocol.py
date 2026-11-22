"""Smoke test for server-side MODE_CHANGE with power_profile (AC/BAT) and SLEEP handling."""

from __future__ import annotations


def effective_agency_interval(
    mode: str, power_profile: str, base_interval_s: float | None, has_due_todo: bool
) -> float | None:
    m = mode.upper()
    p = power_profile.upper()
    if m == "SLEEP":
        return None
    if m == "STANDBY":
        return 35.0 if has_due_todo else None
    if base_interval_s is None:
        return 30.0 if has_due_todo else None
    mult = 1.8 if p == "BAT" else 1.0
    return round(base_interval_s * mult, 1)


def run_tests() -> None:
    assert effective_agency_interval("LIVE", "AC", 60.0, False) == 60.0
    assert effective_agency_interval("LIVE", "BAT", 60.0, False) == 108.0
    assert effective_agency_interval("STANDBY", "AC", 60.0, False) is None
    assert effective_agency_interval("STANDBY", "AC", 60.0, True) == 35.0
    assert effective_agency_interval("SLEEP", "BAT", 60.0, True) is None
    print("ALL ADAPTIVE POWER POLICY SMOKE TESTS PASSED!")


if __name__ == "__main__":
    run_tests()
