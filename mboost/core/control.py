from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoostControl:
    mstop: int = 100
    nu: float = 0.1

    def __post_init__(self) -> None:
        if self.mstop <= 0:
            raise ValueError("mstop must be positive")
        if not (0.0 < self.nu <= 1.0):
            raise ValueError("nu must be in the interval (0, 1]")


def boost_control(*, mstop: int = 100, nu: float = 0.1) -> BoostControl:
    return BoostControl(mstop=mstop, nu=nu)
