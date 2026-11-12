"""Viscous-inviscid airfoil analysis: panel method + coupled integral boundary layer.

Two-equation integral boundary layer (momentum and kinetic-energy shape
equations) with the Drela & Giles (1987) / Drela (1989) laminar and turbulent
closures, a shear-lag equation for the turbulent stress, the envelope e^N
amplification equation for transition, and a wake, solved simultaneously with
the inviscid panel method by Newton's method. Written from the published
equations (Drela & Giles 1987, AIAA J. 25(10); Drela 1989, Lecture Notes in
Engineering 54; Fidkowski 2022, AIAA J. 60(5), "A coupled inviscid-viscous
airfoil analysis solver, revisited"). Incompressible (Mach 0).
"""

from .solver import IblResult, analyse

__all__ = ["IblResult", "analyse"]
