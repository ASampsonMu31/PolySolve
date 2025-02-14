from __future__ import annotations
import numpy as np

CBRT_UNITY_IM = np.sqrt(3)/2 * 1j


def solve_quadratic(
        a: float,
        b:float,
        c:float,
) -> tuple[float, float]:
    """
    Solves the roots of a quadratic equation.
    Uses the quadratic formula. Result must be real.

    Parameters
    ----------------
    a
        :math: 'x^2' coefficient.
    b
        :math: 'x' coefficient
    c
        Constant value.
        
    Returns
    ---------
    tuple[float, float]
        Positive and negative roots of the quadratic.
    """
    
    det = b**2 - (4*a*c)

    return ((-b + np.sqrt(det)) / (2*a),
            (-b - np.sqrt(det)) / (2*a))

def solve_cubic(
        a: float,
        b: float,
        c: float,
        d: float,
) -> tuple[float, float, float]:
    """
    Solves the roots of a cubic equation.
    Uses the cubic formula. Result must be real.

    Parameters
    ----------------
    a
        :math: 'x^3' coefficient.
    b
        :math: 'x^2' coefficient
    c
        :math: 'x' coefficient
    d
        Constant value.
        
    Returns
    ---------
    tuple[float, float]
        Positive and negative roots of the cubic.
    """
    q = (3*a*c - b**2) / (9*a**2)
    r = (9*a*b*c - 27*a**2*d - 2*b**3) / (54*a**3)

    s = np.cbrt(r + np.sqrt(q**3 + r**2))
    t = np.cbrt(r - np.sqrt(q**3 + r**2))

    x1 = s + t - (b/3*a)
    x2 = -(s + t)/2 - (b/3*a) + CBRT_UNITY_IM * (s - t)
    x3 = -(s + t)/2 - (b/3*a) - CBRT_UNITY_IM * (s - t)
    return x1, x2, x3
