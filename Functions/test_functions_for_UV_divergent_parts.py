import sys
from sympy import *
from Functions.Global_variables import *
from Functions.SymPy_classes import *
from typing import Any

# ------------------------------------------------------------------------------------------------------------------#
#                 Answers for integrals over frequencies from the scalar parts of UV divergent diagrams
# ------------------------------------------------------------------------------------------------------------------#


def ScalarTypeIaAfterIntOverFreq():
    """
    Diagram: e12|e3|33||:0B_bB_vV|0b_vv|vV_vv||
    """
    ScalarTypeIaAfterIntOverFreq = (
        pi**2
        * k ** (2 - d - 2 * eps)
        * q ** (2 - d - 2 * eps)
        * go**2
        * nuo
        * (7 * k**2 + 6 * q**2 + 6 * k * q * z + 2 * (2 * k**2 + q**2 + k * q * z) * uo + k**2 * uo**2)
        / (
            4
            * k**4
            * (k**2 + q**2 + k * q * z)
            * (1 + uo) ** 2
            * (k**2 + 2 * q**2 + 2 * k * q * z + k**2 * uo)
        )
    )
    return ScalarTypeIaAfterIntOverFreq


def ScalarTypeIbAfterIntOverFreq():
    """
    Diagram: e12|e3|33||:0B_vv_bB|0b_Bb|bB_vv||
    """
    ScalarTypeIbAfterIntOverFreq = (
        pi**2
        * k ** (2 - d - 2 * eps)
        * q ** (2 - d - 2 * eps)
        * go**2
        * nuo
        / (k**4 * (k**2 + q**2 + (k**2 + q**2 + 2 * k * q * z) * uo) * (1 + uo) ** 2)
    )
    return ScalarTypeIbAfterIntOverFreq


def ScalarTypeIcAfterIntOverFreq():
    """
    Diagram: e12|e3|33||:0B_bB_vv|0b_vV|Vv_vv||
    """
    ScalarTypeIcAfterIntOverFreq = (
        pi**2
        * k ** (2 - d - 2 * eps)
        * q ** (2 - d - 2 * eps)
        * go**2
        * nuo
        / (4 * k**4 * (k**2 + q**2 - k * q * z) * (1 + uo))
    )
    return ScalarTypeIcAfterIntOverFreq


def ScalarTypeIdAfterIntOverFreq():
    """
    Diagram: e12|e3|33||:0B_bB_vV|0b_vV|vv_vv||
    """
    ScalarTypeIdAfterIntOverFreq = (
        pi**2
        * k ** (2 - d - 2 * eps)
        * q ** (2 - d - 2 * eps)
        * go**2
        * nuo
        * (2 * (k**2 + q**2 + k * q * z) + (k**2 + q**2 + 2 * k * q * z) * uo)
        / (
            2
            * (k**2 + q**2 + k * q * z)
            * (k**2 + q**2 + 2 * k * q * z) ** 2
            * (1 + uo)
            * (k**2 + q**2 + (k**2 + q**2 + 2 * k * q * z) * uo)
        )
    )
    return ScalarTypeIdAfterIntOverFreq


def ScalarTypeIIaAfterIntOverFreq():
    """
    Diagram: e12|23|3|e|:0B_bB_vV|vv_bB|vv|0b|
    """
    ScalarTypeIIaAfterIntOverFreq = (
        k**2
        * pi**2
        * k ** (-d - 2 * eps)
        * q ** (-d - 2 * eps)
        * go**2
        * nuo
        * (
            6 * k**4
            + 5 * q**4
            - 10 * k**3 * q * z
            - 8 * k * q**3 * z
            + k**2 * q**2 * (11 + 4 * z**2)
            + 2 * (k**2 + 3 * q**2 - 2 * k * q * z) * (k**2 + q**2 - k * q * z) * uo
            + q**2 * (k**2 + q**2 - 2 * k * q * z) * uo**2
        )
        / (
            2
            * (k**2 + q**2 - 2 * k * q * z)
            * (k**2 + q**2 - k * q * z)
            * (1 + uo) ** 2
            * (q**2 + 2 * k * (k - q * z) + q**2 * uo)
            * (k**2 + q**2 + (k**2 + q**2 - 2 * k * q * z) * uo)
        )
    )
    return ScalarTypeIIaAfterIntOverFreq


def ScalarTypeIIbAfterIntOverFreq():
    """
    Diagram: e12|23|3|e|:0B_bB_vv|bB_vv|bB|0b|
    """
    ScalarTypeIIbAfterIntOverFreq = (
        pi**2
        * k ** (-d - 2 * eps)
        * q ** (-d - 2 * eps)
        * go**2
        * nuo
        / ((k**2 + q**2 + (k**2 + q**2 + 2 * k * q * z) * uo) * (1 + uo) ** 2)
    )
    return ScalarTypeIIbAfterIntOverFreq


def ScalarTypeIIcAfterIntOverFreq():
    """
    Diagram: e12|23|3|e|:0B_bB_vv|vV_bB|vv|0b|
    """
    ScalarTypeIIcAfterIntOverFreq = (
        pi**2
        * k ** (-d - 2 * eps)
        * q ** (-d - 2 * eps)
        * go**2
        * nuo
        * (3 * q**2 + 2 * k * (k - q * z) + q**2 * uo)
        / (2 * (k**2 + q**2 - k * q * z) * (1 + uo) ** 2 * (q**2 + 2 * k * (k - q * z) + q**2 * uo))
    )
    return ScalarTypeIIcAfterIntOverFreq


def ScalarTypeIIdAfterIntOverFreq():
    """
    Diagram: e12|23|3|e|:0B_bB_vv|vv_bB|Vv|0b|
    """
    ScalarTypeIIdAfterIntOverFreq = (
        pi**2
        * k ** (-d - 2 * eps)
        * q ** (-d - 2 * eps)
        * go**2
        * nuo
        * q**2
        / (2 * (k**2 + q**2 + k * q * z) * (1 + uo) * (k**2 + q**2 + (k**2 + q**2 + 2 * k * q * z) * uo))
    )
    return ScalarTypeIIdAfterIntOverFreq


def compare_UV_divergent_parts(nickel_index: str, obtained_UV_divergent_part: Any):
    match nickel_index:
        case "e12|e3|33||:0B_bB_vV|0b_vv|vV_vv||":
            test_UV_divergent_part = ScalarTypeIaAfterIntOverFreq()
        case "e12|e3|33||:0B_vv_bB|0b_Bb|bB_vv||":
            test_UV_divergent_part = ScalarTypeIbAfterIntOverFreq()
        case "e12|e3|33||:0B_bB_vv|0b_vV|Vv_vv||":
            test_UV_divergent_part = ScalarTypeIcAfterIntOverFreq()
        case "e12|e3|33||:0B_bB_vV|0b_vV|vv_vv||":
            test_UV_divergent_part = ScalarTypeIdAfterIntOverFreq()
        case "e12|23|3|e|:0B_bB_vV|vv_bB|vv|0b|":
            test_UV_divergent_part = ScalarTypeIIaAfterIntOverFreq()
        case "e12|23|3|e|:0B_bB_vv|bB_vv|bB|0b|":
            test_UV_divergent_part = ScalarTypeIIbAfterIntOverFreq()
        case "e12|23|3|e|:0B_bB_vv|vV_bB|vv|0b|":
            test_UV_divergent_part = ScalarTypeIIcAfterIntOverFreq()
        case "e12|23|3|e|:0B_bB_vv|vv_bB|Vv|0b|":
            test_UV_divergent_part = ScalarTypeIIdAfterIntOverFreq()
        case _:
            return sys.exit("Unknown diagram contains UV divergent part")

    # it is better to choose the test value of the uo parameter as natural,
    # so that there are no problems with accuracy
    obtained_integrand_for_test = simplify(
        obtained_UV_divergent_part.doit().doit().doit().subs(d, 3).subs(eps, 0).subs(A, 1).subs(b, 0).subs(uo, 2).doit()
    )
    test_UV_divergent_part = simplify(test_UV_divergent_part.subs(d, 3).subs(eps, 0).subs(uo, 2))

    distinction = obtained_integrand_for_test - test_UV_divergent_part

    result = simplify(distinction)

    if result == 0:
        return True
    else:
        return False
