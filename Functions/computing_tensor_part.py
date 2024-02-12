import time
import sys
import sympy as sym

from sympy import *
from typing import Any

from Functions.Data_classes import *
from Functions.SymPy_classes import *


# ------------------------------------------------------------------------------------------------------------------#
#                             Calculation of the tensor part of the integrand for a diagram
# ------------------------------------------------------------------------------------------------------------------#


def tensor_structure_term_classification(expr: sym.core.add.Mul, list_for_expr_args: list):
    """
    This function compares expr (some kind of one multiplier) with a template
    and adds its arguments to the list.
    """
    match expr:
        case mom():
            return list_for_expr_args.append(["mom", list(expr.args)])
        case P():
            return list_for_expr_args.append(["P", list(expr.args)])
        case H():
            return list_for_expr_args.append(["H", list(expr.args)])
        case kd():
            return list_for_expr_args.append(["kd", list(expr.args)])
        # symbols or numbers
        case _:
            return list_for_expr_args.append([expr])


def get_tensor_structure_arguments_structure(expr: Any):
    """
    This function returns the expr structure as [expr_term_1, expr_term_2, ...],
    expr_term_1 = [term_multiplier_1, term_multiplier_2, ...], where, for example,
    term_multiplier_1 = ['P', [k, 2, 4]].

    ARGUMENTS:

    expr -- expression containing functions mom(), P(), H(), kd(), symbols rho, A, I and numbers

    OUTPUT DATA EXAMPLE:

    expr = mom(-q, 9)*kd(11, 10)*P(k, 2, 6) + A*mom(-q, 10)*kd(11, 9) + kd(12, 9)

    get_tensor_structure_arguments_structure(expr) =
    [
    [-1, A, ['mom', [q, 10]], ['kd', [11, 9]]], [-1, ['P', [k, 2, 6]], ['mom', [q, 9]], ['kd', [11, 10]]],
    [['kd', [12, 9]]]
    ]

    """

    # a priori, open all brackets
    expr_new = expand(expr)

    list_for_term_multipliers = list()

    number_of_terms_in_expr = len(expr_new.args)

    for i in range(number_of_terms_in_expr):
        term = expr_new.args[i]

        list_for_multiplier_args = list()
        # adds its arguments to the list
        if term.is_Mul:
            number_of_multipliers_in_term = len(term.args)

            for j in range(number_of_multipliers_in_term):
                multiplier = term.args[j]
                tensor_structure_term_classification(multiplier, list_for_multiplier_args)
        # term in expr_new has one element
        else:
            tensor_structure_term_classification(term, list_for_multiplier_args)

        list_for_term_multipliers.append(list_for_multiplier_args)

    return list_for_term_multipliers


def extract_B_and_nuo_depend_factor_from_tensor_part(tensor_convolution: Any):
    """
    The function divides the tensor structure into parts that depend on and
    do not depend on the integration variables.

    ARGUMENTS:

    tensor_convolution is given by computing_tensor_structures()

    OUTPUT DATA EXAMPLE:

    too long
    """

    tensor_part_numerator = fraction(tensor_convolution)[0]
    tensor_part_denominator = fraction(tensor_convolution)[1]

    tensor_part_numerator_args = tensor_part_numerator.args

    field_and_nuo_depend_factor_in_tonsor_part = 1
    dimless_tensor_part_numerator = 1

    for i in range(len(tensor_part_numerator_args)):
        numerator_term = tensor_part_numerator_args[i]
        if numerator_term.has(I) or numerator_term.has(rho) or numerator_term.has(mom) or numerator_term.has(lcs):
            field_and_nuo_depend_factor_in_tonsor_part *= numerator_term
        else:
            dimless_tensor_part_numerator *= numerator_term

    scale_parameter = B / nuo

    dimless_tensor_part_numerator_after_subs = dimless_tensor_part_numerator.subs(k, k * scale_parameter).subs(
        q, q * scale_parameter
    )
    tensor_part_denominator_after_subs = tensor_part_denominator.subs(k, k * scale_parameter).subs(
        q, q * scale_parameter
    )

    # by construction, the numerator and denominator of the tensor structure
    # are homogeneous polynomials in momentums
    if (
        Poly(dimless_tensor_part_numerator_after_subs, k, q).is_homogeneous
        and Poly(dimless_tensor_part_numerator_after_subs, k, q).is_homogeneous
    ):
        numerator_order = Poly(dimless_tensor_part_numerator_after_subs, k, q).homogeneous_order()
        denomenator_order = Poly(tensor_part_denominator_after_subs, k, q).homogeneous_order()
        field_and_nuo_depend_factor_in_tonsor_part *= scale_parameter ** (numerator_order - denomenator_order)
    else:
        sys.exit("Tensor structure is not a homogeneous function")

    tensor_structure_separated_into_two_parts = IntegrandPrefactorAfterSubstitution(
        dimless_tensor_part_numerator / tensor_part_denominator, field_and_nuo_depend_factor_in_tonsor_part
    )

    return tensor_structure_separated_into_two_parts


def external_index(tensor_structure: list, ext_index: int, index_list: int, position: int):
    """
    The function put the index of external field on the position in list, if the index of external
    field is in the index_list. On the other hand, the result is sn empty list.

    ARGUMENTS:

    tensor_structure -- combination of index and momenta: [i, j, "q", l, "k", j, "p", -1, "k", -1, "q", -1]
    ext_index  -- indexb or indexB
    index_list -- list of index for momenta k or q
    position   -- position on which the index is change to the ex_index
    """
    if ext_index in tensor_structure:
        return tensor_structure
    elif ext_index not in index_list:
        return list()
    elif tensor_structure[position] != -1:
        return list()
    elif len(tensor_structure) - 1 == position:
        return tensor_structure[:position] + [ext_index]
    else:
        return tensor_structure[:position] + [ext_index] + tensor_structure[position + 1 :]


def kronecker_transver_operator(tensor: Any, transver: list, kronecker: list):
    """
    This function performs the following tensor convolution with
    Kronecker delta function: P(k, i, j) * kd(i, l) = P(k, l, j).

    ARGUMENTS:

    tensor    -- Tenzor - projector, kronecker symbol and momenta
    transver  -- P_structure - all possible transverse structure in Tensor
    kronecker -- kd_structure - all possible kronecker structure in Tensor
    """
    for kronecker_indices in kronecker:
        updated_structure = []
        kd_index1, kd_index2 = kronecker_indices

        for transver_args in transver:
            P_arg, P_index1, P_index2 = transver_args

            substitution_term = P(P_arg, P_index1, P_index2) * kd(kd_index1, kd_index2)
            substitution_result = None

            if P_index1 == kd_index1:
                substitution_result = P(P_arg, kd_index2, P_index2)
                updated_structure.append([P_arg, kd_index2, P_index2])
            elif P_index1 == kd_index2:
                substitution_result = P(P_arg, kd_index1, P_index2)
                updated_structure.append([P_arg, kd_index1, P_index2])
            elif P_index2 == kd_index1:
                substitution_result = P(P_arg, P_index1, kd_index2)
                updated_structure.append([P_arg, P_index1, kd_index2])
            elif P_index2 == kd_index2:
                substitution_result = P(P_arg, P_index1, kd_index1)
                updated_structure.append([P_arg, P_index1, kd_index1])

            if substitution_result is not None:
                tensor = tensor.subs(substitution_term, substitution_result)

            if tensor.coeff(kd(kd_index1, kd_index2)) == 0:
                break

        transver.extend(updated_structure)

    return [tensor, transver]


def kronecker_helical_operator(tensor: Any, helical: list, kronecker: list):
    """
    This function performs the following tensor convolution with
    Kronecker delta function: H(k, i, j) * kd(i, l) = H(k, l, j).

    ARGUMENTS:

    tensor    -- Tenzor - projector, kronecker symbol and momenta
    helical   -- H_structure - all possible transverse structure in Tensor
    kronecker -- kd_structure - all possible kronecker structure in Tensor
    """
    for kronecker_indices in kronecker:
        updated_structure = []

        kd_index1, kd_index2 = kronecker_indices

        for helical_args in helical:
            H_arg, H_index1, H_index2 = helical_args

            substitution_term = H(H_arg, H_index1, H_index2) * kd(kd_index1, kd_index2)
            substitution_result = None

            if H_index1 == kd_index1:
                substitution_result = H(H_arg, kd_index2, H_index2)
                updated_structure.append([H_arg, kd_index2, H_index2])
            elif H_index1 == kd_index2:
                substitution_result = H(H_arg, kd_index1, H_index2)
                updated_structure.append([H_arg, kd_index1, H_index2])
            elif H_index2 == kd_index1:
                substitution_result = H(H_arg, H_index1, kd_index2)
                updated_structure.append([H_arg, H_index1, kd_index2])
            elif H_index2 == kd_index2:
                substitution_result = H(H_arg, H_index1, kd_index1)
                updated_structure.append([H_arg, H_index1, kd_index1])

            if substitution_result is not None:
                tensor = tensor.subs(substitution_term, substitution_result)

            if tensor.coeff(kd(kd_index1, kd_index2)) == 0:
                break

        helical.extend(updated_structure)

    return [tensor, helical]


def momenta_transver_operator(tensor: Any, transver: list):
    """
    This function performs the following tensor identity:
    P(k, i, j) * mom(k, i) = 0.

    ARBUMENTS:

    tensor   -- Tenzor - projector, kronecker symbol and momenta
    transver -- P_structure - all possible transverse structure in Tensor
    """
    P_position_in_transver = 0

    while P_position_in_transver < len(transver):
        P_args = transver[P_position_in_transver]
        P_arg, P_index1, P_index2 = P_args

        if tensor.coeff(mom(P_arg, P_index1)) != 0:
            tensor = tensor.subs(P(P_arg, P_index1, P_index2) * mom(P_arg, P_index1), 0)
        if tensor.coeff(mom(P_arg, P_index2)) != 0:
            tensor = tensor.subs(P(P_arg, P_index1, P_index2) * mom(P_arg, P_index2), 0)
        if tensor.coeff(P(P_arg, P_index1, P_index2)) == 0:
            transver.remove(P_args)
        else:
            P_position_in_transver += 1

    return [tensor, transver]


def momenta_helical_operator(tensor: Any, helical: list):
    """
    This function performs the following tensor identity:
    H(k, i, j) * mom(k, i) = 0. It follows from the antisymmetry
    of the tensor operator H(k, i, j).

    ARBUMENTS:

    tensor  -- Tenzor - projector, kronecker symbol and momenta
    helical -- H_structure - all possible helical structure in Tensor
    """
    H_position_in_helical = 0

    while H_position_in_helical < len(helical):
        H_args = helical[H_position_in_helical]
        H_arg, H_index1, H_index2 = H_args

        if tensor.coeff(mom(H_arg, H_index1)) != 0:
            tensor = tensor.subs(H(H_arg, H_index1, H_index2) * mom(H_arg, H_index1), 0)
        if tensor.coeff(mom(H_arg, H_index2)) != 0:
            tensor = tensor.subs(H(H_arg, H_index1, H_index2) * mom(H_arg, H_index2), 0)
        if tensor.coeff(H(H_arg, H_index1, H_index2)) == 0:
            helical.remove(H_args)
        else:
            H_position_in_helical += 1

    return [tensor, helical]


def transfer_helical_operator(tensor: Any, transver: list, helical: list):
    """
    This function performs the following tensor convolution:
    P(k, i, j) * H(k, i, l) = H (k, l, j).

    ARGUMENTS:

    tensor   -- Tenzor - projector, kronecker symbol and momenta
    helical  -- H_structure - all possible helical structure in Tensor
    transver -- P_structure - all possible transver structure in Tensor
    """
    H_position_in_helical = 0

    while H_position_in_helical < len(helical):
        H_args = helical[H_position_in_helical]
        H_arg, H_index1, H_index2 = H_args

        for transver_args in transver:
            P_arg, P_index1, P_index2 = transver_args

            substitution_term = H(H_arg, H_index1, H_index2) * P(P_arg, P_index1, P_index2)
            substitution_result = None

            if H_arg == P_arg and tensor.coeff(substitution_term) != 0:
                if H_index1 == P_index1:
                    substitution_result = H(H_arg, P_index2, H_index2)
                    helical.append([H_arg, P_index2, H_index2])
                elif H_index1 == P_index2:
                    substitution_result = H(H_arg, P_index1, H_index2)
                    helical.append([H_arg, P_index1, H_index2])
                elif H_index2 == P_index1:
                    substitution_result = H(H_arg, H_index1, P_index2)
                    helical.append([H_arg, H_index1, P_index2])
                elif H_index2 == P_index2:
                    substitution_result = H(H_arg, H_index1, P_index1)
                    helical.append([H_arg, H_index1, P_index1])

                if substitution_result is not None:
                    tensor = tensor.subs(substitution_term, substitution_result)

        if tensor.coeff(H(H_arg, H_index1, H_index2)) == 0:
            helical.remove(H_args)
        else:
            H_position_in_helical += 1

    return [tensor, helical]


def transver_transver_operator(tensor: Any, transver: list):
    """
    This function performs the following tensor convolution:
    P(k, i, j) * P(k, i, l) = P (k, l, j).

    ARGUMENTS:

    tensor   -- Tenzor - projector, kronecker symbol and momenta
    transver -- P_structure - all possible transver structure in Tensor
    """
    P_position_in_transver = 0

    while P_position_in_transver < len(transver):
        updated_structure = []

        P1_args = transver[P_position_in_transver]
        P1_arg, P1_index1, P1_index2 = P1_args

        for j in range(P_position_in_transver + 1, len(transver)):
            P2_args = transver[j]
            P2_arg, P2_index1, P2_index2 = P2_args

            substitution_term = P(P1_arg, P1_index1, P1_index2) * P(P2_arg, P2_index1, P2_index2)
            substitution_result = None

            if P1_arg == P2_arg and tensor.coeff(substitution_term) != 0:
                if P1_index1 == P2_index1:
                    substitution_result = P(P1_arg, P2_index2, P1_index2)
                    updated_structure.append([P1_arg, P2_index2, P1_index2])
                elif P1_index1 == P2_index2:
                    substitution_result = P(P1_arg, P2_index1, P1_index2)
                    updated_structure.append([P1_arg, P2_index1, P1_index2])
                elif P1_index2 == P2_index1:
                    substitution_result = P(P1_arg, P1_index1, P2_index2)
                    updated_structure.append([P1_arg, P1_index1, P2_index2])
                elif P1_index2 == P2_index2:
                    substitution_result = P(P1_arg, P1_index1, P2_index1)
                    updated_structure.append([P1_arg, P1_index1, P2_index1])

                if substitution_result is not None:
                    tensor = tensor.subs(substitution_term, substitution_result)

        if tensor.coeff(P(P1_arg, P1_index1, P1_index2)) == 0:
            transver.remove(P1_args)
        else:
            P_position_in_transver += 1

        transver.extend(updated_structure)

    return [tensor, transver]


def kronecker_momenta(tensor: Any, kronecker: list):
    """
    This function performs the following tensor convolution with
    Kronecker delta function: mom( k, i) * kd(i, j) = mom(k, j).

    ARGUMENTS:

    tensor    -- Tenzor - projector, kronecker symbol and momenta
    kronecker -- kd_structure - all possible transver structure in Tensor
    """

    i = 0
    while i == 0:
        for kronecker_indices in kronecker:
            kd_index1, kd_index2 = kronecker_indices
            kd_part = tensor.coeff(kd(kd_index1, kd_index2))

            if kd_part != 0:
                if kd_part.coeff(mom(k, kd_index1)) != 0:
                    tensor = tensor.subs(kd(kd_index1, kd_index2) * mom(k, kd_index1), mom(k, kd_index2))
                    i += 1
                if kd_part.coeff(mom(k, kd_index2)) != 0:
                    tensor = tensor.subs(kd(kd_index1, kd_index2) * mom(k, kd_index2), mom(k, kd_index1))
                    i += 1
                if kd_part.coeff(mom(q, kd_index1)) != 0:
                    tensor = tensor.subs(kd(kd_index1, kd_index2) * mom(q, kd_index1), mom(q, kd_index2))
                    i += 1
                if kd_part.coeff(mom(q, kd_index2)) != 0:
                    tensor = tensor.subs(kd(kd_index1, kd_index2) * mom(q, kd_index2), mom(q, kd_index1))
                    i += 1
                if kd_part.coeff(mom(p, kd_index1)) != 0:
                    tensor = tensor.subs(kd(kd_index1, kd_index2) * mom(p, kd_index1), mom(p, kd_index2))
                    i += 1
                if kd_part.coeff(mom(p, kd_index2)) != 0:
                    tensor = tensor.subs(kd(kd_index1, kd_index2) * mom(p, kd_index2), mom(p, kd_index1))
                    i += 1
            else:
                kronecker.remove(kronecker_indices)
                i = 1
                break
        if i != 0:
            i = 0
        else:
            break

    return [tensor, kronecker]


def momenta_momenta_helical_operator(tensor: Any, helical: list):
    """
    This function performs the following tensor identity:
    H( k, i, j) * mom( q, i) * mom( q, j) = 0. It follows
    from the antisymmetry of the tensor operator H(k, i, j).

    ARGUMENTS:

    tensor  -- Tenzor - projector, kronecker symbol and momenta
    helical -- H_structure - all possible transverse structure in Tensor
    """

    H_position_in_helical = 0
    # calculation for helical term
    while H_position_in_helical < len(helical):
        H_args = helical[H_position_in_helical]
        H_arg, H_index1, H_index2 = H_args

        H_part = tensor.coeff(H(H_arg, H_index1, H_index2))

        if H_arg == k and H_part.coeff(mom(q, H_index1) * mom(q, H_index2)) != 0:
            tensor = tensor.subs(H(k, H_index1, H_index2) * mom(q, H_index1) * mom(q, H_index2), 0)
        if H_arg == q and H_part.coeff(mom(k, H_index1) * mom(k, H_index2)) != 0:
            tensor = tensor.subs(H(q, H_index1, H_index2) * mom(k, H_index1) * mom(k, H_index2), 0)

        H_position_in_helical += 1

    return [tensor, helical]


def four_indices_external_fields(helical: list, indexb: int, indexB: int, k_indices: list, q_indices: list):
    """
    The function looking for the index structure in the helical part with four internal momenta. The result is the list with momenta and indices
    For Example: H(k, i_b, j) mom(q, j) mom(k, indexB) -> [i_b, j, "k", l, "q", j, "p", i_p, "k", i_B, "q", i_]
    H(q, i_b, j) mom(k, j) mom(k, indexB) ... -> [i_b, j, l, "q", l, "k", j, "p", -2, "k", i_B, "q", -1]

    ARGUMENTS:
    helical   - is helical term H(k, i, j) = epsilon(i,j,l) k_l /k
    indexb    - index correspond to the external field b
    indexB    - index correspond to the external field b'
    k_indices - k_structure - list of possible indices for momentum k
    q_indices - q_structure - list of possible indices for momentum q
    """

    if helical[0] == k:
        structure = [helical[1], helical[2], s, k, s, q, -1, p, -2, k, -1, q, -1]
    else:
        structure = [helical[1], helical[2], s, q, s, k, -1, p, -2, k, -1, q, -1]

    if indexB == structure[0]:
        structure[6] = structure[1]
    elif indexB == structure[1]:
        structure[6] = structure[0]
    elif indexb == structure[0]:
        structure[6] = structure[1]
    elif indexb == structure[1]:
        structure[6] = structure[0]

    if structure[3] == q:
        structure = structure[0:3] + structure[5:7] + structure[3:5] + structure[7:13]

    structure_old = [external_index(structure, indexB, k_indices, 10)]
    structure_new = [external_index(structure, indexB, q_indices, 12)]
    if structure_new not in structure_old:
        structure_old += structure_new

    if list() in structure_old:
        structure_old.remove(list())

    structure_new = list()
    for i in structure_old:
        structure_new.append(external_index(i, indexb, k_indices, 10))
        structure = external_index(i, indexb, q_indices, 12)
        if structure not in structure_new:
            structure_new.append(structure)
        if list() in structure_new:
            structure_new.remove(list())

    return structure_new


def four_indices_external_momentum(structure, p_indices, k_indices, q_indices):
    """
    The function give the specified indecies structure for index of external momentum among four momenta and helical term in a tensor structure.
    For Example: H(k, i_b, j) mom(q, j) mom(k, indexB)  mom(p, i) mom(q, i) -> [i_b, j, "k", l, "q", j, "p", i_p, "k", i_B, "q", i_p]

    ARGUMENTS:
    structure -- the structure what it is needed to be replace with
    p_indices -- list all indices for external momentum
    k_indices -- list all indices for k momentum
    q_indices -- list all indices for q momentum
    """

    i = structure.index(-1)
    result = list()

    match i:
        case 4:
            if (structure[0] in p_indices) and (structure[1] in k_indices):
                result += [structure[0:4] + [structure[1]] + structure[5:8] + [structure[0]] + structure[9:13]]
            if (structure[1] in p_indices) and (structure[0] in k_indices):
                result += [structure[0:4] + [structure[0]] + structure[5:8] + [structure[1]] + structure[9:13]]
        case 6:
            if p_indices.count(structure[0]) == 1 and q_indices.count(structure[1]) == 1:
                result += [structure[0:6] + [structure[1]] + structure[7:8] + [structure[0]] + structure[9:13]]
            if p_indices.count(structure[1]) == 1 and q_indices.count(structure[0]) == 1:
                result += [structure[0:6] + [structure[0]] + structure[7:8] + [structure[1]] + structure[9:13]]
        case 10:
            indices = list(set(p_indices).intersection(k_indices))
            for j in indices:
                result += [structure[0:8] + [j] + structure[9:10] + [j] + structure[11:13]]
        case 12:
            indices = list(set(p_indices).intersection(q_indices))
            for j in indices:
                result += [structure[0:8] + [j] + structure[9:12] + [j]]

    return result


def scalar_result(momentum: Any, part: str):
    """
    The function replace the momentum the equivalent form proportional Lambda or B field.

    ARGUMENTS:

    momentum -- the momentum structure
    part     -- keyword (lambda or Bfield)
    """
    kq_dot_product = k * q * z
    kB_dot_product = B * k * z_k
    qB_dot_product = B * q * z_q

    if momentum == k**2 * q**2:
        if part == "lambda":
            return q**2 * k**2 * (1 - z**2) / d / (d - 1)
        if part == "Bfield":
            term1 = 2 * (kB_dot_product) * (qB_dot_product) * (kq_dot_product)
            term2 = (qB_dot_product) ** 2 * k**2
            term3 = (kB_dot_product) ** 2 * q**2
            term4 = B**2 * (k**2 * q**2 - (kq_dot_product) ** 2)
            return (term1 - term2 - term3 + term4) / (B**2 * (d - 2) * (d - 1))
    elif momentum == k**2:
        if part == "lambda":
            return k**2 / d
        if part == "Bfield":
            return (k**2 - (kB_dot_product) ** 2 / B**2) / (d - 1)
    elif momentum == q**2:
        if part == "lambda":
            return q**2 / d
        if part == "Bfield":
            return (q**2 - (qB_dot_product) ** 2 / B**2) / (d - 1)
    elif momentum == k * q:
        if part == "lambda":
            return kq_dot_product / d
        if part == "Bfield":
            return (kq_dot_product - (kB_dot_product) * (qB_dot_product) / B**2) / (d - 1)


def H_structure_calculation_part_1(
    Tenzor: Any,
    H_structure: Any,
    indexB: Any,
    indexb: Any,
    p_structure: list,
    k_structure: list,
    q_structure: list,
    keyword: str,
):
    """
    calculation of H structure - For this particular case, one of the external indices (p_s, b_i or B_j) is paired with a helicity term.
    we will therefore use the information that, in addition to the helicity term H( , i,j), they can be multiplied by a maximum of three internal momenta.
    For examle: H(k, indexb, j) mom(q, j) mom(k, indexB) mom(q, i) mom(p, i) and thus in this step I will calculate all possible combinations for this structure.
    In this case, helical term H(k, i, j) = epsilon(i,j,s) k_s /k

    ARGUMENTS:

    OUTPUT DATA EXAMPLE:

    """

    y = 0
    while y < len(H_structure):
        combination = four_indices_external_fields(H_structure[y], indexb, indexB, k_structure, q_structure)
        combination_new = list()
        for x in combination:
            combination_new += four_indices_external_momentum(x, p_structure, k_structure, q_structure)

        for x in combination_new:
            if x[8] == x[12]:
                if x[1] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * mom(q, x[6]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        -mom(p, s) * lcs(s, x[0], x[10]) * scalar_result(k**2 * q**2, keyword) / k,
                    )
                if x[1] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * mom(k, x[4]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        mom(p, s) * lcs(s, x[0], x[10]) * scalar_result(k**2 * q**2, keyword) / q,
                    )
                if x[0] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * mom(q, x[6]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        mom(p, s) * lcs(s, x[1], x[10]) * scalar_result(k**2 * q**2, keyword) / k,
                    )
                if x[0] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * mom(k, x[4]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        -mom(p, s) * lcs(s, x[1], x[10]) * scalar_result(k**2 * q**2, keyword) / q,
                    )
            if x[8] == x[10]:
                if x[1] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * mom(q, x[6]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        mom(p, s) * lcs(s, x[0], x[12]) * scalar_result(k**2 * q**2, keyword) / k,
                    )
                if x[1] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * mom(k, x[4]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        -mom(p, s) * lcs(s, x[0], x[12]) * scalar_result(k**2 * q**2, keyword) / q,
                    )
                if x[0] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * mom(q, x[6]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        -mom(p, s) * lcs(s, x[1], x[12]) * scalar_result(k**2 * q**2, keyword) / k,
                    )
                if x[0] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * mom(k, x[4]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        mom(p, s) * lcs(s, x[1], x[12]) * scalar_result(k**2 * q**2, keyword) / q,
                    )
            if x[8] == x[0]:
                if x[1] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * mom(q, x[6]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        -mom(p, s) * lcs(s, x[10], x[12]) * scalar_result(k**2 * q**2, keyword) / k,
                    )
                if x[1] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * mom(k, x[4]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        mom(p, s) * lcs(s, x[10], x[12]) * scalar_result(k**2 * q**2, keyword) / q,
                    )
            if x[8] == x[1]:
                if x[0] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * mom(q, x[6]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        mom(p, s) * lcs(s, x[10], x[12]) * scalar_result(k**2 * q**2, keyword) / k,
                    )
                if x[0] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * mom(k, x[4]) * mom(p, x[8]) * mom(k, x[10]) * mom(q, x[12]),
                        -mom(p, s) * lcs(s, x[10], x[12]) * scalar_result(k**2 * q**2, keyword) / q,
                    )
        y += 1

    return Tenzor


def H_structure_calculation_part_2(
    Tenzor: Any,
    H_structure: Any,
    indexB: Any,
    indexb: Any,
    p_structure: list,
    keyword: str,
):
    """
    calculate the structure where there are two external momentums: H(momentum, i, indexB)* p(i) mom( , indexb)
    and other combinations except H(momentum, indexB, indexb) mom(p, i) mom(k, i)

    ARGUMENTS:

    OUTPUT DATA EXAMPLE:

    """

    for x in H_structure:
        if Tenzor.coeff(H(x[0], x[1], x[2])) != 0:
            if x[1] in p_structure and x[2] == indexb:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, x[1], indexb) * mom(p, x[1]) * mom(k, indexB),
                        mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k**2, keyword) / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, x[1], indexb) * mom(p, x[1]) * mom(q, indexB),
                        mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k * q, keyword) / k,
                    )
                if x[0] == q:
                    Tenzor = Tenzor.subs(
                        H(q, x[1], indexb) * mom(p, x[1]) * mom(q, indexB),
                        mom(p, s) * lcs(s, indexb, indexB) * scalar_result(q**2, keyword) / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, x[1], indexb) * mom(p, x[1]) * mom(k, indexB),
                        mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k * q, keyword) / q,
                    )
            if x[2] in p_structure and x[1] == indexb:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, indexb, x[2]) * mom(p, x[2]) * mom(k, indexB),
                        -mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k**2, keyword) / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, indexb, x[2]) * mom(p, x[2]) * mom(q, indexB),
                        -mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k * q, keyword) / k,
                    )
                if x[0] == q:
                    Tenzor = Tenzor.subs(
                        H(q, indexb, x[2]) * mom(p, x[2]) * mom(q, indexB),
                        -mom(p, s) * lcs(s, indexb, indexB) * scalar_result(q**2, keyword) / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, indexb, x[2]) * mom(p, x[2]) * mom(k, indexB),
                        -mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k * q, keyword) / q,
                    )
            if x[1] in p_structure and x[2] == indexB:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, x[1], indexB) * mom(p, x[1]) * mom(k, indexb),
                        -mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k**2, keyword) / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, x[1], indexB) * mom(p, x[1]) * mom(q, indexb),
                        -mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k * q, keyword) / k,
                    )
                if x[0] == q:
                    Tenzor = Tenzor.subs(
                        H(q, x[1], indexB) * mom(p, x[1]) * mom(q, indexb),
                        -mom(p, s) * lcs(s, indexb, indexB) * scalar_result(q**2, keyword) / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, x[1], indexB) * mom(p, x[1]) * mom(k, indexb),
                        -mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k * q, keyword) / q,
                    )
            if x[2] in p_structure and x[1] == indexB:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, indexB, x[2]) * mom(p, x[2]) * mom(k, indexb),
                        mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k**2, keyword) / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, indexB, x[2]) * mom(p, x[2]) * mom(q, indexb),
                        mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k * q, keyword) / k,
                    )
                else:
                    Tenzor = Tenzor.subs(
                        H(q, indexB, x[2]) * mom(p, x[2]) * mom(q, indexb),
                        mom(p, s) * lcs(s, indexb, indexB) * scalar_result(q**2, keyword) / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, indexB, x[2]) * mom(p, x[2]) * mom(k, indexb),
                        mom(p, s) * lcs(s, indexb, indexB) * scalar_result(k * q, keyword) / q,
                    )
        if Tenzor.coeff(H(x[0], x[1], x[2])) != 0:
            for y in p_structure:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, x[1], x[2]) * mom(p, y) * mom(k, y),
                        mom(p, s) * lcs(s, x[1], x[2]) * scalar_result(k**2, keyword) / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, x[1], x[2]) * mom(p, y) * mom(q, y),
                        mom(p, s) * lcs(s, x[1], x[2]) * scalar_result(k * q, keyword) / k,
                    )
                if x[0] == q:
                    Tenzor = Tenzor.subs(
                        H(q, x[1], x[2]) * mom(p, y) * mom(q, y),
                        mom(p, s) * lcs(s, x[1], x[2]) * scalar_result(q**2, keyword) / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, x[1], x[2]) * mom(p, y) * mom(k, y),
                        mom(p, s) * lcs(s, x[1], x[2]) * scalar_result(k * q, keyword) / q,
                    )

    return Tenzor


def process_tensor(Tenzor, index1, index2):
    kd_index = kd(index1, index2)

    if Tenzor.has(kd_index):
        coeff_value = Tenzor.coeff(kd_index)

        if coeff_value == 0:
            Tenzor = simplify(Tenzor.subs(kd_index, 0))

            return Tenzor
        else:
            sys.exit("The asymptotics includes a nonzero tensor contribution linear in p that cannot exist")
    else:
        return Tenzor


def computing_tensor_structures(diagram_data: DiagramData, UV_convergence_criterion: bool, quick_diagrams: str):
    """
    The function replace the Kronecker's delta function and transverse projector by the transverse projector.
    For example: P(k, i, j) * kd(i, l) = P(k, l, j)

    ARGUMENTS:

    diagram_data is given by get_info_about_diagram()

    OUTPUT DATA EXAMPLE:

    """
    moznost = diagram_data.momentums_at_vertices
    indexb = diagram_data.indexb
    indexB = diagram_data.indexB
    P_structure = diagram_data.P_structure
    H_structure = diagram_data.H_structure
    kd_structure = diagram_data.kd_structure
    Tenzor = diagram_data.integrand_tensor_part

    t = time.time()
    # print(f"{Tenzor}")
    Tenzor = expand(Tenzor)  # The final tesor structure from the diagram

    # print(f"{kd_structure}")
    # print(f"{P_structure}")
    # print(f"{H_structure}")
    # exit()

    # What I need from the Tenzor structure
    Tenzor = rho * Tenzor.coeff(rho**rho_degree)
    # print(Tenzor)
    Tenzor = expand(Tenzor.subs(I**5, I))  # calculate the imaginary unit
    # Tenzor = Tenzor.subs(A, 1)              # It depends on which part we want to calculate from the vertex Bbv
    # print(Tenzor)
    # We are interested in the leading (proportional to p) contribution to the diagram asymptotic, when p --> 0.
    print(
        f"Selecting the part proportional to rho "
        f"(corresponds to the asymptotic of the diagram as p --> 0): \n{round(time.time() - t, 1)} sec"
    )

    [Tenzor, P_structure] = kronecker_transver_operator(Tenzor, P_structure, kd_structure)
    [Tenzor, H_structure] = kronecker_helical_operator(Tenzor, H_structure, kd_structure)

    print(
        f"Computing tensor convolutions of the form "
        f"P(k, i, j)*kd(i, l) = P(k, l, j) and H(k, i, j)*kd(i, l) = H(k, l, j): \n{round(time.time() - t, 1)} sec"
    )

    [Tenzor, P_structure] = momenta_transver_operator(Tenzor, P_structure)
    [Tenzor, H_structure] = momenta_helical_operator(Tenzor, H_structure)

    print(
        f"Computing tensor convolutions of the form "
        f"P(k, i, j)*mom(k, i) = 0 and H(k, i, j)*mom(k, i) = 0: \n{round(time.time() - t, 1)} sec"
    )

    [Tenzor, H_structure] = transfer_helical_operator(Tenzor, P_structure, H_structure)

    print(
        f"Computing tensor convolutions of the form "
        f"P(k, i, j)*H(k, i, l) = H(k, l, j): \n{round(time.time() - t, 1)} sec"
    )

    [Tenzor, P_structure] = transver_transver_operator(Tenzor, P_structure)

    print(
        f"Computing tensor convolutions of the form "
        f"P(k, i, j)*P(k, i, l) = P(k, l, j): \n{round(time.time() - t, 1)} sec"
    )

    Tenzor = expand(Tenzor)
    [Tenzor, P_structure] = momenta_transver_operator(Tenzor, P_structure)
    [Tenzor, H_structure] = momenta_helical_operator(Tenzor, H_structure)
    [Tenzor, H_structure] = momenta_momenta_helical_operator(Tenzor, H_structure)

    kd_structure = list()
    for i in P_structure:  # Define transverse projection operator P(k,i,j) = kd(i,j) - mom(k,i)*mom(k,j)/k^2
        k_c = i[0].coeff(k)
        q_c = i[0].coeff(q)
        Tenzor = Tenzor.subs(
            P(i[0], i[1], i[2]),
            kd(i[1], i[2])
            - (k_c * mom(k, i[1]) + q_c * mom(q, i[1]))
            * (k_c * mom(k, i[2]) + q_c * mom(q, i[2]))
            / (k_c**2 * k**2 + q_c**2 * q**2 + 2 * k_c * q_c * k * q * z),
        )
        kd_structure.append([i[1], i[2]])

    print(
        f"Definition of transversal projection operators "
        f"P(k, i, j) = kd(i, j) - mom(k, i) mom(k, j)/||k||**2: \n{round(time.time() - t, 1)} sec"
    )

    Tenzor = expand(Tenzor)

    [Tenzor, H_structure] = momenta_helical_operator(Tenzor, H_structure)
    [Tenzor, H_structure] = momenta_momenta_helical_operator(Tenzor, H_structure)

    # experience shows that if this step takes more than 30-40 seconds,
    # then the remaining calculation time increases sharply
    if quick_diagrams == "y":
        if round(time.time() - t) > 40:
            Tensor_data = IntegrandTensorStructure(None, None)
            return Tensor_data

    print(
        f"Computing tensor convolutions of the form "
        f"H(k, i, j)*mom(q, i)*mom(q, j) = 0 and H(k, i, j)*mom(k, i) = 0: \n{round(time.time() - t, 1)} sec"
    )

    [Tenzor, kd_structure] = kronecker_momenta(Tenzor, kd_structure)

    print(
        f"Computing tensor convolutions of the form "
        f"mom(k, i)*kd(i, j) = mom(k, j): \n{round(time.time() - t, 1)} sec"
    )

    # delete zero values in the Tenzor: H( ,i,j) mom(p, i) mom( ,j) mom(k, indexB) mom(k, indexb) = 0
    Tenzor = Tenzor.subs(mom(q, indexb) * mom(q, indexB), 0)
    Tenzor = Tenzor.subs(mom(k, indexb) * mom(k, indexB), 0)

    [Tenzor, H_structure] = momenta_helical_operator(Tenzor, H_structure)
    [Tenzor, H_structure] = momenta_momenta_helical_operator(Tenzor, H_structure)

    x = 0
    while len(kd_structure) > 0:
        [Tenzor, H_structure] = kronecker_helical_operator(Tenzor, H_structure, kd_structure)
        [Tenzor, H_structure] = momenta_helical_operator(Tenzor, H_structure)
        [Tenzor, H_structure] = momenta_momenta_helical_operator(Tenzor, H_structure)
        for y in kd_structure:
            if Tenzor.coeff(kd(y[0], y[1])) == 0:
                kd_structure.remove(y)
        if x == int(len(moznost) / 3 - 1):
            # Solve a problem: for example kd(indexb, indexB) -> len( kd_structure) > 1, I do not have a cycle
            break
        else:
            x += 1

    # [Tenzor, H_structure] = momenta_helical_operator(Tenzor, H_structure)
    # [Tenzor, H_structure] = momenta_momenta_helical_operator(Tenzor, H_structure)

    print(
        f"Recomputation tensor convolutions with "
        f"H(k, i, j), mom(k, j), and kd(i, j) from all past steps: \n{round(time.time() - t, 1)} sec"
    )

    p_structure = list()  # list of indeces for momentum p in Tenzor
    k_structure = list()  # list of indeces for momentum k in Tenzor
    q_structure = list()  # list of indeces for momentum q in Tenzor
    # It combines quantities with matching indices.
    for in1 in range(len(moznost)):
        Tenzor = Tenzor.subs(mom(k, in1) ** 2, k**2)
        Tenzor = Tenzor.subs(mom(q, in1) ** 2, q**2)
        Tenzor = Tenzor.subs(mom(q, in1) * mom(k, in1), k * q * z)
        # k.q = k q z, where z = cos(angle) = k . q/ |k| /|q|
        if Tenzor.coeff(mom(p, in1)) != 0:
            # H( , j, s) mom( ,j) mom( ,s) mom( , indexb) mom(p, i) mom(q, i) mom(q, indexB) = 0
            # or H( , j, indexb) mom( ,j) mom(p, i) mom(q, i) mom(q, indexB) = 0
            Tenzor = Tenzor.subs(mom(p, in1) * mom(q, in1) * mom(q, indexB), 0)
            Tenzor = Tenzor.subs(mom(p, in1) * mom(k, in1) * mom(k, indexB), 0)
            Tenzor = Tenzor.subs(mom(p, in1) * mom(q, in1) * mom(q, indexb), 0)
            Tenzor = Tenzor.subs(mom(p, in1) * mom(k, in1) * mom(k, indexb), 0)
            p_structure += [in1]  # list correspond to the index of momentum
        if Tenzor.coeff(mom(q, in1)) != 0:
            q_structure += [in1]
        if Tenzor.coeff(mom(k, in1)) != 0:
            k_structure += [in1]

    print(
        f"Computing tensor convolutions of the form mom(k, i)**2 = k**2, "
        f"mom(q, i)*mom(k, i) = k*q*z and mom(p, i)*mom(q, i)*mom(q, indexB) = 0: \n{round(time.time() - t, 1)} sec"
    )

    if UV_convergence_criterion == False:
        Tenzor_Lambda = H_structure_calculation_part_1(
            Tenzor, H_structure, indexB, indexb, p_structure, k_structure, q_structure, "lambda"
        )
        Tenzor_B = H_structure_calculation_part_1(
            Tenzor, H_structure, indexB, indexb, p_structure, k_structure, q_structure, "Bfield"
        )
    else:
        Tenzor_B = H_structure_calculation_part_1(
            Tenzor, H_structure, indexB, indexb, p_structure, k_structure, q_structure, "Bfield"
        )

    print(
        f"Tensor reduction (Passarino-Veltman procedure) of the structures "
        f"with four momentums: \n{round(time.time() - t, 1)} sec"
    )

    if UV_convergence_criterion == False:
        Tenzor_Lambda = H_structure_calculation_part_2(
            Tenzor_Lambda, H_structure, indexB, indexb, p_structure, "lambda"
        )
        Tenzor_B = H_structure_calculation_part_2(Tenzor_B, H_structure, indexB, indexb, p_structure, "Bfield")

        # lcs( i, j, l) - Levi-Civita symbol
        Tenzor_Lambda = Tenzor_Lambda.subs(lcs(s, indexb, indexB), -lcs(s, indexB, indexb))
        Tenzor_Lambda = simplify(expand(Tenzor_Lambda))
        Tenzor_B = Tenzor_B.subs(lcs(s, indexb, indexB), -lcs(s, indexB, indexb))
        Tenzor_B = simplify(expand(Tenzor_B))

        # Process Tenzor_B
        Tenzor_B = process_tensor(Tenzor_B, indexb, indexB)
        Tenzor_B = process_tensor(Tenzor_B, indexB, indexb)

        # Process Tenzor_Lambda
        Tenzor_Lambda = process_tensor(Tenzor_Lambda, indexb, indexB)
        Tenzor_Lambda = process_tensor(Tenzor_Lambda, indexB, indexb)

        Tensor_data = IntegrandTensorStructure(Tenzor_Lambda, Tenzor_B)
    else:
        Tenzor_B = H_structure_calculation_part_2(Tenzor_B, H_structure, indexB, indexb, p_structure, "Bfield")
        # lcs( i, j, l) - Levi-Civita symbol
        Tenzor_B = Tenzor_B.subs(lcs(s, indexb, indexB), -lcs(s, indexB, indexb))
        Tenzor_B = simplify(expand(Tenzor_B))

        # Process Tenzor_B
        Tenzor_B = process_tensor(Tenzor_B, indexb, indexB)
        Tenzor_B = process_tensor(Tenzor_B, indexB, indexb)

        Tensor_data = IntegrandTensorStructure(None, Tenzor_B)

    print(
        f"Tensor reduction (Passarino-Veltman procedure) of the structures "
        f"with two momentums: \n{round(time.time() - t, 1)} sec"
    )

    return Tensor_data
