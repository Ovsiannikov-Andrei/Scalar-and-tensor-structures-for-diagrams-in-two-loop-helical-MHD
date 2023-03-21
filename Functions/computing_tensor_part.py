from sympy import *

from Functions.SymPy_classes import *

# ------------------------------------------------------------------------------------------------------------------#
#                             Calculation of the tensor part of the integrand for a diagram
# ------------------------------------------------------------------------------------------------------------------#


def tensor_structure_term_classification(expr, list_for_expr_args):
    """
    This function compares expr (some kind of one multiplier) with a template
    and adds its arguments to the list.
    """
    match expr:
        case hyb():
            return list_for_expr_args.append(["hyb", list(expr.args)])
        case P():
            return list_for_expr_args.append(["P", list(expr.args)])
        case H():
            return list_for_expr_args.append(["H", list(expr.args)])
        case kd():
            return list_for_expr_args.append(["kd", list(expr.args)])
        case _:  # symbols or numbers
            return list_for_expr_args.append([expr])


def get_tensor_structure_arguments_structure(expr):
    """
    This function returns the expr expression structure as [expr_term_1, expr_term_2, ...],
    expr_term_1 = [term_multiplier_1, term_multiplier_2, ...], where, for example,
    term_multiplier_1 = ['P', [k, 2, 4]].

    ARGUMENTS:

    expr -- expression containing functions hyb(), P(), H(), kd(), symbols rho, A, I and numbers

    OUTPUT DATA EXAMPLE:

    expr = hyb(-q, 9)*kd(11, 10)*P(k, 2, 6) + A*hyb(-q, 10)*kd(11, 9) + kd(12, 9)

    get_tensor_structure_arguments_structure(expr) =
    [
    [-1, A, ['hyb', [q, 10]], ['kd', [11, 9]]], [-1, ['P', [k, 2, 6]], ['hyb', [q, 9]], ['kd', [11, 10]]],
    [['kd', [12, 9]]]
    ]

    """
    expr_new = expand(expr)  # a priori, open all brackets

    list_for_term_multipliers = list()

    number_of_terms_in_expr = len(expr_new.args)

    for i in range(number_of_terms_in_expr):
        term = expr_new.args[i]

        list_for_multiplier_args = list()
        if term.is_Mul:  # adds its arguments to the list
            number_of_multipliers_in_term = len(term.args)

            for j in range(number_of_multipliers_in_term):
                multiplier = term.args[j]
                tensor_structure_term_classification(multiplier, list_for_multiplier_args)

        else:  # term in expr_new has one element
            tensor_structure_term_classification(term, list_for_multiplier_args)

        list_for_term_multipliers.append(list_for_multiplier_args)

    return list_for_term_multipliers


def dosad(zoznam, ind_hod, struktura, pozicia):  # ?????????
    if ind_hod in zoznam:
        return zoznam
    elif ind_hod not in struktura:
        return list()
    elif zoznam[pozicia] != -1:
        return list()
    elif len(zoznam) - 1 == pozicia:
        return zoznam[:pozicia] + [ind_hod]
    else:
        return zoznam[:pozicia] + [ind_hod] + zoznam[pozicia + 1 :]
