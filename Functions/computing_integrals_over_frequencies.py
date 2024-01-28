import math
import sympy as sym
import sys
import time
from sympy import *

from typing import Any
from Functions.Data_classes import *
from Functions.SymPy_classes import *

# ------------------------------------------------------------------------------------------------------------------#
#                                          Сomputing integrals over frequencies
# ------------------------------------------------------------------------------------------------------------------#

# Integrals over frequency are calculated using the residue theorem


def get_info_how_to_close_contour(rational_function: Any, variable1: Any):
    """
    This function evaluates the complexity of calculating integrals of two variables using residues.
    The estimation is made on the basis of counting the number of poles lying in the upper/lower
    complex half-plane for each variable, respectively. To do this, for each variable,
    two counters corresponding to the closing of the loop up down are created.
    Their minimum gives the desired criterion.

    ARGUMENTS:

    rational_function -- a rational function whose denominator consists of the product of the powers
    of the functions chi_1 and chi_2,

    variable1 -- variable over which the integral is first calculated

    Note:

    This function uses information about the structure of the singularities of the functions 1/chi_1(k, w)
    and 1/chi_2(k, w) with respect to the variable w. Under the condition that u is positive,
    it can be shown that the position of the pole is determined only by the sign of w in chi_1, chi_2
    (the functions f_1, f_2 do not change the sign of the im(w) for any k)

    OUTPUT DATA EXAMPLE:

    computational_complexity_when_closing_UP_contour_in_plane_of_variable1 = 6

    computational_complexity_when_closing_DOWN_contour_in_plane_of_variable1 = 2

    computational_complexity_when_closing_UP_contour_in_plane_of_variable2 = 4

    computational_complexity_when_closing_DOWN_contour_in_plane_of_variable2 = 2
    """

    denominator = fraction(rational_function)[1]
    denominator_structure = factor_list(denominator)[1]
    number_of_multipliers = len(denominator_structure)

    computational_complexity_when_closing_UP_contour_in_plane_of_variable1 = 0
    computational_complexity_when_closing_DOWN_contour_in_plane_of_variable1 = 0

    computational_complexity_when_closing_UP_contour_in_plane_of_variable2 = 0
    computational_complexity_when_closing_DOWN_contour_in_plane_of_variable2 = 0

    for pole in range(number_of_multipliers):
        pole_equation = denominator_structure[pole][0]
        multiplicity = denominator_structure[pole][1]
        frequency = pole_equation.args[1]
        if frequency.has(variable1) == True:
            if frequency.could_extract_minus_sign() == True:
                # if this condition is fit, then the pole lies in the lower half-plane
                # (see note in the function description)
                computational_complexity_when_closing_DOWN_contour_in_plane_of_variable1 += multiplicity
            else:
                computational_complexity_when_closing_UP_contour_in_plane_of_variable1 += multiplicity
        else:
            if frequency.could_extract_minus_sign() == True:
                # if this condition is fit, then the pole lies in the lower half-plane
                # (see note in the function description)
                computational_complexity_when_closing_DOWN_contour_in_plane_of_variable2 += multiplicity
            else:
                computational_complexity_when_closing_UP_contour_in_plane_of_variable2 += multiplicity

    return [
        [
            computational_complexity_when_closing_UP_contour_in_plane_of_variable1,
            computational_complexity_when_closing_DOWN_contour_in_plane_of_variable1,
        ],
        [
            computational_complexity_when_closing_UP_contour_in_plane_of_variable2,
            computational_complexity_when_closing_DOWN_contour_in_plane_of_variable2,
        ],
    ]


def calculate_residues_sum(rational_function: Any, main_frequency: Any, second_frequency: Any):
    """
    This function calculates the sum of the residues (symbolically, at the level of functions f_1, f_2)
    of a rational function (without 2*pi*I factor) with respect to one given variable.

    GENERAL REQUIREMENTS FOR A RATIONAL FUNCTION:

    This function works ONLY for A = 0, A = 1.

    The denominator of rational_function must be the product of the powers of the chi_1.doit() and chi_2.doit(),
    i.e. be expressed in terms of functions f_1, f_2.
    The result of calculating the residues sum is also expressed in terms of f_1 and f_2.

    ARGUMENTS:

    rational_function -- a rational function whose denominator consists of the product of the powers
    of the functions chi_1.doit() and chi_2.doit(),

    main_frequency -- variable over which the residues are calculated,

    second_frequency -- other variables (which may not be, then this argument does not affect anything)

    Note:

    Calculating the residue from a standard formula requires a pole reduction operation
    by multiplying the fraction by the corresponding expression. Here raises a problem:
    ((x-a)/(-x + a)).subs(x, a) = 0. In order not to use the simplify() function
    (it takes a very long time), you need to bring the expression exactly to the form
    (x-a) /(-x + a) = -(x-a)/(-x + a), then -((x-a)/(x - a)).subs(x, a) = -1

    OUTPUT DATA EXAMPLE:

    As an example, here is represented the result for a simpler function than the characteristic diagram integrand.
    Nevertheless, the reduced function contains all the characteristic features of the corresponding integrands.

    rational_function = (w_k - 2)/(chi_1(k, w_k)**2*chi_1(q, -w_q-w_k)**2*chi_2(k, -w_k))).doit()

    sum_of_residues_in_upper_half_plane =
    2*(f_1(k, A) - 2)/((-f_1(k, A) - f_2(k, A))*(-w_q - f_1(k, A) - f_1(q, A))**3) +
    1/((-f_1(k, A) - f_2(k, A))*(-w_q - f_1(k, A) - f_1(q, A))**2) +
    (f_1(k, A) - 2)/((-f_1(k, A) - f_2(k, A))**2*(-w_q - f_1(k, A) - f_1(q, A))**2)

    sum_of_residues_in_lower_half_plane =
    (-w_q - f_1(q, A) - 2)/((-w_q - f_1(k, A) - f_1(q, A))**2*(w_q + f_1(q, A) - f_2(k, A))**2)
    + 1/((-w_q - f_1(k, A) - f_1(q, A))**2*(w_q + f_1(q, A) - f_2(k, A))) -
    2*(-w_q - f_1(q, A) - 2)/((-w_q - f_1(k, A) - f_1(q, A))**3*(w_q + f_1(q, A) - f_2(k, A))) -
    (-f_2(k, A) - 2)/((-f_1(k, A) - f_2(k, A))**2*(-w_q - f_1(q, A) + f_2(k, A))**2)

    sum_of_residues_in_upper_half_plane + sum_of_residues_in_lower_half_plane = 0,
    since the residue of the corresponding function at infinity is 0
    """

    numerator = fraction(rational_function)[0]
    denominator = fraction(rational_function)[1]
    denominator_structure = factor_list(denominator)[1]
    number_of_multipliers = len(denominator_structure)

    residues_in_upper_half_plane = list()
    residues_in_lower_half_plane = list()

    for pole in range(number_of_multipliers):
        pole_equation = denominator_structure[pole][0]
        multiplicity = denominator_structure[pole][1]

        if pole_equation.has(main_frequency) == True:
            # we solve a linear equation of the form w - smth = 0
            # solve_linear() function, unlike solve(), is less sensitive to the type of input data
            solution_of_pole_equation = solve_linear(pole_equation, symbols=[main_frequency])[1]
            # see note in description
            if denominator.has(main_frequency - solution_of_pole_equation) == True:
                calcullable_function = (
                    (numerator * (main_frequency - solution_of_pole_equation) ** multiplicity) / denominator
                ).diff(main_frequency, multiplicity - 1)
                residue_by_variable = calcullable_function.subs(main_frequency, solution_of_pole_equation)
                # residue cannot be zero
                if residue_by_variable == 0:
                    sys.exit("An error has been detected. The residue turned out to be zero.")

            else:
                calcullable_function = (
                    (numerator * (-main_frequency + solution_of_pole_equation) ** multiplicity) / denominator
                ).diff(main_frequency, multiplicity - 1)
                residue_by_variable = (-1) ** multiplicity * calcullable_function.subs(
                    main_frequency, solution_of_pole_equation
                )
                # residue cannot be zero
                if residue_by_variable == 0:
                    sys.exit("An error has been detected. The residue turned out to be zero.")

            if solution_of_pole_equation.subs(second_frequency, 0).could_extract_minus_sign() == True:
                residues_in_lower_half_plane.append(residue_by_variable)
            else:
                residues_in_upper_half_plane.append(residue_by_variable)
        # Error handling in this function takes several orders of magnitude longer than the
        # execution of the program itself

        # if simplify(sum(residues_in_upper_half_plane) + sum(residues_in_lower_half_plane)) != 0:
        # the residue at infinity is usually 0
        #     sys.exit("Error when calculating residues")

    sum_of_residues_in_upper_half_plane = sum(residues_in_upper_half_plane)
    sum_of_residues_in_lower_half_plane = sum(residues_in_lower_half_plane)

    return sum_of_residues_in_upper_half_plane, sum_of_residues_in_lower_half_plane


def calculating_frequency_integrals_in_two_loop_diagrams(
    integrand_with_denominator_from_xi_functions: Any, frequency1: Any, frequency2: Any
):
    """
    This function calculates integrals over frequencies using the residue theorem

    GENERAL REQUIREMENTS FOR AN INTEGRAND:

    The integrand_with_denominator_from_xi_functions denominator must be the product of the xi_1 and xi_2 powers.
    (Further, the .doit() operation is applied to integrand_with_denominator_from_xi_functions)

    ARGUMENTS:

    integrand_with_denominator_from_xi_functions -- a rational function whose denominator consists of the
    product of the powers of the functions xi_1 and xi_2,

    frequency1 -- variable over which the integral is calculated a first time

    frequency2 -- variable over which the integral is calculated a second time
    """

    original_numerator1 = fraction(integrand_with_denominator_from_xi_functions)[0]
    denominator_from_chi_functions = fraction(integrand_with_denominator_from_xi_functions)[1].doit()

    integrand_with_denominator_from_chi_functions = original_numerator1 / denominator_from_chi_functions

    computational_complexity_estimation = get_info_how_to_close_contour(
        integrand_with_denominator_from_chi_functions, frequency1
    )

    original_numerator2 = fraction(integrand_with_denominator_from_chi_functions)[0]
    denominator_from_f12_functions = fraction(integrand_with_denominator_from_chi_functions)[1].doit()

    integrand_with_denominator_from_f12_functions = original_numerator2 / denominator_from_f12_functions

    first_factor_2piI = 1
    second_factor_2piI = 1

    # calculate the residues for frequency1
    if computational_complexity_estimation[0][0] >= computational_complexity_estimation[0][1]:
        # computational complexity when closing the contour up >=
        # computational complexity when closing the contour down
        # i.e. it is advantageous to close the contour down
        residues_sum_without_2piI = calculate_residues_sum(
            integrand_with_denominator_from_f12_functions, frequency1, frequency2
        )[1]
        first_factor_2piI = -2 * pi * I
    else:
        # it is advantageous to close the contour up
        residues_sum_without_2piI = calculate_residues_sum(
            integrand_with_denominator_from_f12_functions, frequency1, frequency2
        )[0]
        first_factor_2piI = 2 * pi * I

    number_of_terms_in_sum = len(residues_sum_without_2piI.args)

    # calculate the residues for frequency2
    list_with_residues_sum_for_frequency2 = [0] * number_of_terms_in_sum
    if computational_complexity_estimation[1][0] >= computational_complexity_estimation[1][1]:
        # computational complexity when closing the contour up >=
        # computational complexity when closing the contour down
        # i.e. it is advantageous to close the contour down
        for i in range(number_of_terms_in_sum):
            term = residues_sum_without_2piI.args[i]
            list_with_residues_sum_for_frequency2[i] = calculate_residues_sum(term, frequency2, frequency1)[1]
        second_factor_2piI = -2 * pi * I
    else:
        # it is advantageous to close the contour up
        for i in range(number_of_terms_in_sum):
            term = residues_sum_without_2piI.args[i]
            list_with_residues_sum_for_frequency2[i] = calculate_residues_sum(term, frequency2, frequency1)[0]
        second_factor_2piI = 2 * pi * I

    total_sum_of_residues_for_both_frequencies = (
        first_factor_2piI * second_factor_2piI * sum(list_with_residues_sum_for_frequency2)
    )

    return total_sum_of_residues_for_both_frequencies


def find_duplicates(lst: list):
    """
    The find_duplicates function takes a list lst as input and returns a dictionary where the keys
    are the duplicate elements in lst, and the values are lists of their corresponding indices.
    """
    duplicates = {}
    for i, item in enumerate(lst):
        if item in duplicates:
            duplicates[item].append(i)
        else:
            duplicates[item] = [i]
    return {k: v for k, v in duplicates.items() if len(v) > 1}


def reduction_to_common_denominator(total_sum_of_residues_for_both_frequencies: Any):
    """
    The function reduces to a common denominator the terms obtained after frequency integration.
    The function is entirely focused on a specific input data structure.

    ARGUMENTS:

    total_sum_of_residues_for_both_frequencies -- (+-4pi^2)*sum of residues

    variable_substitution is a parameter (T/F) which says whether the expression is UV-finite

    OUTPUT DATA EXAMPLE:

    residues_sum_without_prefactor = too long

    prefactor = 4*pi**2*A**3*(-sc_prod(B, k) - sc_prod(B, q))*D_v(k)*D_v(q)*sc_prod(B, q)**3
    """

    residues_sum = total_sum_of_residues_for_both_frequencies / (4 * pi**2)
    # the multiplier (4 * pi**2) will be related to prefactor
    prefactor = 1  # here we store the same (located in each term) factors from the numerator

    residues_sum_to_common_denominator = together(residues_sum)
    new_denominator = fraction(residues_sum_to_common_denominator)[1]
    new_numerator = fraction(residues_sum_to_common_denominator)[0]

    # here we store the same (available in each term) multipliers
    list_with_prefactors = list()
    list_with_prefactor_multipliers = list()
    # here we store list_with_prefactors
    list_with_lists_with_prefactor_multipliers = list()
    part_of_numerator_without_prefactor = 1

    list_with_numerator_multipliers = new_numerator.as_ordered_factors()

    # each numerator in residues_sum contains the same multipliers. Here we select them
    for j in range(len(list_with_numerator_multipliers)):
        multiplier = list_with_numerator_multipliers[j]

        if multiplier.has(A) or multiplier.has(D_v) or multiplier.has(sc_prod):
            list_with_prefactor_multipliers.append(multiplier)
        # here we leave those terms that differ from term to term
        elif (
            type(multiplier) == sym.core.numbers.NegativeOne
            or multiplier.has(I)
            or multiplier.has(alpha)
            or multiplier.has(alpha_star)
            or multiplier.has(beta)
            or multiplier.has(beta_star)
        ):
            part_of_numerator_without_prefactor *= multiplier
        else:
            sys.exit("Unknown object type in numerator")

    list_with_lists_with_prefactor_multipliers.append(list_with_prefactor_multipliers)
    prefactor = 4 * pi**2 * math.prod(list_with_prefactor_multipliers)
    list_with_prefactors.append(prefactor)

    # prefactor must be the same for all terms.
    # As an additional verification, we check that this is the case
    for i in range(len(list_with_prefactors)):
        if simplify(prefactor - list_with_prefactors[i - 1]) != 0:
            sys.exit("Different prefactors in different terms")

    residues_sum_without_prefactor = part_of_numerator_without_prefactor / new_denominator

    t = time.time()

    if simplify(together(total_sum_of_residues_for_both_frequencies) - prefactor * residues_sum_without_prefactor) != 0:
        sys.exit("Error in the procedure for reducing to a common denominator")

    print(f"Total verification took: {round(time.time() - t, 1)} sec")

    # calculating obtained fraction after replacing of momentums k, q --> B*k/nuo, B*q/nuo

    print(f"\nPerforming variable substitution k, q --> B*k/nuo, B*q/nuo.")

    fraction_after_substitution = reduction_to_common_denominator_after_substitution(
        prefactor * residues_sum_without_prefactor,
        new_denominator.as_ordered_factors(),
        part_of_numerator_without_prefactor,
        prefactor,
    )
    residues_sum_without_prefactor_after_subs = fraction_after_substitution[0]
    new_prefactor_after_subs = fraction_after_substitution[1]

    # else:
    #     sys.exit("Atypical structure of the residues sum")

    diagram_expression = IntegrandIsRationalFunction(
        residues_sum_without_prefactor, prefactor, residues_sum_without_prefactor_after_subs, new_prefactor_after_subs
    )

    return diagram_expression


def reduction_to_common_denominator_after_substitution(
    residues_sum_with_common_denominator: Any, common_denominator: Any, new_numerator: Any, prefactor: Any
):
    """
    This function is supposed to be used inside the reduction_to_common_denominator() function.
    It transforms the expression obtained in reduction_to_common_denominator() after replacing
    k, q --> B*k/nuo, B*q/nuo. The transformation is that the dimensional factors B and nuo must
    be taken out as a prefactor, and the remaining expression will depend only on uo.

    ARGUMENTS:

    residues_sum_with_common_denominator, common_denominator, new_numerator, prefactor are given by
    reduction_to_common_denominator()

    OUTPUT DATA EXAMPLE:

    residues_sum_without_prefactor_after_subs = too long

    new_prefactor_after_subs =
    """

    # creating a list with multipliers of common denominator
    common_denominator_after_substitution = list()
    # idea is to highlight the common factor (B^2/nuo)^n before the corresponding product
    # (product of common_denominator_after_substitution elemets)
    denominator_prefactor_after_subs = 1
    denominator_counter = 0  # a counter to check that all terms in the corresponding sum are processed

    for i in range(len(common_denominator)):
        # perform changing the variables in the denominator
        multiplier_after_substitution = common_denominator[i].subs(k, B * k / nuo).subs(q, B * q / nuo)
        # Starting processing multipliers. As a result of processing, a factor of the form
        # (B^2/nuo)^some_power should appear before each of them

        if type(multiplier_after_substitution) == sym.core.power.Pow:
            exponent_number_value = multiplier_after_substitution.args[1]
            exponent_base_value = multiplier_after_substitution.args[0]
            exponent_base_value = expand((nuo / B**2) * exponent_base_value)
            multiplier_after_substitution = exponent_base_value**exponent_number_value
            denominator_prefactor_after_subs *= (B**2 / nuo) ** exponent_number_value
            denominator_counter += 1

        elif type(multiplier_after_substitution) == sym.core.mul.Mul:
            pow_multiplier = 1
            for j in range(len(multiplier_after_substitution.args)):
                argument = multiplier_after_substitution.args[j]
                if argument.has(B):
                    if type(argument) == sym.core.power.Pow:
                        pow_value = argument.args[1]
                        denominator_prefactor_after_subs *= B**pow_value
                        pow_multiplier *= 1 / (B**pow_value)

                elif argument.has(nuo):
                    if type(argument) == sym.core.power.Pow:
                        pow_value = argument.args[1]
                        denominator_prefactor_after_subs *= (nuo) ** pow_value
                        pow_multiplier *= 1 / (nuo**pow_value)

                elif argument.has(f_1) or argument.has(f_2):
                    denominator_prefactor_after_subs *= 1
                else:
                    sys.exit("Unexpected expression for type Mul")

            multiplier_after_substitution = multiplier_after_substitution * pow_multiplier
            denominator_counter += 1

        elif type(multiplier_after_substitution) == sym.core.add.Add:
            multiplier_after_substitution = expand((nuo / B**2) * multiplier_after_substitution)
            denominator_prefactor_after_subs *= B**2 / nuo
            denominator_counter += 1

        elif type(multiplier_after_substitution) == sym.core.numbers.Integer:
            multiplier_after_substitution = multiplier_after_substitution
            denominator_counter += 1
        else:
            sys.exit("Unaccounted for type of multiplier in the denominator " "after the momentums replacement.")

        common_denominator_after_substitution.append(multiplier_after_substitution)

    # checking that all multipliers have been processed
    if denominator_counter != len(common_denominator):
        sys.exit(
            "When processing the denominator after the replacement of momentums, "
            "not all multipliers were taken into account."
        )

    new_denominator_after_subs = math.prod(common_denominator_after_substitution)

    if new_denominator_after_subs.has(B) or new_denominator_after_subs.has(nuo):
        sys.exit("Not all factors of the form B**2/nuo in denominator are placed in the prefactor")

    # calculating new numerator after replacing of variables
    # idea is to highlight the common factor (B^2/nuo)^n before the corresponding product
    new_numerator_after_subs = 0

    if type(new_numerator) == sym.core.add.Add:
        list_with_new_numerator_terms = new_numerator.args
        numerator_prefactors_after_subs = list()

        for i in range(len(list_with_new_numerator_terms)):
            term_in_new_numerator = list_with_new_numerator_terms[i]
            numerator_prefactor_after_subs = 1

            if type(term_in_new_numerator) == sym.core.mul.Mul:
                # perform changing the variables in the numerator term
                term_in_new_numerator = term_in_new_numerator.subs(k, B * k / nuo).subs(q, B * q / nuo)
                term_in_new_numerator_after_subs = list()
                counter_check = 0  # a counter to check that all terms in the sum are processed

                for j in range(len(term_in_new_numerator.args)):
                    multiplier_in_term = term_in_new_numerator.args[j]
                    # Starting processing multipliers. As a result of processing, a factor of the form
                    # (B^2/nuo)^some_power should appear before each of them

                    if type(multiplier_in_term) == sym.core.add.Add:
                        multiplier_in_term = expand((nuo / B**2) * multiplier_in_term)
                        numerator_prefactor_after_subs *= B**2 / nuo
                        counter_check += 1

                    elif type(multiplier_in_term) == sym.core.power.Pow:
                        multiplier_power_base = multiplier_in_term.args[0]
                        multiplier_power_exp = multiplier_in_term.args[1]

                        if type(multiplier_power_base) == sym.core.add.Add:
                            multiplier_power_base = expand((nuo / B**2) * multiplier_power_base)
                            numerator_prefactor_after_subs *= (B**2 / nuo) ** multiplier_power_exp
                            multiplier_in_term = multiplier_power_base**multiplier_power_exp
                            counter_check += 1
                        else:
                            if multiplier_in_term.has(B) or multiplier_in_term.has(nuo):
                                numerator_prefactor_after_subs *= multiplier_in_term
                                multiplier_in_term = 1
                                counter_check += 1
                            else:
                                counter_check += 1
                    # For multipliers of these types, the removal of the B^2/nuo factor
                    # is carried out automatically and multiplicatively
                    # (see the description in the SymPY_classes)
                    elif (
                        type(multiplier_in_term) == alpha
                        or type(multiplier_in_term) == alpha_star
                        or type(multiplier_in_term) == beta
                        or type(multiplier_in_term) == beta_star
                        or type(multiplier_in_term) == f_1
                        or type(multiplier_in_term) == f_2
                        or type(multiplier_in_term) == sym.core.numbers.Integer
                        or type(multiplier_in_term) == sym.core.numbers.NegativeOne
                        or type(multiplier_in_term) == sym.core.numbers.ImaginaryUnit
                    ):
                        counter_check += 1

                    else:
                        sys.exit("Unaccounted type of multiplier in the numerator after the momentums replacement.")

                    term_in_new_numerator_after_subs.append(multiplier_in_term)

                # checking that all multipliers have been processed
                if counter_check != len(term_in_new_numerator.args):
                    sys.exit(
                        "When processing the numerator after the replacement of momentums, "
                        "not all multipliers were taken into account."
                    )

                numerator_prefactors_after_subs.append(numerator_prefactor_after_subs)

                new_numerator_after_subs += math.prod(term_in_new_numerator_after_subs)
            else:
                sys.exit("The term in the numerator is of the wrong type")

            # prefactor_after_subs (expression of the form (B^2/nuo)^some_value) must be the same for all terms
            # (the same dimensions for all terms).
            # As an additional verification, we check that this is the case

            if simplify(numerator_prefactor_after_subs - numerator_prefactors_after_subs[i - 1]) != 0:
                sys.exit("Different prefactors in terms")

        numerator_prefactor_after_subs = numerator_prefactors_after_subs[0]

        if new_numerator_after_subs.has(B) or new_numerator_after_subs.has(nuo):
            sys.exit("Not all factors of the form B**2/nuo in numerator are placed in the prefactor")

        residues_sum_after_subs_without_prefactor = new_numerator_after_subs / new_denominator_after_subs

        # perform changing the variables in the prefactor
        # For multipliers in prefactor, the removal of the B^2/nuo factor
        # is carried out automatically and multiplicatively
        prefactor_after_subs = prefactor.subs(k, B * k / nuo).subs(q, B * q / nuo)
        list_with_prefactor_multipliers = list()

        for i in range(len(prefactor_after_subs.args)):
            prefactor_after_subs_mul = prefactor_after_subs.args[i]

            # initially written for expressions like (sc_prod(1) + sc_prod(2))
            if type(prefactor_after_subs_mul) == sym.core.add.Add and prefactor_after_subs_mul.has(sc_prod):
                prefactor_after_subs_mul = B**2 * expand(nuo * prefactor_after_subs_mul / B**2) / nuo
            # initially written for expressions like (sc_prod(1) + sc_prod(2))^some_value
            elif type(prefactor_after_subs_mul) == sym.core.power.Pow and prefactor_after_subs_mul.has(sc_prod):
                prefactor_after_subs_mul_base = prefactor_after_subs_mul.args[0]
                prefactor_after_subs_mul_exp = prefactor_after_subs_mul.args[1]
                prefactor_after_subs_mul = (B**2 / nuo) ** prefactor_after_subs_mul_exp * expand(
                    prefactor_after_subs_mul_base * nuo / B**2
                ) ** prefactor_after_subs_mul_exp

            list_with_prefactor_multipliers.append(prefactor_after_subs_mul)

        prefactor_after_subs = math.prod(list_with_prefactor_multipliers)

        # the final prefactor is the
        # prefactor * (dimension factor from the numerator / dimension factor from the denominator)
        new_prefactor_after_subs = prefactor_after_subs * (
            numerator_prefactor_after_subs / denominator_prefactor_after_subs
        )

        # This is a partial check that nothing is forgotten.
        # An honest calculation of this difference for any B and nuo can take hours.

        print(
            f"\nVerifying that after the momentums replacement k, q --> |B|*k/nuo, |B|*q/nuo "
            f"all terms of the original fraction have been processed. "
        )

        t1 = time.time()

        if (
            simplify(
                (new_prefactor_after_subs * residues_sum_after_subs_without_prefactor)
                .subs(B, 1)
                .subs(nuo, 1)
                .subs(b, 1)
                - residues_sum_with_common_denominator.subs(B, 1).subs(nuo, 1).subs(b, 1)
            )
            != 0
        ):
            sys.exit("Error in the procedure for reducing to a common denominator after variables substitution")

        print(f"Verification took: {round(time.time() - t1, 1)} sec")

        return residues_sum_after_subs_without_prefactor, new_prefactor_after_subs


def partial_simplification_of_diagram_expression(residues_sum_without_prefactor: Any):
    """
    The function performs a partial simplification of the given expression.
    The function is entirely focused on a specific input expression structure.

    ARGUMENTS:

    residues_sum_without_prefactor is given by reduction_to_common_denominator(),

    PARAMETERS:

    PROPERTIES:

    OUTPUT DATA EXAMPLE:

    too long

    """
    t = time.time()

    simplified_expression = re(residues_sum_without_prefactor.doit())

    # I arises in the diagram only from the tensor structure, this part of the diagram must be real
    if simplified_expression.has(I):
        sys.exit("Simplified expression contains raw imaginary parts")

    print(f"Simplification took: {round(time.time() - t, 1)} sec")

    return simplified_expression


def prefactor_simplification(new_prefactor_after_subs: Any):
    """
    The function additionally splits the prefactor into momentum-dependent and momentum-independent parts.

    ARGUMENTS:

    new_prefactor_after_subs is given by reduction_to_common_denominator()

    OUTPUT DATA EXAMPLE:

    dim_factor = 4*pi**2*A**3*nuo**5*(B/nuo)**(-2*d - 4*eps + 8)/B**10

    part_to_integrand = (-sc_prod(1, k) - sc_prod(1, q))*D_v(k)*D_v(q)*sc_prod(1, q)**3
    """
    list_with_prefactor_args = new_prefactor_after_subs.args
    dim_factor = 1
    part_to_integrand = 1
    for i in range(len(list_with_prefactor_args)):
        term = list_with_prefactor_args[i]
        if term.has(sc_prod):
            part_to_integrand *= term
        elif term.has(D_v):
            term = term.doit()
            for j in range(len(term.args)):
                term_arg = term.args[j]
                if term_arg.has(k) or term_arg.has(q):
                    part_to_integrand *= term_arg
                else:
                    dim_factor *= term_arg
        else:
            dim_factor *= term

    prefactor_data = IntegrandPrefactorAfterSubstitution(part_to_integrand, dim_factor)

    return prefactor_data
