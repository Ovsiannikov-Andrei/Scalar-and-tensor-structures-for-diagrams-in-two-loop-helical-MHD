import math
import sympy as sym
from sympy import *

from Functions.SymPy_classes import *

# ------------------------------------------------------------------------------------------------------------------#
#                                          Сomputing integrals over frequencies
# ------------------------------------------------------------------------------------------------------------------#

# Integrals over frequency are calculated using the residue theorem


def get_info_how_to_close_contour(rational_function, variable1):
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


def calculate_residues_sum(rational_function, main_variable, parameter):
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

    main_variable -- variable over which the residues are calculated

    parameter -- other variables (which may not be, then this argument does not affect anything)

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

        if pole_equation.has(main_variable) == True:
            # we solve a linear equation of the form w - smth = 0
            # solve_linear() function, unlike solve(), is less sensitive to the type of input data
            solution_of_pole_equation = solve_linear(pole_equation, symbols=[main_variable])[1]
            # see note in description
            if denominator.has(main_variable - solution_of_pole_equation) == True:
                calcullable_function = (
                    (numerator * (main_variable - solution_of_pole_equation) ** multiplicity) / denominator
                ).diff(main_variable, multiplicity - 1)
                residue_by_variable = calcullable_function.subs(main_variable, solution_of_pole_equation)
                # residue cannot be zero
                if residue_by_variable == 0:
                    sys.exit("An error has been detected. The residue turned out to be zero.")

            else:
                calcullable_function = (
                    (numerator * (-main_variable + solution_of_pole_equation) ** multiplicity) / denominator
                ).diff(main_variable, multiplicity - 1)
                residue_by_variable = (-1) ** multiplicity * calcullable_function.subs(
                    main_variable, solution_of_pole_equation
                )
                # residue cannot be zero
                if residue_by_variable == 0:
                    sys.exit("An error has been detected. The residue turned out to be zero.")

            if solution_of_pole_equation.subs(parameter, 0).could_extract_minus_sign() == True:
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
    integrand_with_denominator_from_xi_functions, frequency1, frequency2
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


def reduction_to_common_denominator(total_sum_of_residues_for_both_frequencies, variable_substitution):
    """
    The function reduces to a common denominator the terms obtained after frequency integration.
    The function is entirely focused on a specific input data structure.

    ARGUMENTS:

    total_sum_of_residues_for_both_frequencies -- (+-4pi^2)*sum of residues

    PARAMETERS:

    PROPERTIES:

    OUTPUT DATA EXAMPLE:

    residues_sum_without_prefactor = too long

    prefactor = 4*pi**2*A**3*(-sc_prod(B, k) - sc_prod(B, q))*D_v(k)*D_v(q)*sc_prod(B, q)**3
    """

    residues_sum = total_sum_of_residues_for_both_frequencies / (4 * pi**2)
    prefactor = 1
    prefactor_after_substitution = 1

    # calculating common denominator
    if type(residues_sum) == sym.core.add.Add:
        number_of_terms = len(residues_sum.args)
        list_with_denominators = list()
        list_with_numerators = list()
        list_with_all_multipliers_in_all_denominators = list()
        list_with_all_multipliers_in_all_numerators = list()

        for i in range(number_of_terms):
            term = residues_sum.args[i]
            term_numerator = fraction(term)[0]
            term_denominator = fraction(term)[1]
            if type(term_denominator) == sym.core.add.Mul:
                list_with_denominators.append(term_denominator)
                list_with_all_multipliers_in_all_denominators.append(list(term_denominator.args))
            else:
                sys.exit("Atypical structure of the denominator")

            if type(term_numerator) == sym.core.add.Mul:
                list_with_all_multipliers_in_all_numerators.append(list(term_numerator.args))
                list_with_numerators.append(term_numerator)
            else:
                sys.exit("Atypical structure of the numerator")

        # combine all factors of all denominators into a common list
        merge_list_with_term_denominators = [
            item for sublist in list_with_all_multipliers_in_all_denominators for item in sublist
        ]

        common_denominator = list()

        # exclude repeating multipliers
        for x in merge_list_with_term_denominators:
            if x not in common_denominator:
                if -x not in common_denominator:
                    common_denominator.append(x)

        new_denominator = math.prod(common_denominator)

        # calculating common denominator after replacing of variables
        if variable_substitution == True:
            # creating a list with multipliers
            common_denominator_after_substitution = list()
            # idea is to highlight the common factor (B^2/nuo)^n before the corresponding product
            denominator_prefactor_after_subs = 1

            for i in range(len(common_denominator)):
                multiplier_after_substitution = common_denominator[i].subs(k, B * k / nuo).subs(q, B * q / nuo)

                if type(multiplier_after_substitution) == sym.core.power.Pow:
                    exponent_value = multiplier_after_substitution.args[1]
                    exponent_base_value = multiplier_after_substitution.args[0]
                    exponent_base_value = expand((nuo / B**2) * exponent_base_value)
                    multiplier_after_substitution = exponent_base_value**exponent_value
                    denominator_prefactor_after_subs *= (B**2 / nuo) ** exponent_value

                elif type(multiplier_after_substitution) == sym.core.mul.Mul:
                    for j in range(len(multiplier_after_substitution.args)):
                        argument = multiplier_after_substitution.args[j]
                        if argument.has(B) == False and argument.has(nuo) == False:
                            multiplier_after_substitution = argument
                        if type(multiplier_after_substitution) == sym.core.power.Pow:
                            exponent_value = multiplier_after_substitution.args[1]
                    denominator_prefactor_after_subs *= (B**2 / nuo) ** exponent_value

                elif type(multiplier_after_substitution) == sym.core.add.Add:
                    multiplier_after_substitution = expand((nuo / B**2) * multiplier_after_substitution)
                    denominator_prefactor_after_subs *= B**2 / nuo

                elif type(multiplier_after_substitution) == sym.core.numbers.Integer:
                    multiplier_after_substitution = multiplier_after_substitution
                else:
                    sys.exit("Problems in the denominator after variables replacing")

                common_denominator_after_substitution.append(multiplier_after_substitution)

            new_denominator_after_subs = math.prod(common_denominator_after_substitution)

        # calculating common numerator
        new_numerator = 0
        list_with_lists_with_prefactor_multipliers = list()
        list_with_prefactors = list()

        for i in range(number_of_terms):
            term = residues_sum.args[i]
            term_numerator = list_with_numerators[i]
            list_with_numerator_multipliers = term_numerator.args
            number_of_numerator_multipliers = len(list_with_numerator_multipliers)
            list_with_prefactor_multipliers = list()
            part_of_numerator_without_prefactor = 1

            for j in range(number_of_numerator_multipliers):
                multiplier = list_with_numerator_multipliers[j]

                if multiplier.has(A):
                    list_with_prefactor_multipliers.append(multiplier)
                elif multiplier.has(D_v):
                    list_with_prefactor_multipliers.append(multiplier)
                elif multiplier.has(sc_prod):
                    list_with_prefactor_multipliers.append(multiplier)
                elif multiplier.has(alpha) or multiplier.has(alpha_star):
                    part_of_numerator_without_prefactor *= multiplier
                elif multiplier.has(beta) or multiplier.has(beta_star):
                    part_of_numerator_without_prefactor *= multiplier
                else:
                    sys.exit("Unknown object type in numerator")

            list_with_lists_with_prefactor_multipliers.append(list_with_prefactor_multipliers)
            prefactor = 4 * pi**2 * math.prod(list_with_prefactor_multipliers)
            list_with_prefactors.append(prefactor)

            # prefactor must be the same for all terms. As an additional verification, we check that this is the case
            if simplify(prefactor - list_with_prefactors[i - 1]) != 0:
                sys.exit("Different prefactors in different terms")

            term_denominator = list_with_denominators[i]

            # divide the common denominator of the fraction by the denominator of the concret term to get the factor
            # by which the numerator should be multiplied
            factor_by_which_numerator_is_multiplied = fraction(math.prod(common_denominator) / term_denominator)[0]

            # no automatic reduction of contributions of the form (a + b)/(-a - b)
            non_automatically_reduced_contribution = fraction(math.prod(common_denominator) / term_denominator)[1]

            # for contributions of the form (a + b)/(-a - b) we explicitly write out the reduction procedure
            if type(non_automatically_reduced_contribution) == sym.core.add.Mul:
                # case when there are product of several contributions of the type (a + b)/(-a - b)
                for j in range(len(non_automatically_reduced_contribution.args)):
                    factor_by_which_numerator_is_multiplied = -factor_by_which_numerator_is_multiplied / (
                        -non_automatically_reduced_contribution.args[j]
                    )
            else:
                # case of one contribution of the type (a + b)/(-a - b)
                factor_by_which_numerator_is_multiplied = -factor_by_which_numerator_is_multiplied / (
                    -non_automatically_reduced_contribution
                )

            # checking that the reduction is done correctly
            if fraction(factor_by_which_numerator_is_multiplied)[1] == 1:
                new_numerator += factor_by_which_numerator_is_multiplied * part_of_numerator_without_prefactor
            else:
                sys.exit("Error in the procedure of the new numerator calculating")

        residues_sum_without_prefactor = new_numerator / new_denominator

        # calculating new numerator after replacing of variables
        if variable_substitution == True:
            # idea is to highlight the common factor (B^2/nuo)^n before the corresponding product
            new_numerator_after_subs = 0

            if type(new_numerator) == sym.core.add.Add:
                list_with_new_numerator_terms = new_numerator.args
                numerator_prefactors_after_subs = list()

                for i in range(len(list_with_new_numerator_terms)):
                    term_in_new_numerator = list_with_new_numerator_terms[i]
                    numerator_prefactor_after_subs = 1

                    if type(term_in_new_numerator) == sym.core.mul.Mul:
                        term_in_new_numerator = term_in_new_numerator.subs(k, B * k / nuo).subs(q, B * q / nuo)
                        term_in_new_numerator_after_subs = list()

                        for j in range(len(term_in_new_numerator.args)):
                            multiplier_in_term = term_in_new_numerator.args[j]

                            if type(multiplier_in_term) == sym.core.add.Add:
                                multiplier_in_term = expand((nuo / B**2) * multiplier_in_term)
                                numerator_prefactor_after_subs *= B**2 / nuo

                            elif type(multiplier_in_term) == sym.core.power.Pow:
                                if multiplier_in_term.has(B) or multiplier_in_term.has(nuo):
                                    numerator_prefactor_after_subs *= multiplier_in_term
                                    multiplier_in_term = 1
                                else:
                                    multiplier_in_term = multiplier_in_term

                            elif type(multiplier_in_term) == alpha or type(multiplier_in_term) == alpha_star:
                                multiplier_in_term = multiplier_in_term

                            elif type(multiplier_in_term) == beta or type(multiplier_in_term) == beta_star:
                                multiplier_in_term = multiplier_in_term

                            elif type(multiplier_in_term) == sym.core.numbers.Integer:
                                multiplier_in_term = multiplier_in_term

                            elif type(multiplier_in_term) == sym.core.numbers.NegativeOne:
                                multiplier_in_term = multiplier_in_term

                            else:
                                sys.exit("Problems in the numerator after variables replacing")

                            term_in_new_numerator_after_subs.append(multiplier_in_term)

                        numerator_prefactors_after_subs.append(numerator_prefactor_after_subs)

                    new_numerator_after_subs += math.prod(term_in_new_numerator_after_subs)
                    # prefactor must be the same for all terms.
                    # As an additional verification, we check that this is the case
                    if simplify(numerator_prefactor_after_subs - numerator_prefactors_after_subs[i - 1]) != 0:
                        sys.exit("Different prefactors in different terms")

                numerator_prefactor_after_subs = numerator_prefactors_after_subs[0]

                residues_sum_without_prefactor_after_subs = new_numerator_after_subs / new_denominator_after_subs

                prefactor_after_subs = prefactor.subs(k, B * k / nuo).subs(q, B * q / nuo)

                list_with_prefactor_multipliers = list()

                for i in range(len(prefactor_after_subs.args)):
                    prefactor_after_subs_mul = prefactor_after_subs.args[i]

                    if type(prefactor_after_subs_mul) == sym.core.add.Add and prefactor_after_subs_mul.has(sc_prod):
                        prefactor_after_subs_mul = B**2 * expand(nuo * prefactor_after_subs_mul / B**2) / nuo
                    list_with_prefactor_multipliers.append(prefactor_after_subs_mul)

                prefactor_after_subs = math.prod(list_with_prefactor_multipliers)

                new_prefactor_after_subs = prefactor_after_subs * (
                    numerator_prefactor_after_subs / denominator_prefactor_after_subs
                )

    else:
        sys.exit("Atypical structure of the residues sum")

    # verification of procedure for self-consistency
    # this procedure must be the identity transformation
    if simplify(prefactor * residues_sum_without_prefactor - total_sum_of_residues_for_both_frequencies) != 0:
        sys.exit("Error in the procedure for reducing to a common denominator")

    if variable_substitution == True:
        if (
            simplify(
                new_prefactor_after_subs * residues_sum_without_prefactor_after_subs
                - total_sum_of_residues_for_both_frequencies.subs(k, B * k / nuo).subs(q, B * q / nuo)
            )
            != 0
        ):
            sys.exit("Error in the procedure for reducing to a common denominator after variables substitution")

    return [
        [residues_sum_without_prefactor, prefactor],
        [residues_sum_without_prefactor_after_subs, new_prefactor_after_subs],
    ]


def partial_simplification_of_diagram_expression(residues_sum_without_prefactor):
    """
    The function performs a partial simplification of the given expression.
    The function is entirely focused on a specific input expression structure.

    ARGUMENTS:

    residues_sum_without_prefactor is given by reduction_to_common_denominator(),
    variable_substitution

    PARAMETERS:

    PROPERTIES:

    OUTPUT DATA EXAMPLE:

    too long

    """

    original_numerator = fraction(residues_sum_without_prefactor)[0]
    original_denominator = fraction(residues_sum_without_prefactor)[1]

    simplified_denominator = 1
    simplified_numerator = 0

    original_denominator = original_denominator.doit()
    original_numerator = original_numerator.doit()

    # after applying the .doit() operation, original_denominator and original_numerator may have a
    # numerical denominator (integer_factor). Eliminate it as follows
    # original_numerator/original_denominator = original_numerator*integer_factor/integer_factor*original_denominator
    if fraction(original_denominator)[1] != 1:
        if type(fraction(original_denominator)[1]) == sym.core.numbers.Integer:
            integer_factor = fraction(original_denominator)[1]
            original_denominator_new = original_denominator * integer_factor
            original_numerator_new = original_numerator * integer_factor
        else:
            sys.exit("Atypical structure of the denominator")

    # handling the numerator
    if type(original_numerator_new) == sym.core.add.Add:
        number_of_terms_in_numerator = len(original_numerator_new.args)

        for j in range(number_of_terms_in_numerator):
            term_in_numerator = original_numerator_new.args[j]
            simplified_term_in_numerator = 1
            I_factor_in_numerator = 1

            for k in range(len(term_in_numerator.args)):
                multiplier = term_in_numerator.args[k]
                re_multiplier = re(multiplier)
                im_multiplier = im(multiplier)

                if re_multiplier == 0 and im_multiplier != 0:
                    new_multiplier = im_multiplier
                    I_factor_in_numerator *= I
                elif re_multiplier != 0 and im_multiplier == 0:
                    new_multiplier = re_multiplier
                else:
                    sys.exit("Atypical structure of the multiplier")

                z = expand(new_multiplier)
                n = len(z.args)
                terms_with_sqrt = 0
                terms_without_sqrt = 0

                if type(z) == sym.core.add.Add:
                    for m in range(n):
                        elementary_term = z.args[m]
                        if elementary_term.has(D) == False:
                            terms_without_sqrt += elementary_term
                        else:
                            terms_with_sqrt += elementary_term

                    simplified_terms_without_sqrt = simplify(simplify(terms_without_sqrt))

                    # for some reason it doesn't automatically rewrite it
                    if simplified_terms_without_sqrt.has(uo**2 + 2 * uo + 1):
                        simplified_terms_without_sqrt = simplified_terms_without_sqrt.subs(
                            uo**2 + 2 * uo + 1, (uo + 1) ** 2
                        )
                    elif simplified_terms_without_sqrt.has(-(uo**2) - 2 * uo - 1):
                        simplified_terms_without_sqrt = simplified_terms_without_sqrt.subs(
                            -(uo**2) - 2 * uo - 1, -((uo + 1) ** 2)
                        )

                    fully_simplified_multiplier_in_numerator = simplified_terms_without_sqrt + terms_with_sqrt

                # z can just be a numeric factor
                elif z.is_complex == True:
                    fully_simplified_multiplier_in_numerator = z
                else:
                    sys.exit("Unaccounted multiplier type")

                simplified_term_in_numerator *= fully_simplified_multiplier_in_numerator

            simplified_numerator += I_factor_in_numerator * simplified_term_in_numerator
    else:
        sys.exit("Atypical structure of the numerator")

    # handling the denominator
    number_of_multipliers_in_denominator = len(original_denominator_new.args)

    for j in range(number_of_multipliers_in_denominator):
        denominator_multiplier = original_denominator_new.args[j]
        z = expand(denominator_multiplier)
        re_z = re(z)
        im_z = im(z)

        if re_z != 0 and im_z == 0:
            if type(re_z) == sym.core.add.Add:
                n = len(re_z.args)
                terms_with_sqrt = 0
                terms_without_sqrt = 0
                for m in range(n):
                    if re_z.args[m].has(D) == False:
                        terms_without_sqrt += re_z.args[m]
                    else:
                        terms_with_sqrt += re_z.args[m]

                simplified_terms_without_sqrt = simplify(terms_without_sqrt)

                # for some reason it doesn't automatically rewrite it
                if simplified_terms_without_sqrt.has(uo**2 + 2 * uo + 1):
                    simplified_terms_without_sqrt = simplified_terms_without_sqrt.subs(
                        uo**2 + 2 * uo + 1, (uo + 1) ** 2
                    )
                elif simplified_terms_without_sqrt.has(-(uo**2) - 2 * uo - 1):
                    simplified_terms_without_sqrt = simplified_terms_without_sqrt.subs(
                        -(uo**2) - 2 * uo - 1, -((uo + 1) ** 2)
                    )

                fully_simplified_multiplier_in_denominator = simplified_terms_without_sqrt + terms_with_sqrt
            else:
                if re_z.has(D):
                    fully_simplified_multiplier_in_denominator = re_z
                else:
                    sys.exit("The denominator multiplier has non-zero real part contains unaccounted structures")

        elif re_z == 0 and im_z != 0:
            if type(im_z) == sym.core.add.Add:
                n = len(im_z.args)
                terms_with_sqrt = 0
                terms_without_sqrt = 0
                for m in range(n):
                    if im_z.args[m].has(D) == False:
                        terms_without_sqrt += im_z.args[m]
                    else:
                        terms_with_sqrt += im_z.args[m]

                simplified_terms_without_sqrt = simplify(terms_without_sqrt)

                # for some reason it doesn't automatically rewrite it
                if simplified_terms_without_sqrt.has(uo**2 + 2 * uo + 1):
                    simplified_terms_without_sqrt = simplified_terms_without_sqrt.subs(
                        uo**2 + 2 * uo + 1, (uo + 1) ** 2
                    )
                elif simplified_terms_without_sqrt.has(-(uo**2) - 2 * uo - 1):
                    simplified_terms_without_sqrt = simplified_terms_without_sqrt.subs(
                        -(uo**2) - 2 * uo - 1, -((uo + 1) ** 2)
                    )

                fully_simplified_multiplier_in_denominator = I * (simplified_terms_without_sqrt + terms_with_sqrt)
            else:
                if im_z.has(D):
                    fully_simplified_multiplier_in_denominator = I * im_z
                else:
                    sys.exit(
                        "The denominator multiplier has non-zero imaginary part and contains unaccounted structures"
                    )
        else:
            sys.exit("Multiplier has non-zero both real and imaginary parts")

        simplified_denominator *= fully_simplified_multiplier_in_denominator

    # # verification of procedure for self-consistency
    # # this procedure must be the identity transformation
    # delta1 = simplified_denominator - original_denominator_new
    # delta2 = simplified_numerator - original_numerator_new

    # if simplify(delta1) != 0 and simplify(delta2) != 0:
    #     sys.exit("Error in the simplification procedure")

    return simplified_numerator / simplified_denominator


def prefactor_simplification(new_prefactor_after_subs):
    """
    The function additionally splits the prefactor into momentum-dependent and momentum-independent parts.

    ARGUMENTS:

    new_prefactor_after_subs is given by reduction_to_common_denominator()

    OUTPUT DATA EXAMPLE:

    numeric_factor = 4*pi**2*A**3*nuo**5*(B/nuo)**(-2*d - 4*eps + 8)/B**10

    part_to_integrand = (-sc_prod(1, k) - sc_prod(1, q))*D_v(k)*D_v(q)*sc_prod(1, q)**3
    """
    list_with_prefactor_args = new_prefactor_after_subs.args
    numeric_factor = 1
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
                    numeric_factor *= term_arg
        else:
            numeric_factor *= term

    return part_to_integrand, numeric_factor
