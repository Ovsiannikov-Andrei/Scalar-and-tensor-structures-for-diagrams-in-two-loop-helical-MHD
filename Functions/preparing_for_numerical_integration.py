from sympy import *

from Functions.SymPy_classes import *


def replace_function(big_string, replace_what, replace_with):
    """
    This function was created using the GPT4 neural network!
    """
    # Replace all instances of "replace_what(" with "replace_with["
    start_index = 0
    while start_index < len(big_string):
        index = big_string.find(f"{replace_what}(", start_index)
        if index == -1:
            break
        big_string = f"{big_string[:index]}{replace_with}[{big_string[index+len(replace_what)+1:]}"
        start_index = index + len(replace_with) + 2

    # Add a closing bracket for each "replace_with[" expression
    start_index = 0
    while start_index < len(big_string):
        index = big_string.find(f"{replace_with}[", start_index)
        if index == -1:
            break
        bracket_count = 1
        j = index + len(replace_with) + 1
        while bracket_count > 0 and j < len(big_string):
            if big_string[j] == "(":
                bracket_count += 1
            elif big_string[j] == ")":
                bracket_count -= 1
                if bracket_count == 0 and big_string[j] == ")":
                    big_string = f"{big_string[:j]}]{big_string[j+1:]}"
            j += 1

        start_index = index + len(replace_with) + 1

    return big_string


def transform_to_Wolfram_Mathematica_format(integrand_str):
    """
    The function translates the input data into the Wolfram Mathematica format.

    ARGUMENTS:

    input_string_expression is given by diagram_integrand_calculation(...)[1].subs(
        d, number).subs(eps, number).subs(A, number)


    OUTPUT DATA EXAMPLE:

    too long
    """
    integrand_str = str(integrand_str)

    replace_function(integrand_str, "sqrt", "Sqrt")

    integrand_str = integrand_str.replace("**", "^")

    integrand_str = integrand_str.replace("uo", "ToExpression[SubscriptBox[u, 0]]")

    cos_k_str = "Cos[ToExpression[SubscriptBox[\[Theta], 1]]]"
    cos_q_str = "Cos[ToExpression[SubscriptBox[\[Theta], 2]]]"
    cos_2u_str = "Cos[2*u]"
    sin_k_str = "Sin[ToExpression[SubscriptBox[\[Theta], 1]]]"
    sin_q_str = "Sin[ToExpression[SubscriptBox[\[Theta], 2]]]"

    integrand_str = integrand_str.replace("z_k", cos_k_str)

    integrand_str = integrand_str.replace("z_q", cos_q_str)

    integrand_str = integrand_str.replace("z", f"{sin_q_str}*{sin_k_str}*{cos_2u_str} + {cos_k_str}*{cos_q_str}")

    # print(pretty(Tenzor, use_unicode=False))

    # Fey_graphs.write("\n"+ pretty(Tenzor, use_unicode=False) + "\n")

    # Fey_graphs.write("\n"+ latex(Tenzor) + "\n")

    return integrand_str


def preparing_diagram_for_numerical_integration(
    output_file_name,
    Tensor,
    eps_default,
    d_default,
    A_MHD,
    list_with_uo_values,
    field_and_nuo_depend_factor,
    integrand_scalar_part_depending_only_on_uo,
):
    """
    The function finally prepares the integrand for numerical integration. Works only with converging diagrams.

    ARGUMENTS:

    output_file_name is given by get_info_about_diagram(),
    Tensor, field_and_nuo_depend_factor, and  integrand_scalar_part_depending_only_on_uo are given by
    diagram_integrand_calculation()

    eps_default -- value of the eps regularization parameter (default is assumed to be 0),
    d_default -- coordinate space dimension (assumed to be 3 by default)
    A_MHD -- value of the model parameter A (MHD corresponds to A = 1)
    list_with_uo_values -- the desired values of the unrenormalized magnetic Prandtl number uo
    introduced before the calculations

    OUTPUT DATA EXAMPLE:

    field_and_nuo_depend_factor = 4*I*pi**2*go**2*nuo**11*rho*lcs(s, 0, 9)*hyb(p, s)/B**10

    list_with_integrands = too long
    """

    tensor_part_numerator = fraction(Tensor)[0]
    tensor_part_denominator = fraction(Tensor)[1]
    scalar_part_numerator = fraction(integrand_scalar_part_depending_only_on_uo)[0]
    scalar_part_denominator = fraction(integrand_scalar_part_depending_only_on_uo)[1]

    tensor_part_numerator_args = tensor_part_numerator.args

    for i in range(len(tensor_part_numerator_args)):
        numerator_term = tensor_part_numerator_args[i]
        if numerator_term.has(I) or numerator_term.has(rho):
            field_and_nuo_depend_factor *= numerator_term
        elif numerator_term.has(hyb) or numerator_term.has(lcs):
            field_and_nuo_depend_factor *= numerator_term
        else:
            scalar_part_numerator *= numerator_term

    integrand_denominator = scalar_part_denominator * tensor_part_denominator
    integrand_numerator = scalar_part_numerator

    integrand = integrand_numerator / integrand_denominator

    integrand_for_numerical_calc = integrand.subs(d, d_default).subs(eps, eps_default).subs(A, A_MHD)

    field_and_nuo_depend_factor = field_and_nuo_depend_factor.subs(d, d_default).subs(eps, eps_default).subs(A, A_MHD)

    print(f"\nConverting the result to Wolfram Mathematica format.")

    WF_integrand_for_numerical_calc = transform_to_Wolfram_Mathematica_format(integrand_for_numerical_calc)

    Feynman_graph = open(f"Results/{output_file_name}", "a+")

    Feynman_graph.write(f"\nDefault parameter values: d = {d_default}, eps = {eps_default}, and A = {A_MHD}.\n")

    Feynman_graph.write(f"\nDiagram integrand with default parameters: \n{integrand_for_numerical_calc}\n")

    Feynman_graph.write(f"\nDiagram integrand in Wolfram Mathematica format: \n{WF_integrand_for_numerical_calc}\n")

    list_with_integrands = list()

    if list_with_uo_values[0] != "uo":

        Feynman_graph.write(f"\nThe entered values of the bare magnetic Prandtl number uo: {list_with_uo_values}")

        Feynman_graph.write(f"\nBelow are the integrands corresponding to the entered values of uo.\n")

        for i in range(len(list_with_uo_values)):
            uo_value = list_with_uo_values[i]
            list_with_integrands.append(integrand_for_numerical_calc.subs(uo, uo_value))
            integrand = list_with_integrands[i]
            integrand_in_Wolfram_format = transform_to_Wolfram_Mathematica_format(integrand)
            # Feynman_graph.write(f"\nDiagram integrand for uo = {uo_value}: \n{integrand} \n")
            Feynman_graph.write(
                f"\nDiagram integrand for uo = {uo_value} in Wolfram Mathematica format: "
                f"\n{integrand_in_Wolfram_format} \n"
            )

    Feynman_graph.close()

    return field_and_nuo_depend_factor, list_with_integrands
