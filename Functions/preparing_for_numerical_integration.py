import sys
from sympy import *

from Functions.Data_classes import *
from Functions.SymPy_classes import *


def replace_function(big_string: str, replace_what: str, replace_with: str):
    """
    This is a function that replaces expressions (math functions) with brackets ()
    with expressions with [] (the name of the function can also be changed).

    Note. This function was created using the GPT4 neural network!

    ARGUMENTS:

    big_string -- string containing functions with parentheses,
    replace_what -- what needs to be replaced,
    replace_with -- what should be replaced

    OUTPUT DATA EXAMPLE:

    replace_function(sqrt(), sqrt, Sqrt) = Sqrt[]
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


def residues_sum_in_Wolfram_Mathematica_format(residues_sum: str):
    """
    This function converts the sum of residues obtained from the calculation of integrals
    over frequencies into a format suitable for use in the Wolfram Mathematica system.

    ARGUMENTS:

    residues_sum is given by calculating_frequency_integrals_in_two_loop_diagrams()

    OUTPUT DATA EXAMPLE:

    too long
    """
    replacing_f_1 = replace_function(residues_sum, "f_1", "f_1")
    replacing_f_2 = replace_function(replacing_f_1, "f_2", "f_2")
    replacing_sc_prod = replace_function(replacing_f_2, "sc_prod", "sc_prod")
    replacing_D_v = replace_function(replacing_sc_prod, "D_v", "D_v")
    replacing_alpha = replace_function(replacing_D_v, "alpha", "alpha")
    replacing_beta = replace_function(replacing_alpha, "beta", "beta")
    replace_final = replacing_beta.replace("pi", "Pi")

    return replace_final


def nintegrand_to_WfMath_format(integrand_str: str):
    """
    The function transforms the input data into the Wolfram Mathematica format.

    ARGUMENTS:

    integrand_str

    OUTPUT DATA EXAMPLE:

    too long
    """

    replace_function(integrand_str, "sqrt", "Sqrt")
    integrand_str = integrand_str.replace("**", "^")

    # theta_k = ToExpression[SubscriptBox[\[Theta], 1]] in WfMath format
    # theta_q = ToExpression[SubscriptBox[\[Theta], 2]] in WfMath format
    cos_k_str = "Cos[theta_k]"
    cos_q_str = "Cos[theta_q]"
    cos_2u_str = "Cos[2*u]"
    sin_k_str = "Sin[theta_k]"
    sin_q_str = "Sin[theta_q]"

    integrand_str = integrand_str.replace("z_k", cos_k_str).replace("z_q", cos_q_str)
    integrand_str = integrand_str.replace("z", f"{sin_q_str}*{sin_k_str}*{cos_2u_str} + {cos_k_str}*{cos_q_str}")

    # print(pretty(Tenzor, use_unicode=False))
    # peint("\n"+ pretty(Tenzor, use_unicode=False) + "\n")
    # print("\n"+ latex(Tenzor) + "\n")

    return integrand_str


def preparing_diagram_for_numerical_integration(
    output_file_name: DiagramData,
    diagram_integrand_data: IntegrandData,
    eps_input: int,
    d_input: int,
    A_input: int,
    list_with_uo_values: list,
    output_in_WfMath_format: str,
    UV_convergence_criterion: bool,
):
    """
    The function finally prepares the integrand for numerical integration.
    Works only with converging diagrams.

    ARGUMENTS:

    output_file_name is given by get_info_about_diagram(),
    Tensor, field_and_nuo_depend_factor, and  integrand_scalar_part_depending_only_on_uo are given by
    diagram_integrand_calculation(), output_in_WfMath_format -- parameter for geting results in a
    format suitable for use in Wolfram Mathematica, UV_convergence_criterion is a parameter saying that
    the corresponding diagram (part of the diagram) is UV-finite

    eps_default -- value of the eps regularization parameter (default is assumed to be 0),
    d_default -- coordinate space dimension (assumed to be 3 by default)
    A_MHD -- value of the model parameter A (MHD corresponds to A = 1)
    list_with_uo_values -- the desired values of the unrenormalized magnetic Prandtl number uo
    introduced before the calculations

    OUTPUT DATA EXAMPLE:

    field_and_nuo_depend_factor = 4*I*pi**2*go**2*nuo**11*rho*lcs(s, 0, 9)*hyb(p, s)/B**10

    list_with_integrands = too long
    """
    if UV_convergence_criterion == True:
        dimless_tensor_part = diagram_integrand_data.tensor_convolution_dimensionless_part
        dim_factor_from_tensor_part = diagram_integrand_data.tensor_convolution_field_and_nuo_factor
        dim_factor_from_scalar_part = diagram_integrand_data.scalar_part_field_and_nuo_factor
        dimless_scalar_part = diagram_integrand_data.scalar_part_depending_only_on_uo

        tensor_part_numerator = fraction(dimless_tensor_part)[0]
        tensor_part_denominator = fraction(dimless_tensor_part)[1]
        scalar_part_numerator = fraction(dimless_scalar_part)[0]
        scalar_part_denominator = fraction(dimless_scalar_part)[1]

        dim_factor_from_scalar_part = dim_factor_from_scalar_part * dim_factor_from_tensor_part
        integrand_denominator = scalar_part_denominator * tensor_part_denominator
        integrand_numerator = scalar_part_numerator * tensor_part_numerator
        integrand = integrand_numerator / integrand_denominator

        Feynman_graph = open(f"Results/{output_file_name}", "a+")
        Feynman_graph.write(
            f"\nThe integrand of the diagram in a form suitable for numerical calculations, i.e. integrand = F1*T1_ij: \n{integrand}\n"
            f"\nDimensional multiplier before the integrand, i.e. C_int = C_F*C_T: \n{dim_factor_from_scalar_part}\n"
        )

        integrand_for_numerical_calc = integrand.subs(d, d_input).subs(eps, eps_input).subs(A, A_input)
        dim_factor_from_scalar_part = dim_factor_from_scalar_part.subs(d, d_input).subs(eps, eps_input).subs(A, A_input)

        Feynman_graph.write(
            f"\nInput parameter values: d = {d_input}, eps = {eps_input}, and A = {A_input}.\n"
            f"\nDiagram integrand for input parameters: \n{integrand_for_numerical_calc}\n"
        )

        if output_in_WfMath_format == "y":
            print(f"\nConverting the result to Wolfram Mathematica format.")

            WF_integrand_without_parameters = nintegrand_to_WfMath_format(str(integrand))
            WF_integrand_with_parameters = nintegrand_to_WfMath_format(str(integrand_for_numerical_calc))

            Feynman_graph.write(
                f"\nExpression for F1*T1_ij in Wolfram Mathematica format: \n{WF_integrand_without_parameters}\n"
                f"\nExpression for F1*T1_ij with input parameter values in Wolfram Mathematica format: \n{WF_integrand_with_parameters}\n"
            )

        if list_with_uo_values[0] != None:
            list_with_integrands = list()

            Feynman_graph.write(
                f"\nThe entered values of the bare magnetic Prandtl number uo: {list_with_uo_values}"
                f"\nBelow are the integrands corresponding to the entered values of uo.\n"
            )

            for i in range(len(list_with_uo_values)):
                uo_value = list_with_uo_values[i]
                list_with_integrands.append(integrand_for_numerical_calc.subs(uo, uo_value))
                integrand = list_with_integrands[i]
                Feynman_graph.write(f"\nDiagram integrand for uo = {uo_value}: \n{integrand} \n")
                if output_in_WfMath_format == "y":
                    integrand_in_Wolfram_format = nintegrand_to_WfMath_format(str(integrand))
                    Feynman_graph.write(
                        f"\nDiagram integrand for uo = {uo_value} in Wolfram Mathematica format: "
                        f"\n{integrand_in_Wolfram_format} \n"
                    )

        Feynman_graph.close()
