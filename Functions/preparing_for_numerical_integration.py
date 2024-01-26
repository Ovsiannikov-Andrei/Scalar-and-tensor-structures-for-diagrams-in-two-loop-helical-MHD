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


def nintegrand_to_WfMath_format(integrand_str: str, replace_z: bool):
    """
    The function transforms the input data into the Wolfram Mathematica format.

    ARGUMENTS:

    integrand_str

    replace_z

    OUTPUT DATA EXAMPLE:

    too long
    """

    integrand_str = replace_function(integrand_str, "sqrt", "Sqrt")
    integrand_str = replace_function(integrand_str, "Abs", "RealAbs")
    integrand_str = integrand_str.replace("**", "^")

    # theta_k = ToExpression[SubscriptBox[\[Theta], 1]] in WfMath format
    # theta_q = ToExpression[SubscriptBox[\[Theta], 2]] in WfMath format
    cos_k_str = "Cos[thetak]"
    cos_q_str = "Cos[thetaq]"
    cos_2phi_str = "Cos[2*phi]"
    sin_k_str = "Sin[thetak]"
    sin_q_str = "Sin[thetaq]"

    integrand_str = integrand_str.replace("z_k", cos_k_str).replace("z_q", cos_q_str)

    if replace_z == True:
        integrand_str = integrand_str.replace(
            "z", f"({sin_q_str}*{sin_k_str}*{cos_2phi_str} + {cos_k_str}*{cos_q_str})"
        )

    # print(pretty(Tenzor, use_unicode=False))
    # peint("\n"+ pretty(Tenzor, use_unicode=False) + "\n")
    # print("\n"+ latex(Tenzor) + "\n")

    return integrand_str


def preparing_diagram_for_numerical_integration(
    diagram_data: DiagramData,
    diagram_integrand_data: IntegrandData,
    eps_input: int,
    d_input: int,
    A_input: int,
    uo_default: float,
    list_with_uo_values: list,
    output_in_WfMath_format: str,
):
    """
    The function finally prepares the integrand for numerical integration.

    ARGUMENTS:

    output_file_name and symmetry_multiplier are given by get_info_about_diagram(),
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

    field_and_nuo_depend_factor = 4*I*pi**2*go**2*nuo**11*rho*lcs(s, 0, 9)*mom(p, s)/B**10

    list_with_integrands = too long
    """
    if diagram_integrand_data.tensor_structure_done == True:
        output_file_name = diagram_data.output_file_name
        symmetry_multiplier = diagram_data.symmetry_factor
        UV_convergence_criterion = diagram_data.expression_UV_convergence_criterion

        if diagram_data.nickel_topology == "e12_e3_33_":
            Feynman_graph = open(f"Details about the diagrams/Double loops/{output_file_name}", "a+")
        elif diagram_data.nickel_topology == "e12_23_3_e":
            Feynman_graph = open(f"Details about the diagrams/Cross loops/{output_file_name}", "a+")
        else:
            Feynman_graph = open(f"Details about the diagrams/{output_file_name}", "a+")

        Feynman_graph.write(f"\nCalculation of the final expression for the integrand. \n")

        if UV_convergence_criterion == True:
            Final_integrand_for_numeric_calc = open(f"Final Results/UV-finite diagrams/{output_file_name}", "w")

            momentum_depend_tensor_part_B = diagram_integrand_data.tensor_convolution_B_momentum_depend_part
            momentum_depend_scalar_part_B = diagram_integrand_data.convergent_scalar_part_depending_only_on_uo

            common_multiplier_from_tensor_part_B = diagram_integrand_data.tensor_convolution_B_field_and_nuo_factor
            common_multiplier_from_scalar_part_B = diagram_integrand_data.scalar_part_field_and_nuo_factor_B

            complete_common_factor = simplify(
                common_multiplier_from_scalar_part_B * common_multiplier_from_tensor_part_B * symmetry_multiplier
            )
            convergent_integrand = momentum_depend_tensor_part_B * momentum_depend_scalar_part_B

            WF_complete_common_factor = (
                replace_function(replace_function(str(complete_common_factor), "lcs", "lcs"), "mom", "mom")
                .replace("**", "^")
                .replace("pi", "Pi")
            )

            Feynman_graph.write(f"\nF = C_int*F1*T1_ij, where C_int = C_F*C_T.")

            Feynman_graph.write(
                f"\nDimensional multiplier before the integrand (with symmetric coefficient), i.e. C_int: \n{WF_complete_common_factor}\n"
                f"\nThe UV-convergent part of the integrand without C_int, i.e. F1*T1_ij: \n{convergent_integrand}\n"
            )

            Final_integrand_for_numeric_calc.write(
                f"Nickel index of the Feynman diagram: \n{diagram_data.nickel_index} \n"
                f"The UV-convergent part of the integrand without C_int: \n{nintegrand_to_WfMath_format(str(convergent_integrand), True)}\n"
                f"Dimensional multiplier before the integrand (with symmetric coefficient), i.e. C_int: \n{complete_common_factor}\n"
            )

            if output_in_WfMath_format == "y":
                print(f"\nConverting the result to Wolfram Mathematica format.")

                WF_integrand_without_parameters = nintegrand_to_WfMath_format(str(convergent_integrand), True)

                Feynman_graph.write(
                    f"\nExpression for F1*T1_ij in Wolfram Mathematica format: \n{WF_integrand_without_parameters}\n"
                )

            integrand_for_numerical_calc = convergent_integrand.subs(d, d_input).subs(eps, eps_input).subs(A, A_input)

            list_with_integrands = list()

            Feynman_graph.write(
                f"\nOne-loop reciprocal magnetic Prandtl number at the fixed point {uo} = {uo_default}."
                f"\nThe entered values of the bare magnetic Prandtl number {uo}: {list_with_uo_values}"
                f"\nBelow are the integrands corresponding to the entered values of {uo}.\n"
            )

            for i in range(len(list_with_uo_values)):
                uo_value = list_with_uo_values[i]
                list_with_integrands.append(integrand_for_numerical_calc.subs(uo, uo_value))
                integrand = list_with_integrands[i]
                Feynman_graph.write(f"\nDiagram integrand for {uo} = {uo_value}: \n{integrand} \n")
                if output_in_WfMath_format == "y":
                    integrand_in_Wolfram_format = nintegrand_to_WfMath_format(str(integrand), True)
                    Feynman_graph.write(
                        f"\nDiagram integrand for {uo} = {uo_value} in Wolfram Mathematica format: "
                        f"\n{integrand_in_Wolfram_format} \n"
                    )

        else:
            Final_integrand_for_numeric_calc = open(f"Final Results/UV-infinite diagrams/{output_file_name}", "w")

            momentum_depend_tensor_part_lambda = diagram_integrand_data.tensor_convolution_lambda_momentum_depend_part
            momentum_depend_scalar_part_lambda = diagram_integrand_data.divergent_scalar_part_depending_only_on_uo
            momentum_depend_tensor_part_B = diagram_integrand_data.tensor_convolution_B_momentum_depend_part
            momentum_depend_scalar_part_B = diagram_integrand_data.convergent_scalar_part_depending_only_on_uo

            common_multiplier_from_tensor_part_lambda = (
                diagram_integrand_data.tensor_convolution_lambda_field_and_nuo_factor
            )
            common_multiplier_from_tensor_part_B = diagram_integrand_data.tensor_convolution_B_field_and_nuo_factor
            common_multiplier_from_scalar_part_lambda = diagram_integrand_data.scalar_part_field_and_nuo_factor_lambda
            common_multiplier_from_scalar_part_B = diagram_integrand_data.scalar_part_field_and_nuo_factor_B

            complete_common_factor_lambda = simplify(
                common_multiplier_from_scalar_part_lambda
                * common_multiplier_from_tensor_part_lambda
                * symmetry_multiplier
            )
            complete_common_factor_B = simplify(
                common_multiplier_from_scalar_part_B * common_multiplier_from_tensor_part_B * symmetry_multiplier
            )

            convergent_integrand = momentum_depend_tensor_part_B * momentum_depend_scalar_part_B
            divergent_integrand = momentum_depend_tensor_part_lambda * momentum_depend_scalar_part_lambda

            # dimension_test = simplify(complete_common_factor_lambda - complete_common_factor_B)

            # assert dimension_test == 0, "Incorrect dimension of the integrand."

            # for divergent parts of diagrams, the only difference is that the replacement should be done not through the B field,
            # but through the cutoff parameter

            WF_complete_common_factor_lambda = (
                replace_function(replace_function(str(complete_common_factor_lambda), "lcs", "lcs"), "mom", "mom")
                .replace("**", "^")
                .replace("pi", "Pi")
            )
            WF_complete_common_factor_B = (
                replace_function(replace_function(str(complete_common_factor_B), "lcs", "lcs"), "mom", "mom")
                .replace("**", "^")
                .replace("pi", "Pi")
            )

            Feynman_graph.write(
                f"\nF ==> C_int_lambda*F1(B = 0)*T1_lambda_ij + C_int_B*(F1 - F1(B = 0))*T1_B_ij, \n"
                f"where C_int_lambda = C_F_lambda*C_lambda_T and C_int_B = C_F_B*C_B_T.\n"
            )

            Feynman_graph.write(
                f"\nDimensional multiplier before UV-divergent part of the integrand (with symmetric coefficient), i.e. C_int_lambda: \n{complete_common_factor_lambda}\n"
                f"\nDimensional multiplier before UV-convergent part of the integrand (with symmetric coefficient), i.e. C_int_B: \n{complete_common_factor_B}\n"
                f"\nThe UV-divergent part of the integrand without C_int_lambda, i.e. F1(B = 0)*T1_lambda_ij: \n{divergent_integrand}\n"
                f"\nThe UV-convergent part of the integrand without C_int_B, i.e. (F1 - F1(B = 0))*T1_B_ij: \n{convergent_integrand}\n"
            )

            Final_integrand_for_numeric_calc.write(
                f"Nickel index of the Feynman diagram: \n{diagram_data.nickel_index} \n"
                f"The UV-divergent part of the integrand without C_int_lambda: \n{nintegrand_to_WfMath_format(str(divergent_integrand), False)}\n"
                f"The UV-convergent part of the integrand without C_int_B: \n{nintegrand_to_WfMath_format(str(convergent_integrand), True)}\n"
                f"Dimensional multiplier before UV-divergent part of the integrand (with symmetric coefficient), i.e. C_int_lambda: \n{WF_complete_common_factor_lambda}\n"
                f"Dimensional multiplier before UV-convergent part of the integrand (with symmetric coefficient), i.e. C_int_B: \n{WF_complete_common_factor_B}\n"
            )

            if output_in_WfMath_format == "y":
                print(f"\nConverting the result to Wolfram Mathematica format.")

                WF_divergent_integrand_without_parameters = nintegrand_to_WfMath_format(str(divergent_integrand), False)
                WF_convergent_integrand_without_parameters = nintegrand_to_WfMath_format(
                    str(convergent_integrand), True
                )

                Feynman_graph.write(
                    f"\nExpression for F1(B = 0)*T1_lambda_ij in Wolfram Mathematica format: \n{WF_divergent_integrand_without_parameters}\n"
                    f"\nExpression for (F1 - F1(B = 0))*T1_B_ij with input parameter values in Wolfram Mathematica format: \n{WF_convergent_integrand_without_parameters}\n"
                )

            divergent_integrand_for_numerical_calc = (
                divergent_integrand.subs(d, d_input).subs(eps, eps_input).subs(A, A_input)
            )
            convergent_integrand_for_numerical_calc = (
                convergent_integrand.subs(d, d_input).subs(eps, eps_input).subs(A, A_input)
            )

            list_with_divergent_integrands = list()
            list_with_convergent_integrands = list()

            Feynman_graph.write(
                f"\nOne-loop reciprocal magnetic Prandtl number at the fixed point {uo} = {uo_default}."
                f"\nThe entered values of the bare magnetic Prandtl number {uo}: {list_with_uo_values}."
                f"\nBelow are the integrands corresponding to the entered values of {uo}.\n"
            )

            for i in range(len(list_with_uo_values)):
                uo_value = list_with_uo_values[i]
                list_with_divergent_integrands.append(divergent_integrand_for_numerical_calc.subs(uo, uo_value))
                list_with_convergent_integrands.append(convergent_integrand_for_numerical_calc.subs(uo, uo_value))
                divergent_integrand = list_with_divergent_integrands[i]
                convergent_integrand = list_with_convergent_integrands[i]
                Feynman_graph.write(
                    f"\nDiagram integrand's divergent part for {uo} = {uo_value}: \n{divergent_integrand} \n"
                )
                Feynman_graph.write(
                    f"\nDiagram integrand's convergent part for {uo} = {uo_value}: \n{convergent_integrand} \n"
                )
                if output_in_WfMath_format == "y":
                    divergent_integrand_in_Wolfram_format = nintegrand_to_WfMath_format(str(divergent_integrand), False)
                    convergent_integrand_in_Wolfram_format = nintegrand_to_WfMath_format(
                        str(convergent_integrand), True
                    )
                    Feynman_graph.write(
                        f"\nDiagram integrand's divergent part for {uo} = {uo_value} in Wolfram Mathematica format: "
                        f"\n{divergent_integrand_in_Wolfram_format} \n"
                        f"\nDiagram integrand's convergent part for {uo} = {uo_value} in Wolfram Mathematica format: "
                        f"\n{convergent_integrand_in_Wolfram_format} \n"
                    )

        Feynman_graph.close()
        Final_integrand_for_numeric_calc.close()
