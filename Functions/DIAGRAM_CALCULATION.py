from sympy import *
from typing import Any

from Functions.Data_classes import *
from Functions.computing_integrals_over_frequencies import *
from Functions.computing_tensor_part import *
from Functions.create_file_with_general_notation import *
from Functions.preparing_for_numerical_integration import residues_sum_in_Wolfram_Mathematica_format
from Functions.test_functions_for_UV_divergent_parts import *

# ------------------------------------------------------------------------------------------------------------------#
#                   Part 2. Diagram calculation (integrals over frequencies, tensor convolutions, etc.)
# ------------------------------------------------------------------------------------------------------------------#


def diagram_integrand_calculation(diagram_data: DiagramData, output_in_WfMath_format: str):
    """
    This function calculates the integrand of the corresponding diagram in terms of tensor and scalar parts.

    ARGUMENTS:

    diagram_data is given by get_info_about_diagram(),
    dimensional_factor_for_test -- diagram dimension for tests,
    output_in_WfMath_format -- parameter for geting results in a
    format suitable for use in Wolfram Mathematica.

    OUTPUT DATA EXAMPLE:

    too long
    """

    # start filling the results of calculation to file
    Feynman_graph = open(f"Details about the diagrams/{diagram_data.output_file_name}", "a+")

    # starts filling the results of calculations (integrals over frequencies, tensor convolutions) to file
    Feynman_graph.write(f"\nDiagram integrand calculation start.\n")

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing integrals over frequencies
    # --------------------------------------------------------------------------------------------------------------#

    Feynman_graph.write(f"\nCalculation of F. \n")

    print(f"\nComputing integrals over frequencies.")

    # calculate integrals over frequencies using the residue theorem
    total_sum_of_residues_for_both_frequencies = calculating_frequency_integrals_in_two_loop_diagrams(
        diagram_data.integrand_scalar_part, w_k, w_q
    )

    Feynman_graph.write(
        f"\nThe expression for F after integration over frequencies: "
        f"\n{total_sum_of_residues_for_both_frequencies} \n"
    )

    if output_in_WfMath_format == "y":
        residues_sum_in_WfMath_format = residues_sum_in_Wolfram_Mathematica_format(
            str(total_sum_of_residues_for_both_frequencies)
        )

        Feynman_graph.write(
            f"\nThe previous expression in a Wolfram Mathematica-friendly format: "
            f"\n{residues_sum_in_WfMath_format} \n"
        )

    print(f"\nReducing the all obtained integrand scalar part to a common denominator.")

    # reduce to a common denominator the result obtained after calculating the integrals over frequencies
    diagram_expression = reduction_to_common_denominator(total_sum_of_residues_for_both_frequencies)

    current_integrand = diagram_expression.common_factor * diagram_expression.residues_sum_without_common_factor

    if diagram_data.expression_UV_convergence_criterion == False:
        compare_answers_test = compare_UV_divergent_parts(diagram_data.nickel_index, current_integrand)
        assert compare_answers_test, """The answer for the divergent part of the diagram does not match the result 
obtained by direct integration in Wolfram Mathematica."""

    Feynman_graph.write(f"\nThe expression for F after reduction to a common denominator: \n{current_integrand} \n")

    # the common factor after the replacement k, q --> B*k/nuo, B*q/nuo is transformed into a part
    # that depends only on k and q and the dimension factor depends on |B|, nuo, etc
    common_factor = prefactor_simplification(diagram_expression.new_dim_factor_after_subs)

    # dimension factor before the diagram to check the result
    # (all diagrams must have the same dimension)
    scalar_part_dimensional_factor_for_test = go**2 * nuo**11 * (B / nuo) ** (-2 * d - 4 * eps + 8) / B**10

    # here we are testing that all diagrams must have the same dimension
    assert (
        common_factor.momentum_independ_factor.has(scalar_part_dimensional_factor_for_test) == True
    ), "Incorrect dimension of the UV-finite scalar part of the diagram."

    print(f"\nSimplifying the expression of the obtained scalar part F.")

    # simplification of the expression for F (square roots cancellation somewhere, etc.)
    particular_integrand_simplification = partial_simplification_of_diagram_expression(
        diagram_expression.residues_sum_without_dim_factor_after_subs
    )

    Feynman_graph.write(
        f"\nPreparation for numerical integration for the integrand's scalar part consists in carrying out a "
        f"replacing of variables {k}, {q} --> {B*k/nuo}, {B*q/nuo}, after which UV-convergent part of the "
        f"diagram is divided into a dimensional factor C_F and a function F1 depending only on {uo} and "
        f"integration variables {k} and {q}: F = C_F*F1.\n"
        f"\n1. In case of UV-convergent diagrams UV-convergent part is equal to F."
        f"\n1. In case of UV-divergent diagrams UV-convergent part is equal to F - F({B} = 0) [1].\n"
    )

    # preparing the UV-finine diagrams for numerical integration
    if diagram_data.expression_UV_convergence_criterion == True:
        UV_divergent_part_at_zero_B = 0

        integrand_scalar_part_depending_only_on_uo = particular_integrand_simplification.doit().doit().subs(b, 1).doit()

        # define F1 (see General_notation.txt) at the level of variables k and q
        integrand_scalar_part_depending_only_on_uo_and_eps = (
            common_factor.momentum_depend_factor.doit().doit().subs(b, 1) * integrand_scalar_part_depending_only_on_uo
        )

        assert (
            integrand_scalar_part_depending_only_on_uo_and_eps.has(go) == False,
            integrand_scalar_part_depending_only_on_uo_and_eps.has(nuo) == False,
            integrand_scalar_part_depending_only_on_uo_and_eps.has(B) == False,
        ), "Error when getting integrand scalar part."

        Feynman_graph.write(
            f"\nThe expression for F1 after momentums replacing: "
            f"\n{integrand_scalar_part_depending_only_on_uo_and_eps} \n"
            f"\nThe expression for common C_F after momentums replacing: "
            f"\n{common_factor.momentum_independ_factor} \n"
        )
    # preparing the diagrams containind UV-infinine parts for numerical integration
    else:
        # using the auxiliary parameter b we set the field B here to 0
        UV_divergent_part_at_zero_B = particular_integrand_simplification.subs(b, 0).doit().doit().doit()

        # according to [1], all corrections to all propagators in their expansion in B are UV-finite.
        # Accordingly, the divergent part of the diagram is concentrated in the function F(B = 0)
        UV_convergent_part = (
            together(particular_integrand_simplification.subs(b, 1) - particular_integrand_simplification.subs(b, 0))
            .doit()
            .doit()
            .doit()
            .subs(b, 1)
        )

        # define F1 (see General_notation.txt) at the level of variables k and q
        integrand_scalar_part_depending_only_on_uo_and_eps = (
            common_factor.momentum_depend_factor.doit().doit() * UV_convergent_part
        )

        assert (
            integrand_scalar_part_depending_only_on_uo_and_eps.has(go) == False,
            integrand_scalar_part_depending_only_on_uo_and_eps.has(nuo) == False,
            integrand_scalar_part_depending_only_on_uo_and_eps.has(B) == False,
        ), "Error when getting integrand scalar part."

        # by dimension B ~ nuo* Cutoff
        scalar_common_factor_lambda = common_factor.momentum_independ_factor.subs(B, Cutoff * nuo)
        scalar_common_factor_B = common_factor.momentum_independ_factor

        Feynman_graph.write(
            f"\nThe expression for F1 after momentums replacing: F1 ==> C_F_lambda*F1(B = 0) + C_F_B*(F1 - F1(B = 0))."
            f"\nThe expression for F1 - F1(B = 0) (UV-convergent part):"
            f"\n{integrand_scalar_part_depending_only_on_uo_and_eps} \n"
            f"\nThe expression for F1(B = 0) (UV-divergent part):"
            f"\n{common_factor.momentum_depend_factor * UV_divergent_part_at_zero_B}\n"
            f"\nThe expression for common C_F_lambda after momentums replacing: "
            f"\n{scalar_common_factor_lambda} \n"
            f"\nThe expression for common C_F_B after momentums replacing: "
            f"\n{scalar_common_factor_B} \n"
        )

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing diagram tensor structures
    # --------------------------------------------------------------------------------------------------------------#

    Feynman_graph.write(f"\nCalculation of T_ij. \n")

    print(f"\nComputing tensor convolutions. \n")

    tensor_data = computing_tensor_structures(diagram_data, diagram_data.expression_UV_convergence_criterion)

    Tensor_Lambda = tensor_data.lambda_proportional_term
    Tensor_B = tensor_data.B_proportional_term

    print(f"\nDiagram tensor structure after computing tensor convolutions: ")
    print(f"\n1. Tensor structure without the presence of {B} field (Lambda part): \n{Tensor_Lambda}")
    print(f"\n2. Tensor structure with the presence of {B} field (B part): \n{Tensor_B}\n")

    Feynman_graph.write(
        f"\nFor UV-divergent diagrams, the expression for T_ij after calculating tensor convolutions "
        f"is calculated both in the case of the presence of an external field B and for {B} = 0. "
        f"For UV-finite diagrams, there is an expression for T_ij only in the external field. \n"
        f"\n1. Tensor structure without the presence of B field (Lambda part): \n{Tensor_Lambda} \n"
        f"\n2. Tensor structure with the presence of B field (B part): \n{Tensor_B}\n"
    )

    if diagram_data.expression_UV_convergence_criterion == True:
        Tensor_Lambda_at_zero_A = None
        separated_tensor_Lambda_part = None
    else:
        Tensor_Lambda_at_zero_A = Tensor_Lambda.subs(A, 1)
        separated_tensor_Lambda_part = extract_B_and_nuo_depend_factor_from_tensor_part(Tensor_Lambda)

    Tensor_B_at_zero_A = Tensor_B.subs(A, 1)
    separated_tensor_B_part = extract_B_and_nuo_depend_factor_from_tensor_part(Tensor_B)

    Feynman_graph.write(
        f"\nThe expression for T_ij in helical MHD ({A} = 1): \n"
        f"\n1. Tensor structure without the presence of B field (Lambda part): \n{Tensor_Lambda_at_zero_A} \n"
        f"\n2. Tensor structure with the presence of B field (B part): \n{Tensor_B_at_zero_A}\n"
    )
    Feynman_graph.write(
        f"\nPreparation for numerical integration for the integrand's tensor part consists in carrying out a "
        f"replacing of variables {k}, {q} --> {B*k/nuo}, {B*q/nuo}, after which T_ij is divided into a dimensional "
        f"factor C_T and a function T1_ij depending only on uo and integration variables: T_ij = C_T*T1_ij. "
        f"\nFor UV divergent diagrams, this procedure applies to both Lambda part of T_ij and to B part of T_ij.\n"
    )

    # in terms of dimension, the tensor part is proportional to the 4th degree of momentum
    # (only the vertex factors contribute), one of which is external and does not participate
    # in the procedure of replacing of variables
    tensor_part_dimensional_factor_for_test = B**3 / nuo**3

    # here we are testing that all diagrams must have the same dimension
    if diagram_data.expression_UV_convergence_criterion == False:
        assert (
            separated_tensor_Lambda_part.momentum_independ_factor.has(tensor_part_dimensional_factor_for_test) == True
        ), "Incorrect dimension of the tensor part of the diagram."

    assert (
        separated_tensor_B_part.momentum_independ_factor.has(tensor_part_dimensional_factor_for_test) == True
    ), "Incorrect dimension of the tensor part of the diagram."

    if diagram_data.expression_UV_convergence_criterion == False:
        Lambda_part_momentum_depend_factor = separated_tensor_Lambda_part.momentum_depend_factor
        # by dimension B ~ nuo* Cutoff
        Lambda_part_momentum_independ_factor = separated_tensor_Lambda_part.momentum_independ_factor.subs(
            B, Cutoff * nuo
        )
        Feynman_graph.write(
            f"\nThe expression for T1_lambda_ij (Lambda part of T_1_ij) after momentums replacing: "
            f"\n{Lambda_part_momentum_depend_factor} \n"
            f"\nThe expression for C_lambda_T after momentums replacing: "
            f"\n{Lambda_part_momentum_independ_factor} \n"
        )
    else:
        Lambda_part_momentum_depend_factor = None
        Lambda_part_momentum_independ_factor = None

    Feynman_graph.write(
        f"\nThe expression for T1_B_ij (B part of T_1_ij) after momentums replacing: "
        f"\n{separated_tensor_B_part.momentum_depend_factor} \n"
        f"\nThe expression for C_B_T after momentums replacing: "
        f"\n{separated_tensor_B_part.momentum_independ_factor} \n"
    )

    Feynman_graph.write(f"\nDiagram integrand calculation end.\n")

    # finish  filling the results of calculation to file
    Feynman_graph.close()

    diagram_integrand_data = IntegrandData(
        diagram_expression.common_factor * diagram_expression.residues_sum_without_common_factor,
        integrand_scalar_part_depending_only_on_uo_and_eps,
        common_factor.momentum_depend_factor * UV_divergent_part_at_zero_B,
        scalar_common_factor_lambda,
        scalar_common_factor_B,
        Lambda_part_momentum_depend_factor,
        Lambda_part_momentum_independ_factor,
        separated_tensor_B_part.momentum_depend_factor,
        separated_tensor_B_part.momentum_independ_factor,
    )

    return diagram_integrand_data
