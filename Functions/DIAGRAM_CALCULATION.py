from sympy import *
from typing import Any

from Functions.Data_classes import *
from Functions.computing_integrals_over_frequencies import *
from Functions.computing_tensor_part import *
from Functions.create_file_with_general_notation import *
from Functions.preparing_for_numerical_integration import residues_sum_in_Wolfram_Mathematica_format

# ------------------------------------------------------------------------------------------------------------------#
#                   Part 2. Diagram calculation (integrals over frequencies, tensor convolutions, etc.)
# ------------------------------------------------------------------------------------------------------------------#


def diagram_integrand_calculation(
    diagram_data: DiagramData, dimensional_factor_for_test: Any, output_in_WfMath_format: str
):
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
    Feynman_graph = open(f"Results/{diagram_data.output_file_name}", "a+")

    # starts filling the results of calculations (integrals over frequencies, tensor convolutions) to file
    Feynman_graph.write(f"\nDiagram integrand calculation start.\n")

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing integrals over frequencies
    # --------------------------------------------------------------------------------------------------------------#

    Feynman_graph.write(f"\nCalculation F. \n")

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

    print(f"\nReducing the obtained integrand scalar part to a common denominator.")

    # reduce to a common denominator the result obtained after calculating the integrals over frequencies
    diagram_expression = reduction_to_common_denominator(
        total_sum_of_residues_for_both_frequencies, diagram_data.expression_UV_convergence_criterion
    )

    Feynman_graph.write(
        f"\nThe expression for F after reduction to a common denominator: "
        f"\n{diagram_expression.common_factor * diagram_expression.residues_sum_without_common_factor} \n"
    )

    if diagram_data.expression_UV_convergence_criterion == True:
        print(f"\nSimplifying the expression of the obtained scalar part F.")

        # simplification of the expression for F (square roots cancellation somewhere, etc.)
        particular_integrand_simplification = partial_simplification_of_diagram_expression(
            diagram_expression.residues_sum_without_dim_factor_after_subs
        )

        # the common factor after the replacement k, q --> B*k/nuo, B*q/nuo is transformed into a part
        # that depends only on k and q and the dimension factor depends on |B|, nou, etc
        dim_and_dimless_factor = prefactor_simplification(diagram_expression.new_dim_factor_after_subs)

        # here we are testing that all diagrams must have the same dimension
        assert (
            dim_and_dimless_factor.dim_factor.has(dimensional_factor_for_test) == True
        ), "Diagram dimension do not match"

        # define F1 (see General_notation.txt) at the level of variables k and q
        integrand_scalar_part_depending_only_on_uo = (
            (dim_and_dimless_factor.dimensionless_factor * particular_integrand_simplification).doit().doit()
        )

        if integrand_scalar_part_depending_only_on_uo.has(go):
            sys.exit("Error when getting integrand for numerical integration")
        elif integrand_scalar_part_depending_only_on_uo.has(B):
            sys.exit("Error when getting integrand for numerical integration")
        elif integrand_scalar_part_depending_only_on_uo.has(nuo):
            sys.exit("Error when getting integrand for numerical integration")

        Feynman_graph.write(
            f"\nPreparation for numerical integration consists in carrying out a replacing of variables "
            f"k, q --> B*k/nuo, B*q/nuo, after which F is divided into a dimensional factor C_F and a function F1 "
            f"depending only on uo and integration variables: F = C_F*F1.\n"
        )

        Feynman_graph.write(
            f"\nThe expression for F1 after momentums replacing: "
            f"\n{integrand_scalar_part_depending_only_on_uo} \n"
            f"\nThe expression for C_F after momentums replacing: "
            f"\n{dim_and_dimless_factor.dim_factor} \n"
        )

    else:
        # TODO
        integrand_scalar_part_depending_only_on_uo = None

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing diagram tensor structures
    # --------------------------------------------------------------------------------------------------------------#

    Feynman_graph.write(f"\nCalculation T_ij. \n")

    print(f"\nComputing tensor convolutions. \n")

    Tensor = computing_tensor_structures(diagram_data)

    Feynman_graph.write(f"\nThe expression for T_ij after computing tensor convolutions: \n{Tensor} \n")

    Feynman_graph.write(
        f"\nPreparation for numerical integration consists in carrying out a replacing of variables "
        f"k, q --> B*k/nuo, B*q/nuo, after which T_ij is divided into a dimensional factor C_T and a function T1_ij "
        f"depending only on uo and integration variables: T_ij = C_T*T1_ij.\n"
    )

    separated_tensor_part = extract_B_and_nuo_depend_factor_from_tensor_part(Tensor)

    Feynman_graph.write(
        f"\nThe expression for T1_ij after momentums replacing: "
        f"\n{separated_tensor_part.dimensionless_factor} \n"
        f"\nThe expression for C_T after momentums replacing: "
        f"\n{separated_tensor_part.dim_factor} \n"
    )

    Feynman_graph.write(f"\nDiagram integrand calculation end.\n")

    # finish  filling the results of calculation to file
    Feynman_graph.close()

    diagram_integrand_data = IntegrandData(
        diagram_expression.common_factor * diagram_expression.residues_sum_without_common_factor,
        integrand_scalar_part_depending_only_on_uo,
        dim_and_dimless_factor.dim_factor,
        separated_tensor_part.dimensionless_factor,
        separated_tensor_part.dim_factor,
    )

    return diagram_integrand_data
