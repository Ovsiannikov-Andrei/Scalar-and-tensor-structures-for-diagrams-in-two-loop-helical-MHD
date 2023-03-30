from sympy import *

from Functions.computing_integrals_over_frequencies import *
from Functions.computing_tensor_part import *
from Functions.create_file_with_general_notation import *

# ------------------------------------------------------------------------------------------------------------------#
#                   Part 2. Diagram calculation (integrals over frequencies, tensor convolutions, etc.)
# ------------------------------------------------------------------------------------------------------------------#


def diagram_integrand_calculation(
    output_file_name,
    moznost,
    indexb,
    indexB,
    P_structure,
    H_structure,
    kd_structure,
    hyb_structure,
    Tenzor,
    Product,
    is_diagram_convergent,
):
    """
    This function calculates the integrand of the corresponding diagram in terms of tensor and scalar parts.

    ARGUMENTS:

    output_file_name
    moznost
    indexb, indexB
    P_structure
    H_structure
    kd_structure
    hyb_structure
    Tenzor
    Product
    is_diagram_convergent

    OUTPUT DATA EXAMPLE:
    """

    Feynman_graph = open(f"Results/{output_file_name}", "a+")
    # start filling the results of calculation to file

    Feynman_graph.write(f"\nDiagram integrand calculation begin:\n")
    # starts filling the results of calculations (integrals over frequencies, tensor convolutions) to file

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing integrals over frequencies
    # --------------------------------------------------------------------------------------------------------------#

    print(f"\nComputing integrals over frequencies.")

    total_sum_of_residues_for_both_frequencies = calculating_frequency_integrals_in_two_loop_diagrams(Product, w_k, w_q)
    # calculate integrals over frequencies using the residue theorem

    Feynman_graph.write(
        f"\nThe scalar part of the given diagram integrand (after computing integrals over frequencies): "
        f"\n{total_sum_of_residues_for_both_frequencies} \n"
    )

    print(f"\nReducing the obtained integrand scalar part to a common denominator.")

    diagram_expression = reduction_to_common_denominator(
        total_sum_of_residues_for_both_frequencies, is_diagram_convergent
    )

    diagram_expression_without_prefactor_and_substitution = diagram_expression[0][0]
    diagram_prefactor_without_substitution = diagram_expression[0][1]
    diagram_expression_without_prefactor_and_after_substitution = diagram_expression[1][0]
    diagram_prefactor_after_substitution = diagram_expression[1][1]

    Feynman_graph.write(
        f"\nPrefactor (part of the integrand scalar part numerator, initially frequency-independent): "
        f"\n{diagram_prefactor_without_substitution} \n"
    )

    diagram_sc_part_without_mom_repl = (
        diagram_prefactor_without_substitution * diagram_expression_without_prefactor_and_substitution
    )

    Feynman_graph.write(
        f"\nThe scalar part of the given diagram integrand (with prefactor) after reduction to a common denominator: "
        f"\n{diagram_sc_part_without_mom_repl} \n"
    )

    if is_diagram_convergent == True:
        print(f"\nSimplifying the obtained scalar part expression.")

        partial_simplification_of_diagram_scalar_part = partial_simplification_of_diagram_expression(
            diagram_expression_without_prefactor_and_after_substitution
        )

        additional_prefactor_simplification = prefactor_simplification(diagram_prefactor_after_substitution)

        part_to_integrand = additional_prefactor_simplification[0]

        field_and_nuo_depend_factor = additional_prefactor_simplification[1]

        Feynman_graph.write(
            f"\nPartially simplified expression for a integrand scalar part without prefactor "
            f"after momentums replacement k, q --> B*k/nuo, B*q/nuo:"
            f"\n{part_to_integrand*partial_simplification_of_diagram_scalar_part} \n"
        )

        assert (
            field_and_nuo_depend_factor.has(go**2 * nuo**11 * (B / nuo) ** (-2 * d - 4 * eps + 8) / B**10) == True
        ), "Diagram dimension do not match"

        Feynman_graph.write(
            f"\nNumerical factor depending on the absolute value of the field and viscosity "
            f"(after replacing of momentums, all entire dependence on |B| and nuo of the diagram is concentrated here)"
            f"\n{field_and_nuo_depend_factor} \n"
        )

        integrand_scalar_part_depending_only_on_uo = (
            (part_to_integrand * partial_simplification_of_diagram_scalar_part).doit().doit()
        )

        if integrand_scalar_part_depending_only_on_uo.has(go):
            sys.exit("Error when getting integrand for numerical integration")
        elif integrand_scalar_part_depending_only_on_uo.has(B):
            sys.exit("Error when getting integrand for numerical integration")
        elif integrand_scalar_part_depending_only_on_uo.has(nuo):
            sys.exit("Error when getting integrand for numerical integration")

        Feynman_graph.write(
            f"\nCompletely simplified expression for a integrand scalar part without prefactor "
            f"after momentums replacement k, q --> B*k/nuo, B*q/nuo:"
            f"\n{integrand_scalar_part_depending_only_on_uo} \n"
        )

    else:
        integrand_scalar_part_depending_only_on_uo = None
        field_and_nuo_depend_factor = None

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing diagram tensor structure
    # --------------------------------------------------------------------------------------------------------------#

    print(f"\nBeginning the tensor convolutions calculation: \n")

    Tensor = computing_tensor_structures(
        moznost, indexb, indexB, P_structure, H_structure, kd_structure, hyb_structure, Tenzor
    )

    Feynman_graph.write(f"\nDiagram tensor structure after computing tensor convolutions: \n{Tensor} \n")

    # finish  filling the results of calculation to file
    Feynman_graph.write(f"\nDiagram integrand calculation end.\n")

    Feynman_graph.close()

    return [
        diagram_sc_part_without_mom_repl,
        integrand_scalar_part_depending_only_on_uo,
        field_and_nuo_depend_factor,
        Tensor,
    ]
