from Functions.Data_classes import *
from Functions.create_file_with_output import *
from Functions.get_diagram_loop_structure import *
from Functions.get_momentum_frequency_distribution import *
from Functions.get_diagram_integrand import *

# ------------------------------------------------------------------------------------------------------------------#
#                                 Part 1: Diagram description (lines, vertices, etc.)
# ------------------------------------------------------------------------------------------------------------------#


def get_info_about_diagram(Nickel_index: str, output_in_WfMath_format: str, counter: int):
    """
    This function returns all information about the structure of a particular diagram.

    ARGUMENTS:

    Nickel_index is given by get_information_from_Nickel_index(),
    output_in_WfMath_format -- parameter for geting results in a
    format suitable for use in Wolfram Mathematica,
    counter -- some counter of the number of computed diagrams

    OUTPUT DATA EXAMPLE:

    output_file_name = 1. Diagram__e12_23_3_e__0B_bB_vv__vB_bb__bV__0b_.txt

    momentums_at_vertices = [[-1, 'B', p], [0, 'b', k], [1, 'v', -k], [0, 'B', -k], [2, 'v', k + q],
    [3, 'b', -q], [1, 'v', k], [2, 'B', -k - q], [4, 'b', q], [-1, 'b', -p], [3, 'b', q], [4, 'V', -q]]

    indexb = 9

    indexB = 0

    P_data = [[-k, 1, 3], [k, 2, 6], [-k - q, 4, 7], [q, 5, 10], [-q, 8, 11]]

    H_data = [[k, 2, 6], [q, 5, 10]]

    kd_data = [[0, 1], [0, 2], [3, 5], [3, 4], [7, 8], [7, 6], [11, 10], [11, 9]]

    mom_data = [[p, 2], [p, 1], [-k, 4], [-k, 5], [-k - q, 6], [-k - q, 8], [-q, 9], [-q, 10]]

    tensor_part = (-kd(11, 9)*mom(q, 10) - kd(11, 10)*mom(q, 9))*(I*rho*H(k, 2, 6) + P(k, 2, 6))*
    (I*rho*H(q, 5, 10) + P(q, 5, 10))*(-A*(-mom(k, 8) - mom(q, 8))*kd(7, 6) + (-mom(k, 6) - mom(q, 6))*
    kd(7, 8))*(-A*kd(0, 2)*mom(p, 1) + kd(0, 1)*mom(p, 2))*(A*kd(3, 4)*mom(k, 5) - kd(3, 5)*mom(k, 4))*
    P(k, 1, 3)*P(q, 8, 11)*P(k + q, 4, 7)

    scalar_part = A**3*(-sc_prod(B, k) - sc_prod(B, q))*D_v(k)*D_v(q)*alpha(nuo, k, w_k)*beta(nuo, k, w_k)*
    beta_star(nuo, k, w_k)*sc_prod(B, q)**3/(xi(k, w_k)**2*xi(q, w_q)**2*xi(k + q, w_k + w_q)*xi_star(k, w_k)*
    xi_star(q, w_q))
    """

    # --------------------------------------------------------------------------------------------------------------#
    #                     Create a file and start write the information about diagram into it
    # --------------------------------------------------------------------------------------------------------------#

    # according to the given Nickel index of the diagram, create the name of the file with results,
    # select nickel index and symmetry factor
    information_from_Nickel_index = get_information_from_Nickel_index(Nickel_index, counter)

    # creating a file with all output data for the corresponding diagram
    Feynman_graph = open(f"Details about the diagrams/{information_from_Nickel_index.result_file_name}", "w")

    # display the Nickel index of the diagram
    print(f"\nNickel index of the Feynman diagram: {information_from_Nickel_index.nickel_index}")

    # write the Nickel index and symmetry coefficient to the file
    Feynman_graph.write(
        f"Nickel index of the Feynman diagram: {information_from_Nickel_index.nickel_index} \n"
        f"\nDiagram symmetry factor: {information_from_Nickel_index.symmetry_factor} \n"
    )

    # --------------------------------------------------------------------------------------------------------------#
    #                Define a loop structure of the diagram (which lines form loops) and write it into file
    # --------------------------------------------------------------------------------------------------------------#

    """
    Here we implement the following algorithm for determining the diagram loop structure by the Nickel index.
    In our model, a loop (a standard QFT loop in a diagram) is a directed cycle (in terms of graph theory)
    in which the edge containing the core D_v enters only once. To such edges we will then assign momentums
    k and q, while the remaining momentums are determined from the corresponding momentum conservation laws
    at the vertices.
    """

    # start filling the supporting information (topology, momentum and frequency distribution) to file
    Feynman_graph.write(f"\nDiagram description start.\n")

    # list with diagram internal and external lines
    diagram_lines = get_list_with_propagators_from_nickel_index(Nickel_index)

    """
    Here we put the list of all internal lines in the diagram to a dictionary,
    where keys are the numbers that encode lines in a diagram.
    Encoding is done as follows:
    The key i is the number of the element in the list internal_lines that 
    describes the line in diagram and is obtained from the Nickel index using 
    the function get_list_with_propagators_from_nickel_index()
    """

    # write the dictionary with all internal lines to the file
    Feynman_graph.write(
        f"\nPropagators (lines) in the diagram: \n"
        f"{get_line_keywards_to_dictionary(diagram_lines.dict_internal_propagators)} \n"
        f"Notation for line n: [(vertex_1, vertex_2), (field_flows_out_of_vertex_1, field_flows_into_vertex_2)]\n"
    )

    # get list of all loops in the diagram (this function works for diagrams with any number of loops)
    list_of_all_loops_in_diagram = check_if_the_given_lines_combination_is_a_loop_in_diagram(
        diagram_lines,
    )

    # create a dictionary for momentums and frequencies flowing in lines containing kernel D_v
    args_in_helical_propagators = put_momentums_and_frequencies_to_propagators_with_helicity(
        diagram_lines,
        propagators_with_helicity,
        momentums_for_helicity_propagators,
        frequencies_for_helicity_propagators,
    )

    # select those loops that contain only one helical propagator
    loop = get_usual_QFT_loops(list_of_all_loops_in_diagram, args_in_helical_propagators.momentums_for_helical_lines)

    loop_to_txt = list()

    for seq in loop:
        loop_to_txt.append(tuple(map(lambda x: f"line {x}", seq)))

    # write the loop structure of the diagram to the file
    Feynman_graph.write(f"\nLoops in the diagram for a given internal momentum: \n" f"{loop_to_txt} \n")

    # --------------------------------------------------------------------------------------------------------------#
    #                      Get a distribution over momentums and frequencies flowing over lines
    # --------------------------------------------------------------------------------------------------------------#

    """
    Vertices are numbered in ascending order from 0 to number_int_vert - 1 
    in the order they occur in the Nickel index.
    """
    # determine the numeration of start and end vertices in the diagram
    vertex_begin = 0
    vertex_end = number_int_vert - 1

    # assign momentums and frequencies to the corresponding lines of the diagram
    momentum_and_frequency_distribution = get_momentum_and_frequency_distribution(
        diagram_lines,
        args_in_helical_propagators,
        p,
        w,
        vertex_begin,
        vertex_end,
        number_int_vert,
    )

    # obtain the distribution of momentums and frequencies along the lines in the diagram
    # at zero external arguments
    propagator_args_distribution_at_zero_p_and_w = get_momentum_and_frequency_distribution_at_zero_p_and_w(
        diagram_lines,
        momentum_and_frequency_distribution,
        p,
        w,
        momentums_for_helicity_propagators,
        frequencies_for_helicity_propagators,
    )

    Feynman_graph.write(
        f"\nMomentum propagating along the lines: "
        f"\n{get_line_keywards_to_dictionary(momentum_and_frequency_distribution.momentum_distribution)}\n"
        f"\nFrequency propagating along the lines: "
        f"\n{get_line_keywards_to_dictionary(momentum_and_frequency_distribution.frequency_distribution)}\n"
    )

    """
    Next, we need a distribution of inflowing and outflowing fields, momentums and frequencies at each vertex 
    (distinguishing by sign at each vertex the inflowing and outflowing arguments).
    This is necessary, to determine the corresponding VERTEX FACTORS. 
    The following function (see its description for details) gives us all the structures we need.
    """

    # all information about vertexes is collected and summarized
    distribution_of_diagram_parameters_over_vertices = momentum_and_frequency_distribution_at_vertexes(
        diagram_lines, number_int_vert, p, w, propagator_args_distribution_at_zero_p_and_w
    )

    # --------------------------------------------------------------------------------------------------------------#
    #                 Geting the integrand for the diagram (scalar rational function and tensor part)
    # --------------------------------------------------------------------------------------------------------------#

    propagator_product_scalar_and_tensor_parts = IntegrandScalarAndTensorParts()

    structure_of_propagator_product = get_propagator_product(
        distribution_of_diagram_parameters_over_vertices,
        diagram_lines,
        propagator_product_scalar_and_tensor_parts,
        propagator_args_distribution_at_zero_p_and_w,
    )

    propagator_product_scalar_and_tensor_parts = IntegrandScalarAndTensorParts(
        propagator_prod=structure_of_propagator_product.propagator_prod,
        scalar_part=structure_of_propagator_product.scalar_part,
        tensor_part=structure_of_propagator_product.tensor_part,
        P_data=structure_of_propagator_product.P_data,
        H_data=structure_of_propagator_product.H_data,
        WfMath_propagators_prod=structure_of_propagator_product.WfMath_propagators_prod[:-1]
        # delete last symbol "*"
    )

    tensor_and_scalar_parts_of_integrand = adding_vertex_factors_to_product_of_propagators(
        propagator_product_scalar_and_tensor_parts,
        number_int_vert,
        distribution_of_diagram_parameters_over_vertices,
    )

    print(f"\nDiagram integrand: \n{tensor_and_scalar_parts_of_integrand.propagator_prod}")
    print(f"\nProduct of propagators without tensor structure: \n{structure_of_propagator_product.scalar_part}")
    print(
        f"\nDiagram tensor structure before computing tensor convolutions: "
        f"\n{tensor_and_scalar_parts_of_integrand.tensor_part}"
    )

    Feynman_graph.write(f"\nDiagram integrand: \n{tensor_and_scalar_parts_of_integrand.propagator_prod}\n")

    Feynman_graph.write(
        f"\nThe calculation of the integrand is divided into two stages: \n"
        f"1. The calculation of the tensor structure T_ij \n"
        f"2. The calculation of the scalar part F \n"
        f"See file General_notation.txt for details.\n"
    )

    Feynman_graph.write(f"\nExpression for the scalar function F: \n{structure_of_propagator_product.scalar_part}\n")

    Feynman_graph.write(
        f"\nExpression for the tensor function T_ij (numbers for vector indices instead of alphabetic [1]): "
        f"\n{tensor_and_scalar_parts_of_integrand.tensor_part}\n"
    )

    if output_in_WfMath_format == "y":
        Feynman_graph.write(
            f"\nScalar part of the propagator product "
            f" in a Wolfram Mathematica-friendly format: "
            f"\n{propagator_product_scalar_and_tensor_parts.WfMath_propagators_prod}\n"
        )

    # Attention!!!
    # Here we introduce AN EFFECTIVE criterion for the diagram's convergence is that it must
    # consist of propagators proportional to B, i.e. Product(B = 0) = 0.

    # define the parameter responsible for replacing the variables k, q --> B*k/nuo, B*q/nuo
    if tensor_and_scalar_parts_of_integrand.scalar_part.subs(B, 0) == 0:
        is_diagram_convergent = True
        Feynman_graph.write(f"\nThe diagram is UV-convergent: \n{is_diagram_convergent}\n")
    else:
        is_diagram_convergent = False
        Feynman_graph.write(f"\nThe diagram is UV-convergent: \n{is_diagram_convergent}\n")

    # After the change of variables, the diagram depends only on the parameter uo and can be calculated numerically.

    Feynman_graph.write(f"\nDiagram description end.\n")

    # finish filling the supporting information to file
    Feynman_graph.close()

    diagram_data = DiagramData(
        information_from_Nickel_index.symmetry_factor,
        information_from_Nickel_index.nickel_index,
        information_from_Nickel_index.result_file_name,
        distribution_of_diagram_parameters_over_vertices.momentums_at_vertices,
        distribution_of_diagram_parameters_over_vertices.indexb,
        distribution_of_diagram_parameters_over_vertices.indexB,
        tensor_and_scalar_parts_of_integrand.P_data,
        tensor_and_scalar_parts_of_integrand.H_data,
        tensor_and_scalar_parts_of_integrand.kd_data,
        tensor_and_scalar_parts_of_integrand.mom_data,
        tensor_and_scalar_parts_of_integrand.tensor_part,
        tensor_and_scalar_parts_of_integrand.scalar_part,
        is_diagram_convergent,
    )

    return diagram_data
