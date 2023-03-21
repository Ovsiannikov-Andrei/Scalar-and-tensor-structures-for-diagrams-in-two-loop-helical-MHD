from Functions.create_file_with_output import *
from Functions.get_diagram_loop_structure import *
from Functions.get_momentum_frequency_distribution import *
from Functions.get_diagram_integrand import *

# ------------------------------------------------------------------------------------------------------------------#
#                                 Part 1: Diagram description (lines, vertices, etc.)
# ------------------------------------------------------------------------------------------------------------------#


def get_info_about_diagram(graf, counter):
    """
    This function returns all information about the structure of a particular diagram.

    ARGUMENTS:

    graf is given by get_information_from_Nickel_index(),
    counter -- some counter of the number of computed diagrams

    OUTPUT DATA EXAMPLE:

    output_file_name = 1. Diagram__e12_23_3_e__0B_bB_vv__vB_bb__bV__0b_.txt

    moznost = [[-1, 'B', p], [0, 'b', k], [1, 'v', -k], [0, 'B', -k], [2, 'v', k + q], [3, 'b', -q],
    [1, 'v', k], [2, 'B', -k - q], [4, 'b', q], [-1, 'b', -p], [3, 'b', q], [4, 'V', -q]]

    indexb = 9

    indexB = 0

    P_structure = [[-k, 1, 3], [k, 2, 6], [-k - q, 4, 7], [q, 5, 10], [-q, 8, 11]]

    H_structure = [[k, 2, 6], [q, 5, 10]]

    kd_structure = [[0, 1], [0, 2], [3, 5], [3, 4], [7, 8], [7, 6], [11, 10], [11, 9]]

    hyb_structure = [[p, 2], [p, 1], [-k, 4], [-k, 5], [-k - q, 6], [-k - q, 8], [-q, 9], [-q, 10]]

    Tenzor = (-kd(11, 9)*hyb(q, 10) - kd(11, 10)*hyb(q, 9))*(I*rho*H(k, 2, 6) + P(k, 2, 6))*
    (I*rho*H(q, 5, 10) + P(q, 5, 10))*(-A*(-hyb(k, 8) - hyb(q, 8))*kd(7, 6) + (-hyb(k, 6) - hyb(q, 6))*
    kd(7, 8))*(-A*kd(0, 2)*hyb(p, 1) + kd(0, 1)*hyb(p, 2))*(A*kd(3, 4)*hyb(k, 5) - kd(3, 5)*hyb(k, 4))*
    P(k, 1, 3)*P(q, 8, 11)*P(k + q, 4, 7)

    Product = A**3*(-sc_prod(B, k) - sc_prod(B, q))*D_v(k)*D_v(q)*alpha(k, w_k)*beta(k, w_k)*beta_star(k, w_k)*
    sc_prod(B, q)**3/(xi(k, w_k)**2*xi(q, w_q)**2*xi(k + q, w_k + w_q)*xi_star(k, w_k)*xi_star(q, w_q))
    """

    # --------------------------------------------------------------------------------------------------------------#
    #                     Create a file and start write the information about diagram into it
    # --------------------------------------------------------------------------------------------------------------#

    information_from_Nickel_index = get_information_from_Nickel_index(graf, counter)

    output_file_name = information_from_Nickel_index[0]
    # according to the given Nickel index of the diagram, create the name of the file with results

    nickel_index = information_from_Nickel_index[1]
    # get Nickel index from the line with the data

    symmetry_coefficient = information_from_Nickel_index[2]
    # get symmetry factor from the line with the data

    Feynman_graph = open(f"Results/{output_file_name}", "w")
    # creating a file with all output data for the corresponding diagram

    print(f"\nNickel index of the Feynman diagram: {nickel_index}")
    # display the Nickel index of the diagram

    Feynman_graph.write(f"Nickel index of the Feynman diagram: {nickel_index} \n")
    # write the Nickel index to the file

    Feynman_graph.write(f"\nDiagram symmetry factor: {symmetry_coefficient} \n")
    # write the symmetry coefficient to the file

    # --------------------------------------------------------------------------------------------------------------#
    #                Define a loop structure of the diagram (which lines form loops) and write it into file
    # --------------------------------------------------------------------------------------------------------------#

    Feynman_graph.write(f"\nSupporting information begin:\n")
    # start filling the supporting information (topology, momentum and frequency distribution) to file

    internal_lines = get_list_with_propagators_from_nickel_index(graf)[0]
    # list with diagram internal lines

    dict_with_internal_lines = get_list_as_dictionary(internal_lines)
    # put the list of all internal lines in the diagram to a dictionary

    Feynman_graph.write(
        f"\nPropagators in the diagram: \n" f"{get_line_keywards_to_dictionary(dict_with_internal_lines)} \n"
    )  # write the dictionary with all internal lines to the file

    list_of_all_loops_in_diagram = check_if_the_given_lines_combination_is_a_loop_in_diagram(
        list_of_all_possible_lines_combinations(dict_with_internal_lines),
        dict_with_internal_lines,
    )  # get list of all loops in the diagram (this function works for diagrams with any number of loops)

    momentums_in_helical_propagators = put_momentums_and_frequencies_to_propagators_with_helicity(
        dict_with_internal_lines,
        propagators_with_helicity,
        momentums_for_helicity_propagators,
        frequencies_for_helicity_propagators,
    )[
        0
    ]  # create a dictionary for momentums flowing in lines containing kernel D_v

    loop = get_usual_QFT_loops(list_of_all_loops_in_diagram, momentums_in_helical_propagators)
    # select only those loops that contain only one helical propagator (usual QFT loops)

    Feynman_graph.write(
        f"\nLoops in the diagram for a given internal momentum "
        f"(digit corresponds to the line from previous dictionary): \n{loop} \n"
    )  # write the loop structure of the diagram to the file

    # --------------------------------------------------------------------------------------------------------------#
    #                      Get a distribution over momentums and frequencies flowing over lines
    # --------------------------------------------------------------------------------------------------------------#

    frequencies_in_helical_propagators = put_momentums_and_frequencies_to_propagators_with_helicity(
        dict_with_internal_lines,
        propagators_with_helicity,
        momentums_for_helicity_propagators,
        frequencies_for_helicity_propagators,
    )[
        1
    ]  # create a dictionary with frequency arguments for propagators defining loops

    # determine the start and end vertices in the diagram
    vertex_begin = 0
    vertex_end = number_int_vert - 1

    momentum_and_frequency_distribution = get_momentum_and_frequency_distribution(
        dict_with_internal_lines,
        momentums_in_helical_propagators,
        frequencies_in_helical_propagators,
        p,
        w,
        vertex_begin,
        vertex_end,
        number_int_vert,
    )  # assign momentums and frequencies to the corresponding lines of the diagram

    momentum_distribution = momentum_and_frequency_distribution[0]
    # dictionary with momentums distributed along lines
    frequency_distribution = momentum_and_frequency_distribution[1]
    # dictionary with frequencies distributed along lines

    propagator_args_distribution_at_zero_p_and_w = get_momentum_and_frequency_distribution_at_zero_p_and_w(
        dict_with_internal_lines,
        momentum_distribution,
        frequency_distribution,
        p,
        w,
        momentums_for_helicity_propagators,
        frequencies_for_helicity_propagators,
    )  # obtain the distribution of momentums and frequencies along the lines in the diagram
    # at zero external arguments

    momentum_distribution_at_zero_external_momentum = propagator_args_distribution_at_zero_p_and_w[0]
    # dictionary with momentums distributed along lines (at zero p)
    frequency_distribution_at_zero_external_frequency = propagator_args_distribution_at_zero_p_and_w[1]
    # dictionary with frequencies distributed along lines (at zero w)

    Feynman_graph.write(
        f"\nMomentum propagating along the lines: " f"\n{get_line_keywards_to_dictionary(momentum_distribution)}\n"
    )

    Feynman_graph.write(
        f"\nFrequency propagating along the lines: " f"\n{get_line_keywards_to_dictionary(frequency_distribution)}\n"
    )

    external_lines = get_list_with_propagators_from_nickel_index(graf)[1]
    # list with diagram external lines

    distribution_of_diagram_parameters_over_vertices = momentum_and_frequency_distribution_at_vertexes(
        external_lines,
        dict_with_internal_lines,
        number_int_vert,
        p,
        w,
        momentum_distribution_at_zero_external_momentum,
        frequency_distribution_at_zero_external_frequency,
    )  # all information about the diagram is collected and summarized
    # (which fields and with which arguments form pairs (lines in the diagram))

    indexB = distribution_of_diagram_parameters_over_vertices[0]

    indexb = distribution_of_diagram_parameters_over_vertices[1]

    frequency_and_momentum_distribution_at_vertexes = distribution_of_diagram_parameters_over_vertices[3]

    moznost = distribution_of_diagram_parameters_over_vertices[4]

    Feynman_graph.write(
        f"\nMomentum and frequency distribution at the vertices: "
        f"\n{frequency_and_momentum_distribution_at_vertexes} \n"
    )

    # --------------------------------------------------------------------------------------------------------------#
    #                 Geting the integrand for the diagram (scalar rational function and tensor part)
    # --------------------------------------------------------------------------------------------------------------#

    Tenzor = 1
    # here we save the tensor structure

    Product = 1
    # here we save the product of propagators (without tensor structure)

    P_structure = []
    # here we save all indices of the projctors in the form [[momentum, index1, index2]]

    H_structure = []
    # here we save all indices of the helical structures in the form [[momentum, index1, index2]]

    propagator_product_for_WfMath = ""
    # here we save the propagator product argument structure (for Wolfram Mathematica file)

    structure_of_propagator_product = get_propagator_product(
        moznost,
        dict_with_internal_lines,
        P_structure,
        H_structure,
        Tenzor,
        propagator_product_for_WfMath,
        Product,
        momentum_distribution_at_zero_external_momentum,
        frequency_distribution_at_zero_external_frequency,
    )

    Tenzor = structure_of_propagator_product[0]

    P_structure = structure_of_propagator_product[1]

    H_structure = structure_of_propagator_product[2]

    propagator_product_for_WfMath = structure_of_propagator_product[3]

    Product = structure_of_propagator_product[4]

    Feynman_graph.write(f"\nArgument structure in the propagator product: \n{propagator_product_for_WfMath}\n")

    print(f"\nProduct of propagators without tensor structure: \n{Product}")

    Feynman_graph.write(f"\nProduct of propagators without tensor structure: \n{Product}\n")

    kd_structure = []
    # here we save all indices in Kronecker delta in the form [ [index 1, index 2]]

    hyb_structure = []
    # here we save all momentums and their components in the form [ [ k, i] ]

    whole_tensor_structure_of_integrand_numerator = adding_vertex_factors_to_product_of_propagators(
        Tenzor, kd_structure, hyb_structure, number_int_vert, moznost
    )

    Tenzor = whole_tensor_structure_of_integrand_numerator[0]

    kd_structure = whole_tensor_structure_of_integrand_numerator[1]

    hyb_structure = whole_tensor_structure_of_integrand_numerator[2]

    print(f"\nDiagram tensor structure before computing tensor convolutions: \n{Tenzor}")

    Feynman_graph.write(f"\nDiagram tensor structure before computing tensor convolutions: \n{Tenzor}\n")

    # Here we introduce an effective criterion for the of a diagram convergence is that it must
    # consist of propagators proportional to B, i.e. Product(B = 0) = 0.

    if Product.subs(B, 0) == 0:
        # define the parameter responsible for replacing the variables k, q --> B*k/nuo, B*q/nuo
        is_diagram_convergent = True
        Feynman_graph.write(f"\nThe diagram is convergent.\n")
        Feynman_graph.write(f"diagram_is_convergent = {is_diagram_convergent}\n")
    else:
        is_diagram_convergent = False
        Feynman_graph.write(f"\nThe diagram contains divergent contributions.\n")
        Feynman_graph.write(f"diagram_is_convergent = {is_diagram_convergent}\n")

    # After the change of variables, the diagram depends only on the parameter uo and can be calculated numerically.

    Feynman_graph.write(f"\nSupporting information end.\n")

    Feynman_graph.close()
    # finish filling the supporting information to file

    return [
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
    ]
