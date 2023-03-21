from Functions.SymPy_classes import *

# ------------------------------------------------------------------------------------------------------------------#
#                    Get the integrand for the diagram (rational function and tensor part)
# ------------------------------------------------------------------------------------------------------------------#


def define_propagator_product(
    empty_P_data,
    empty_H_data,
    empty_numerator,
    empty_space,
    empty_propagator_data,
    fields_in_propagator,
    momentum_arg,
    frequency_arg,
    in1,
    in2,
):
    """
    The function contains all the information about the propagators of the model.
    It is supposed to apply it to the list of propagators of each specific diagram
    to obtain the corresponding integrand.

    ARGUMENTS:

    empty_P_data = ([]) -- list where information (momentum, frequency, indices) about
    projectors is stored,

    empty_H_data = ([]) -- list where information (momentum, frequency, indices) about
    Helical terms is stored,

    empty_numerator = 1 -- factor by which the corresponding index structure of the propagator
    is multiplied,

    empty_space = "" -- empty string space where momentum and frequency arguments are stored,

    empty_propagator_data = 1 -- factor by which the corresponding propagator is multiplied
    (without index structure),

    fields_in_propagator -- argument(["field1", "field2"]) passed to the function,

    momentum_arg, frequency_arg -- propagator arguments,

    in1, in2 -- indices of the propagator tensor structure
    """
    projector_argument_list = empty_P_data
    helical_argument_list = empty_H_data
    product_of_tensor_operators = empty_numerator
    interspace = empty_space
    product_of_propagators = empty_propagator_data

    match fields_in_propagator:
        case ["v", "v"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2) + I * rho * H(momentum_arg, in1, in2)
            interspace = interspace + f"Pvv[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                beta(nuo, momentum_arg, frequency_arg)
                * beta_star(nuo, momentum_arg, frequency_arg)
                * D_v(momentum_arg)
                / (xi(momentum_arg, frequency_arg) * xi_star(momentum_arg, frequency_arg))
            )

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["v", "V"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = interspace + f"PvV[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= beta_star(nuo, momentum_arg, frequency_arg) / xi_star(momentum_arg, frequency_arg)

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["V", "v"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = interspace + f"PbB[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= beta_star(nuo, momentum_arg, frequency_arg) / xi_star(momentum_arg, frequency_arg)

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["b", "B"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = interspace + f"PbB[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= alpha_star(nuo, momentum_arg, frequency_arg) / xi_star(
                momentum_arg, frequency_arg
            )

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["B", "b"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = interspace + f"PBb[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= alpha_star(nuo, momentum_arg, frequency_arg) / xi_star(
                momentum_arg, frequency_arg
            )

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["v", "b"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2) + I * rho * H(momentum_arg, in1, in2)
            interspace = interspace + f"Pvb[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                I
                * A
                * beta(nuo, momentum_arg, frequency_arg)
                * sc_prod(B, momentum_arg)
                * D_v(momentum_arg)
                / (xi(momentum_arg, frequency_arg) * xi_star(momentum_arg, frequency_arg))
            )

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["b", "v"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2) + I * rho * H(momentum_arg, in1, in2)
            interspace = interspace + f"Pbv[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                I
                * A
                * beta(nuo, momentum_arg, frequency_arg)
                * sc_prod(B, momentum_arg)
                * D_v(momentum_arg)
                / (xi(momentum_arg, frequency_arg) * xi_star(momentum_arg, frequency_arg))
            )

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["b", "b"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2) + I * rho * H(momentum_arg, in1, in2)
            interspace = interspace + f"Pbb[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                A**2
                * sc_prod(B, momentum_arg) ** 2
                * D_v(momentum_arg)
                / (xi(momentum_arg, frequency_arg) * xi_star(momentum_arg, frequency_arg))
            )

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["V", "b"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = interspace + "PVb[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= I * A * sc_prod(B, momentum_arg) / xi_star(momentum_arg, frequency_arg)

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["b", "V"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = interspace + f"PbV[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= I * A * sc_prod(B, momentum_arg) / xi_star(momentum_arg, frequency_arg)

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["B", "v"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = interspace + f"PBv[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= I * sc_prod(B, momentum_arg) / xi_star(momentum_arg, frequency_arg)

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case ["v", "B"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = interspace + f"PvB[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= I * sc_prod(B, momentum_arg) / xi_star(momentum_arg, frequency_arg)

            return [
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                interspace,
                product_of_propagators,
            ]
        case _:
            return sys.exit("Nickel index contains unknown propagator type")


def get_propagator_product(
    distribution_of_momentums_over_vertices,
    set_with_internal_lines,
    empty_P_data,
    empty_H_data,
    empty_numerator,
    empty_space,
    empty_propagator_data,
    momentum_distribution_with_zero_p,
    frequency_distribution_with_zero_w,
):
    """
    This function applies the function define_propagator_product() to the list
    with propagators of a particular diagram.

    ARGUMENTS:

    distribution_of_momentums_over_vertices is given by the function
    momentum_and_frequency_distribution_at_vertexes(),

    set_with_internal_lines is given by the function get_list_with_propagators_from_nickel_index()

    empty_P_data = ([]) -- list where information (momentum, frequency, indices) about
    projectors is stored (this argument is passed to the function define_propagator_product()),

    empty_H_data = ([]) -- list where information (momentum, frequency, indices) about
    Helical terms is stored (this argument is passed to the function define_propagator_product()),

    empty_numerator = 1 -- factor by which the corresponding index structure of the propagator
    is multiplied (this argument is passed to the function define_propagator_product()),

    empty_space = "" -- empty string space where momentum and frequency arguments are stored
    (this argument is passed to the function define_propagator_product()),

    empty_propagator_data = 1 -- factor by which the corresponding propagator is multiplied
    (without index structure) (this argument is passed to the function define_propagator_product()),

    momentum_distribution_with_zero_p, frequency_distribution_with_zero_w are given by the function
    get_momentum_and_frequency_distribution_at_zero_p_and_w()

    OUTPUT DATA EXAMPLE:

    product_of_tensor_operators = (I*rho*H(k, w_k, 2, 6) + P(k, w_k, 2, 6))*(I*rho*H(q, w_q, 5, 10) +
    P(q, w_q, 5, 10))*P(-k, -w_k, 1, 3)*P(-q, -w_q, 8, 11)*P(-k - q, -w_k - w_q, 4, 7)

    projector_argument_list = [
    [-k, -w_k, 1, 3], [k, w_k, 2, 6], [-k - q, -w_k - w_q, 4, 7], [q, w_q, 5, 10], [-q, -w_q, 8, 11]
    ]

    helical_argument_list = [[k, w_k, 2, 6], [q, w_q, 5, 10]]

    propagator_product_for_Wolphram_Mathematica[:-1] = PbB[-k, -w_k]*Pvv[k, w_k]*PvB[-k - q, -w_k - w_q]*
    Pbb[q, w_q]*PbV[-q, -w_q]
    """

    # according to list distribution_of_momentums_over_vertices (vertices ordered) returns a list of indices
    # (see note in the description of momentum_and_frequency_distribution_at_vertexes())
    indexy = list(map(lambda x: x[0], distribution_of_momentums_over_vertices))

    for i in set_with_internal_lines:
        line = set_with_internal_lines[i]
        """
        lines are numbered by digits 
        the indexes in the indexy list are also digits, each of which occurs twice, 
        i.e. forms a line between the corresponding vertices 
        (each three indices in indexy belong to one vertex)
        """
        # .index(i) function returns the position of the first encountered element in the list
        in1 = indexy.index(i)
        # select the first (of two) index corresponding to line i (i encodes line in set_with_internal_lines)
        indexy[in1] = len(set_with_internal_lines)
        # rewrite in1 with a large enough number
        in2 = indexy.index(i)
        # select the secondgt (of two) index corresponding to line i (i encodes line in set_with_internal_lines)
        fields_in_propagator = line[1]
        momentum_arg = momentum_distribution_with_zero_p[i]
        frequency_arg = frequency_distribution_with_zero_w[i]

        all_structures_in_numerator = define_propagator_product(
            empty_P_data,
            empty_H_data,
            empty_numerator,
            empty_space,
            empty_propagator_data,
            fields_in_propagator,
            momentum_arg,
            frequency_arg,
            in1,
            in2,
        )

        empty_numerator = all_structures_in_numerator[0]
        empty_P_data = all_structures_in_numerator[1]
        empty_H_data = all_structures_in_numerator[2]
        empty_space = all_structures_in_numerator[3]
        empty_propagator_data = all_structures_in_numerator[4]

    return [
        empty_numerator,
        empty_P_data,
        empty_H_data,
        empty_space[:-1],  # delete last symbol "*"
        empty_propagator_data,
    ]


def adding_vertex_factors_to_product_of_propagators(
    product_of_tensor_operators,
    Kronecker_delta_structure,
    momentum_structure,
    number_of_vertices,
    distribution_of_momentums_over_vertices,
):
    """
    This function adds tensor vertex factors to the product of the tensor parts of the propagators.
    Thus, this function completes the definition of the tensor part of the integrand of the corresponding diagram

    ARGUMENTS:

    product_of_tensor_operators is given by the function get_propagator_product(),

    Kronecker_delta_structure = ([]) -- list where information (indices) about
    vertex factors is stored,

    momentum_structure = ([]) -- list where information (momentums) about
    vertex factors is stored,

    number_of_vertices = 4 -- see global variables,

    distribution_of_momentums_over_vertices is given by the function
    momentum_and_frequency_distribution_at_vertexes()
    """
    # according to list distribution_of_momentums_over_vertices (vertices ordered) returns a list of fields
    ordered_list_of_fields_flowing_from_vertices = list(map(lambda x: x[1], distribution_of_momentums_over_vertices))

    for vertex_number in range(number_of_vertices):

        vertex_triple = ordered_list_of_fields_flowing_from_vertices[
            3 * vertex_number : 3 * (vertex_number + 1)
        ]  # field triple for corresponding vertex
        sorted_vertex_triple = sorted(vertex_triple, reverse=False)  # ascending sort

        match sorted_vertex_triple:
            case [
                "B",
                "b",
                "v",
            ]:
                in1 = 3 * vertex_number + vertex_triple.index("B")
                in2 = 3 * vertex_number + vertex_triple.index("b")
                in3 = 3 * vertex_number + vertex_triple.index("v")
                Bbv = vertex_factor_Bbv(distribution_of_momentums_over_vertices[in1][2], in1, in2, in3).doit()

                product_of_tensor_operators = product_of_tensor_operators * Bbv
                Kronecker_delta_structure.append([in1, in2])
                Kronecker_delta_structure.append([in1, in3])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])

            case ["V", "v", "v"]:
                in1 = 3 * vertex_number + vertex_triple.index("V")
                # since the two fields are the same, we don't know in advance what position in2 is in
                index_set = [
                    3 * vertex_number,
                    3 * vertex_number + 1,
                    3 * vertex_number + 2,
                ]
                index_set.remove(in1)
                in2 = index_set[0]
                in3 = index_set[1]
                Vvv = vertex_factor_Vvv(distribution_of_momentums_over_vertices[in1][2], in1, in2, in3).doit()

                product_of_tensor_operators = product_of_tensor_operators * Vvv
                Kronecker_delta_structure.append([in1, in3])
                Kronecker_delta_structure.append([in1, in2])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3])

            case ["V", "b", "b"]:
                in1 = 3 * vertex_number + vertex_triple.index("V")
                # since the two fields are the same, we don't know in advance what position in2 is in
                index_set = [
                    3 * vertex_number,
                    3 * vertex_number + 1,
                    3 * vertex_number + 2,
                ]
                index_set.remove(in1)
                in2 = index_set[0]
                in3 = index_set[1]
                Vbb = vertex_factor_Vvv(
                    distribution_of_momentums_over_vertices[in1][2], in1, in2, in3
                ).doit()  # vertex_factor_Vvv = vertex_factor_Vbb by definiton

                product_of_tensor_operators = product_of_tensor_operators * Vbb
                Kronecker_delta_structure.append([in1, in3])
                Kronecker_delta_structure.append([in1, in2])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3])
            case _:
                sys.exit("Unknown vertex type")
    return product_of_tensor_operators, Kronecker_delta_structure, momentum_structure
