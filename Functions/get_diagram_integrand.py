from Functions.Data_classes import *
from Functions.SymPy_classes import *

# ------------------------------------------------------------------------------------------------------------------#
#                    Get the integrand for the diagram (rational function and tensor part)
# ------------------------------------------------------------------------------------------------------------------#


def define_propagator_product(
    empty_propagator_product_scalar_and_tensor_parts: IntegrandPropagatorProduct,
    fields_in_propagator: Any,
    momentum_arg: Any,
    frequency_arg: Any,
    index1: int,
    index2: int,
):
    """
    The function contains all the information about the propagators of the model.
    It is supposed to apply it to the list of propagators of each specific diagram
    to obtain the corresponding integrand.

    ARGUMENTS:

    projector_argument_list = [] -- list where information (momentum, frequency, indices) about
    projectors is stored,

    helical_argument_list = [] -- list where information (momentum, frequency, indices) about
    Helical terms is stored,

    product_of_tensor_operators = 1 -- factor by which the corresponding index structure of the propagator
    is multiplied,

    interspace = "" -- empty string space where momentum and frequency arguments are stored,

    product_of_propagators = 1 -- factor by which the corresponding propagator is multiplied
    (without index structure),

    fields_in_propagator -- argument(["field1", "field2"]) passed to the function,

    momentum_arg, frequency_arg -- propagator arguments,

    in1, in2 -- indices of the propagator tensor structure
    """
    propagator = empty_propagator_product_scalar_and_tensor_parts.propagator_prod
    projector_argument_list = empty_propagator_product_scalar_and_tensor_parts.P_data
    helical_argument_list = empty_propagator_product_scalar_and_tensor_parts.H_data
    product_of_tensor_operators = empty_propagator_product_scalar_and_tensor_parts.tensor_part
    product_of_propagators_scalar_parts = empty_propagator_product_scalar_and_tensor_parts.scalar_part
    WfMath_propagators_prod = empty_propagator_product_scalar_and_tensor_parts.WfMath_propagators_prod

    match fields_in_propagator:
        case ["v", "v"]:
            propagator *= Pvv(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= R(momentum_arg, index1, index2).doit()
            WfMath_propagators_prod = WfMath_propagators_prod + f"Pvv[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            helical_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= Pvv_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["v", "V"]:
            propagator *= PvV(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= P(momentum_arg, index1, index2)
            WfMath_propagators_prod = WfMath_propagators_prod + f"PvV[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= PvV_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["V", "v"]:
            # Vv = complex_conjugate(vV)
            propagator *= PVv(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= P(momentum_arg, index1, index2)
            WfMath_propagators_prod = WfMath_propagators_prod + f"PbB[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= PVv_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["b", "B"]:
            propagator *= PbB(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= P(momentum_arg, index1, index2)
            WfMath_propagators_prod = WfMath_propagators_prod + f"PbB[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= PbB_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["B", "b"]:
            # Bb = complex_conjugate(bB)
            propagator *= PBb(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= P(momentum_arg, index1, index2)
            WfMath_propagators_prod = WfMath_propagators_prod + f"PBb[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= PBb_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["v", "b"]:
            propagator *= Pvb(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= R(momentum_arg, index1, index2).doit()
            WfMath_propagators_prod = WfMath_propagators_prod + f"Pvb[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            helical_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= Pvb_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["b", "v"]:
            # bv = complex_conjugate(vb)
            propagator *= Pbv_scalar_part(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= R(momentum_arg, index1, index2).doit()
            WfMath_propagators_prod = WfMath_propagators_prod + f"Pbv[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            helical_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= Pbv_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["b", "b"]:
            propagator *= Pbb(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= R(momentum_arg, index1, index2).doit()
            WfMath_propagators_prod = WfMath_propagators_prod + f"Pbb[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            helical_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= Pbb_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["V", "b"]:
            propagator *= PVb(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= P(momentum_arg, index1, index2)
            WfMath_propagators_prod = WfMath_propagators_prod + "PVb[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= PVb_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["b", "V"]:
            # bV = complex_conjugate(Vb)
            propagator *= PbV(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= P(momentum_arg, index1, index2)
            WfMath_propagators_prod = WfMath_propagators_prod + f"PbV[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= PbV_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["B", "v"]:
            propagator *= PBv(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= P(momentum_arg, index1, index2)
            WfMath_propagators_prod = WfMath_propagators_prod + f"PBv[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= PBv_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case ["v", "B"]:
            # vB = complex_conjugate(Bv)
            propagator *= PvB(momentum_arg, frequency_arg, index1, index2)
            product_of_tensor_operators *= P(momentum_arg, index1, index2)
            WfMath_propagators_prod = WfMath_propagators_prod + f"PvB[{momentum_arg}, {frequency_arg}]*"
            projector_argument_list += [[momentum_arg, index1, index2]]
            product_of_propagators_scalar_parts *= PvB_scalar_part(momentum_arg, frequency_arg).doit()

            integrand_data = IntegrandPropagatorProduct(
                propagator,
                product_of_propagators_scalar_parts,
                product_of_tensor_operators,
                projector_argument_list,
                helical_argument_list,
                WfMath_propagators_prod,
            )
            return integrand_data
        case _:
            return sys.exit("Nickel index contains unknown propagator type")


def get_propagator_product(
    distribution_of_diagram_parameters_over_vertices: MomentumFrequencyDistributionAtVertices,
    diagram_lines: InternalAndExternalLines,
    empty_propagator_product_scalar_and_tensor_parts: IntegrandPropagatorProduct,
    line_args_distribution_at_zero_external_args: ArgumentsDistributionAlongLinesAtZeroExternalArguments,
):
    """
    This function applies the function define_propagator_product() to the list
    with propagators of a particular diagram.

    ARGUMENTS:

    distribution_of_momentums_over_vertices is given by the function
    momentum_and_frequency_distribution_at_vertexes(),

    set_with_internal_lines is given by the function get_list_with_propagators_from_nickel_index()

    empty_P_data = [] -- list where information (momentum, frequency, indices) about
    projectors is stored (this argument is passed to the function define_propagator_product()),

    empty_H_data = [] -- list where information (momentum, frequency, indices) about
    Helical terms is stored (this argument is passed to the function define_propagator_product()),

    empty_tensor_part = 1 -- factor by which the corresponding index structure of the propagator
    is multiplied (this argument is passed to the function define_propagator_product()),

    empty_space = "" -- empty string space where momentum and frequency arguments are stored
    (this argument is passed to the function define_propagator_product()),

    empty_scalar_part = 1 -- factor by which the corresponding propagator is multiplied
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
    distribution_of_momentums_over_vertices = distribution_of_diagram_parameters_over_vertices.momentums_at_vertices
    dict_with_internal_lines = diagram_lines.dict_internal_propagators

    momentum_distribution_with_zero_p = line_args_distribution_at_zero_external_args.momentum_distribution
    frequency_distribution_with_zero_w = line_args_distribution_at_zero_external_args.frequency_distribution

    # according to list distribution_of_momentums_over_vertices (vertices ordered) returns a list of indices
    # (see note in the description of momentum_and_frequency_distribution_at_vertexes())
    indexy = list(map(lambda x: x[0], distribution_of_momentums_over_vertices))

    for i in dict_with_internal_lines:
        line = dict_with_internal_lines[i]
        """
        lines are numbered by digits 
        the indexes in the indexy list are also digits, each of which occurs twice, 
        i.e. forms a line between the corresponding vertices 
        (each three indices in indexy belong to one vertex)
        """
        # .index(i) function returns the position of the first encountered element in the list
        in1 = indexy.index(i)
        # select the first (of two) index corresponding to line i (i encodes line in set_with_internal_lines)
        indexy[in1] = len(dict_with_internal_lines)
        # rewrite in1 with a large enough number
        in2 = indexy.index(i)
        # select the secondgt (of two) index corresponding to line i (i encodes line in set_with_internal_lines)
        fields_in_propagator = line[1]
        momentum_arg = momentum_distribution_with_zero_p[i]
        frequency_arg = frequency_distribution_with_zero_w[i]

        all_structures_in_propagator_product = define_propagator_product(
            empty_propagator_product_scalar_and_tensor_parts,
            fields_in_propagator,
            momentum_arg,
            frequency_arg,
            in1,
            in2,
        )

        empty_propagator_product_scalar_and_tensor_parts = all_structures_in_propagator_product

    return empty_propagator_product_scalar_and_tensor_parts


def adding_vertex_factors_to_product_of_propagators(
    integrand_scalar_and_tensor_parts: IntegrandScalarAndTensorParts,
    number_of_vertices: int,
    distribution_of_diagram_parameters_over_vertices: MomentumFrequencyDistributionAtVertices,
):
    """
    This function adds tensor vertex factors to the product of the tensor parts of the propagators.
    Thus, this function completes the definition of the tensor part of the integrand of the corresponding diagram

    ARGUMENTS:

    product_of_tensor_operators is given by the function get_propagator_product(),

    Kronecker_delta_structure = [] -- list where information (indices) about
    vertex factors is stored,

    momentum_structure = [] -- list where information (momentums) about
    vertex factors is stored,

    number_of_vertices = 4 -- see global variables,

    distribution_of_momentums_over_vertices is given by the function
    momentum_and_frequency_distribution_at_vertexes()
    """
    propagator_product = integrand_scalar_and_tensor_parts.propagator_prod
    product_of_tensor_operators = integrand_scalar_and_tensor_parts.tensor_part
    Kronecker_delta_structure = integrand_scalar_and_tensor_parts.kd_data
    momentum_structure = integrand_scalar_and_tensor_parts.mom_data

    distribution_of_momentums_over_vertices = distribution_of_diagram_parameters_over_vertices.momentums_at_vertices

    # according to list distribution_of_momentums_over_vertices (vertices ordered) returns a list of fields
    ordered_list_of_fields_flowing_from_vertices = list(map(lambda x: x[1], distribution_of_momentums_over_vertices))

    for vertex_number in range(number_of_vertices):
        vertex_triple = ordered_list_of_fields_flowing_from_vertices[
            3 * vertex_number : 3 * (vertex_number + 1)
        ]  # field triple for corresponding vertex
        sorted_vertex_triple = sorted(vertex_triple, reverse=False)  # ascending sort

        match sorted_vertex_triple:
            case ["B", "b", "v"]:
                in1 = 3 * vertex_number + vertex_triple.index("B")
                in2 = 3 * vertex_number + vertex_triple.index("b")
                in3 = 3 * vertex_number + vertex_triple.index("v")
                Bbv = vertex_factor_Bbv(distribution_of_momentums_over_vertices[in1][2], in1, in2, in3)

                propagator_product *= Bbv
                product_of_tensor_operators = product_of_tensor_operators * Bbv.doit()
                Kronecker_delta_structure.append([in1, in2])
                Kronecker_delta_structure.append([in1, in3])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])

            case ["V", "v", "v"]:
                in1 = 3 * vertex_number + vertex_triple.index("V")
                # since the two fields are the same, we don't know in advance what position in2 is in
                index_set = [3 * vertex_number, 3 * vertex_number + 1, 3 * vertex_number + 2]
                index_set.remove(in1)
                in2 = index_set[0]
                in3 = index_set[1]
                Vvv = vertex_factor_Vvv(distribution_of_momentums_over_vertices[in1][2], in1, in2, in3)

                propagator_product *= Vvv
                product_of_tensor_operators = product_of_tensor_operators * Vvv.doit()
                Kronecker_delta_structure.append([in1, in3])
                Kronecker_delta_structure.append([in1, in2])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3])

            case ["V", "b", "b"]:
                in1 = 3 * vertex_number + vertex_triple.index("V")
                # since the two fields are the same, we don't know in advance what position in2 is in
                index_set = [3 * vertex_number, 3 * vertex_number + 1, 3 * vertex_number + 2]
                index_set.remove(in1)
                in2 = index_set[0]
                in3 = index_set[1]
                # vertex_factor_Vvv = -vertex_factor_Vbb
                Vbb = vertex_factor_Vbb(distribution_of_momentums_over_vertices[in1][2], in1, in2, in3)

                propagator_product *= Vbb
                product_of_tensor_operators = product_of_tensor_operators * Vbb.doit()
                Kronecker_delta_structure.append([in1, in3])
                Kronecker_delta_structure.append([in1, in2])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])
                momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3])
            case _:
                sys.exit("Unknown vertex type")

    tensor_and_scalar_parts_of_integrand = IntegrandScalarAndTensorParts(
        propagator_prod=propagator_product,
        scalar_part=integrand_scalar_and_tensor_parts.scalar_part,
        tensor_part=product_of_tensor_operators,
        P_data=integrand_scalar_and_tensor_parts.P_data,
        H_data=integrand_scalar_and_tensor_parts.H_data,
        WfMath_propagators_prod=integrand_scalar_and_tensor_parts.WfMath_propagators_prod,
        mom_data=momentum_structure,
        kd_data=Kronecker_delta_structure,
    )
    return tensor_and_scalar_parts_of_integrand
