import sympy as sym
from sympy import *

# ------------------------------------------------------------------------------------------------------------------#
#                      We get a distribution over momentums and frequencies flowing over lines
# ------------------------------------------------------------------------------------------------------------------#


def get_momentum_and_frequency_distribution(
    internal_lines,
    momentums_in_helical_propagators,
    frequencies_in_helical_propagators,
    external_momentum,
    external_frequency,
    begin_vertex,
    end_vertex,
    number_int_vert,
):
    """
    It assigns momentums and frequencies to the internal lines of the diagram.

    ARGUMENTS:

    list internal_lines is given by the function get_list_with_propagators_from_nickel_index(),

    for momentums_in_helical_propagators, frequencies_in_helical_propagators, external_momentum = p,
    external_frequency = w see global variables,

    begin_vertex = 0 -- vertex through which the field B flows into and
    end_vertex = 3 -- vertex through which the field b flows out

    OUTPUT DATA EXAMPLE:

    [{0: -k + p, 1: k, 2: -k + p - q, 3: q, 4: p - q}, {0: w - w_k, 1: w_k, 2: w - w_k - w_q, 3: w_q, 4: w - w_q}]
    """

    length = len(internal_lines)

    # creating unknown momentums and frequencies for each line
    momentums_for_all_propagators = [symbols(f"k_{i}") for i in range(length)]
    frequencies_for_all_propagators = [symbols(f"w_{i}") for i in range(length)]

    distribution_of_arbitrary_momentums = dict()
    distribution_of_arbitrary_frequencies = dict()

    # we assign arbitrary momentums and frequencies to propogators, excluding those
    # that contain helical terms, since they are already assigned arguments
    for i in range(length):
        if i not in momentums_in_helical_propagators:
            distribution_of_arbitrary_momentums[i] = momentums_for_all_propagators[i]
        else:
            distribution_of_arbitrary_momentums[i] = momentums_in_helical_propagators[i]
        if i not in frequencies_in_helical_propagators:
            distribution_of_arbitrary_frequencies[i] = frequencies_for_all_propagators[i]
        else:
            distribution_of_arbitrary_frequencies[i] = frequencies_in_helical_propagators[i]

    momentum_conservation_law = [0] * number_int_vert
    frequency_conservation_law = [0] * number_int_vert

    """
    The unknown momentums and frequencies are determined using the appropriate conservation
    law at each vertex: the sum of the inflowing and outflowing arguments must equal to 0 
    for each vertex.

    In our case, momentum and frequency flows into the diagram via field B and flows out 
    through field b. We assume that the arguments flowing into the vertex are positive, and
    the arguments flowing out it are negative.
    """
    for vertex in range(number_int_vert):
        if vertex == begin_vertex:
            # external argument flows out from this vertex to the diagram
            momentum_conservation_law[vertex] += -external_momentum
            frequency_conservation_law[vertex] += -external_frequency
        elif vertex == end_vertex:
            # external argument flows into this vertex from the diagram
            momentum_conservation_law[vertex] += external_momentum
            frequency_conservation_law[vertex] += external_frequency

        for line_number in range(length):
            momentum = distribution_of_arbitrary_momentums[line_number]
            frequency = distribution_of_arbitrary_frequencies[line_number]
            line = internal_lines[line_number][0]

            if vertex in line:
                # condition that vertex is the starting point of the line
                if line.index(vertex) % 2 == 0:
                    # if the vertex is the end point of the line, then the argument flows into it (with (+)),
                    # otherwise, it flows out (with (-))
                    momentum_conservation_law[vertex] += momentum
                    frequency_conservation_law[vertex] += frequency
                else:
                    momentum_conservation_law[vertex] += -momentum
                    frequency_conservation_law[vertex] += -frequency

    # there are 1 more conservation laws than unknown variables ==> one equation must hold identically

    list_of_momentum_conservation_laws = [momentum_conservation_law[i] for i in range(number_int_vert)]
    list_of_arbitrary_momentums = [
        momentums_for_all_propagators[i] for i in range(length) if i not in momentums_in_helical_propagators
    ]
    list_of_frequency_conservation_laws = [frequency_conservation_law[i] for i in range(number_int_vert)]
    list_of_arbitrary_frequencies = [
        frequencies_for_all_propagators[i] for i in range(length) if i not in frequencies_in_helical_propagators
    ]

    define_arbitrary_momentums = sym.solve(
        list_of_momentum_conservation_laws, list_of_arbitrary_momentums
    )  # overcrowded system solved
    define_arbitrary_frequencies = sym.solve(
        list_of_frequency_conservation_laws, list_of_arbitrary_frequencies
    )  # overcrowded system solved

    momentum_distribution = dict()
    frequency_distribution = dict()
    # dictionaries with momentums and frequencies flowing along the corresponding line are created
    for i in range(length):
        if i not in momentums_in_helical_propagators:
            momentum_distribution[i] = define_arbitrary_momentums[momentums_for_all_propagators[i]]
        else:
            momentum_distribution[i] = momentums_in_helical_propagators[i]
        if i not in frequencies_in_helical_propagators:
            frequency_distribution[i] = define_arbitrary_frequencies[frequencies_for_all_propagators[i]]
        else:
            frequency_distribution[i] = frequencies_in_helical_propagators[i]

    return [momentum_distribution, frequency_distribution]


def get_momentum_and_frequency_distribution_at_zero_p_and_w(
    internal_lines,
    momentum_distribution,
    frequency_distribution,
    external_momentum,
    external_frequency,
    momentums_in_helical_propagators,
    frequencies_in_helical_propagators,
):
    """
    Gives the distribution along the lines of momentums and frequencies in the diagram at zero
    inflowing paremeters

    ARGUMENTS:

    internal_lines is given by get_list_with_propagators_from_nickel_index(),

    momentum_distribution and frequency_distribution are given by get_momentum_and_frequency_distribution()

    for external_momentum = p, external_frequency = w,  momentums_in_helical_propagators, and
    frequencies_in_helical_propagators see global variables

    OUTPUT DATA EXAMPLE:

    [{0: -k, 1: k, 2: -k - q, 3: q, 4: -q}, {0: -w_k, 1: w_k, 2: -w_k - w_q, 3: w_q, 4: -w_q}]
    """

    momentum_distribution_at_zero_external_momentum = dict()
    frequency_distribution_at_zero_external_frequency = dict()

    length = len(internal_lines)

    list_with_momentums = [0] * length
    list_with_frequencies = [0] * length

    for i in range(length):
        # momentum distribution at zero external momentum
        if i not in momentums_in_helical_propagators:
            list_with_momentums[i] += momentum_distribution[i].subs(external_momentum, 0)
            momentum_distribution_at_zero_external_momentum.update({i: list_with_momentums[i]})
        else:
            momentum_distribution_at_zero_external_momentum.update({i: momentums_in_helical_propagators[i]})
        # frequency distribution at zero external frequency
        if i not in frequencies_in_helical_propagators:
            list_with_frequencies[i] += frequency_distribution[i].subs(external_frequency, 0)
            frequency_distribution_at_zero_external_frequency.update({i: list_with_frequencies[i]})
        else:
            frequency_distribution_at_zero_external_frequency.update({i: frequencies_in_helical_propagators[i]})

    return [
        momentum_distribution_at_zero_external_momentum,
        frequency_distribution_at_zero_external_frequency,
    ]


def momentum_and_frequency_distribution_at_vertexes(
    external_lines,
    internal_lines,
    number_of_all_vertices,
    external_momentum,
    external_frequency,
    momentum_distribution_at_zero_external_momentum,
    frequency_distribution_at_zero_external_frequency,
):
    """
    Gives the distribution  of momentums and frequencies at the three-point vertices in the diagram at zero
    inflowing paremeters

    ARGUMENTS:

    external_lines and internal_lines are given by get_list_with_propagators_from_nickel_index(),

    for external_momentum = p, external_frequency = w, and number_of_all_vertices see global variables,

    momentum_distribution_at_zero_external_momentum, frequency_distribution_at_zero_external_frequency are
    given by get_momentum_and_frequency_distribution_at_zero_p_and_w()

    Note:

    The numeric indexes here are the indexes of the corresponding field resulting from the given vertex.
    This field can pair with another (to form a line in diagram) only if it has exactly the same index
    (there is only one such field!).

    OUTPUT DATA EXAMPLE:

    indexB = 0

    indexb = 9

    data_for_vertexes_distribution = [
    [-1, 'B', p, w], [0, 'b', k, w_k], [1, 'v', -k, -w_k], [0, 'B', -k, -w_k], [2, 'v', k + q, w_k + w_q],
    [3, 'b', -q, -w_q], [1, 'v', k, w_k], [2, 'B', -k - q, -w_k - w_q], [4, 'b', q, w_q], [-1, 'b', -p, -w],
    [3, 'b', q, w_q], [4, 'V', -q, -w_q]]
    ]

    frequency_and_momentum_distribution_at_vertexes = {
    ('vertex', 0): [[-1, 'B', p, w], [0, 'b', k, w_k], [1, 'v', -k, -w_k]], ('vertex', 1): [[0, 'B', -k, -w_k],
    [2, 'v', k + q, w_k + w_q], [3, 'b', -q, -w_q]], ('vertex', 2): [[1, 'v', k, w_k], [2, 'B', -k - q, -w_k - w_q],
    [4, 'b', q, w_q]], ('vertex', 3): [[-1, 'b', -p, -w], [3, 'b', q, w_q], [4, 'V', -q, -w_q]]
    }

    data_for_vertexes_momentum_distribution = [
    [-1, 'B', p], [0, 'b', k], [1, 'v', -k], [0, 'B', -k], [2, 'v', k + q], [3, 'b', -q], [1, 'v', k],
    [2, 'B', -k - q], [4, 'b', q], [-1, 'b', -p], [3, 'b', q], [4, 'V', -q]
    ]
    """

    # each vertex has three tails. We create an array to store information about each tail of each vertex.
    data_for_vertexes_distribution = [0] * (number_of_all_vertices * 3)

    # here we deploy external momentum and frequency
    # (out of 12 available tails in the two-loop diagram, 2 are external)
    for line in external_lines:
        end_vertex = line[0][1]
        outflowing_field = line[1][1]
        if outflowing_field == "B":
            data_for_vertexes_distribution[3 * end_vertex] = [
                -1,
                "B",
                external_momentum,
                external_frequency,
            ]
            indexB = 3 * end_vertex  # save the index of the external field b
        else:
            data_for_vertexes_distribution[3 * end_vertex] = [
                -1,
                outflowing_field,
                -external_momentum,
                -external_frequency,
            ]
            indexb = 3 * end_vertex  # save the index of the outer field b
    """
    If the momentum flows into the vertex (= end_vertex), we assign to it a sign (+).
    If it flows out (from begin_vertex), we give it a sign (-).    
    """
    for propagator_key in internal_lines:
        line = internal_lines[propagator_key]
        begin_vertex = line[0][0]
        end_vertex = line[0][1]
        outflowing_field = line[1][0]
        inflowing_field = line[1][1]
        outflowing_data = [
            propagator_key,
            outflowing_field,
            -momentum_distribution_at_zero_external_momentum[propagator_key],
            -frequency_distribution_at_zero_external_frequency[propagator_key],
        ]
        inflowing_data = [
            propagator_key,
            inflowing_field,
            momentum_distribution_at_zero_external_momentum[propagator_key],
            frequency_distribution_at_zero_external_frequency[propagator_key],
        ]

        if data_for_vertexes_distribution[begin_vertex * 3] == 0:
            data_for_vertexes_distribution[begin_vertex * 3] = outflowing_data
        elif data_for_vertexes_distribution[begin_vertex * 3 + 1] == 0:
            data_for_vertexes_distribution[begin_vertex * 3 + 1] = outflowing_data
        else:
            data_for_vertexes_distribution[begin_vertex * 3 + 2] = outflowing_data

        if data_for_vertexes_distribution[end_vertex * 3] == 0:
            data_for_vertexes_distribution[end_vertex * 3] = inflowing_data
        elif data_for_vertexes_distribution[end_vertex * 3 + 1] == 0:
            data_for_vertexes_distribution[end_vertex * 3 + 1] = inflowing_data
        else:
            data_for_vertexes_distribution[end_vertex * 3 + 2] = inflowing_data

    # change the keywords in the dictionary from numeric to string (for convenience)
    frequency_and_momentum_distribution_at_vertexes = dict()
    for i in range(number_of_all_vertices):
        frequency_and_momentum_distribution_at_vertexes["vertex", i] = data_for_vertexes_distribution[
            3 * i : 3 * (i + 1)
        ]

    data_for_vertexes_momentum_distribution = [0] * (number_of_all_vertices * 3)
    for j in range(len(data_for_vertexes_distribution)):
        data_for_vertexes_momentum_distribution[j] = data_for_vertexes_distribution[j][0:3]

    return [
        indexB,
        indexb,
        data_for_vertexes_distribution,
        frequency_and_momentum_distribution_at_vertexes,
        data_for_vertexes_momentum_distribution,
    ]
