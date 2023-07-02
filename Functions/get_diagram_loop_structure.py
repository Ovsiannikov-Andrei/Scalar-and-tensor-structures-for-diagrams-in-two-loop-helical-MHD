import copy
import itertools
from collections import Counter

from Functions.Data_classes import *

# ------------------------------------------------------------------------------------------------------------------#
#                       We get a loop structure of the diagram (which lines form loops)
# ------------------------------------------------------------------------------------------------------------------#


def get_line_keywards_to_dictionary(some_dictionary: dict):
    """
    Turns the dictionary with digits keys to dictionary which string keys

    ARGUMENTS:

    some_dictionary -- dictionary with information about lines structure

    OUTPUT DATA EXAMPLE:

    {'line 0': [(0, 1), ['b', 'B']], 'line 1': [(0, 2), ['v', 'v']], 'line 2': [(1, 2), ['v', 'B']],
    'line 3': [(1, 3), ['b', 'b']], 'line 4': [(2, 3), ['b', 'V']]}
    """
    new_some_dictionary = copy.copy(some_dictionary)
    dim = len(new_some_dictionary)
    for i in range(dim):
        new_some_dictionary[f"line {i}"] = new_some_dictionary.pop(i)
    return new_some_dictionary


def get_all_possible_line_combinations(diagram_lines: InternalAndExternalLines):
    """
    Return all possible (in principle) combinations of lines (propagators).
    Each digit in output list = key from dict_with_internal_lines, i.e. line in diagram.

    ARGUMENTS:

    dict_with_internal_lines is defined by the functions get_list_with_propagators_from_nickel_index()
    and get_list_as_dictionary()

    OUTPUT DATA EXAMPLE: (digit corresponds to the line from dict_with_internal_lines):

    [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), (0, 1, 2),
    (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),
    (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4), (0, 1, 2, 3, 4)]

    """
    dict_with_internal_lines = diagram_lines.dict_internal_propagators

    list_of_loops = list()
    for i in range(len(dict_with_internal_lines) - 1):
        ordered_list_of_r_cardinality_subsets = list(itertools.combinations(dict_with_internal_lines.keys(), r=i + 2))
        # returns an ordered list of ordered subsets of the given set,
        # starting with subsets of cardinality 2
        [list_of_loops.append(x) for x in ordered_list_of_r_cardinality_subsets]
    return list_of_loops


def check_if_the_given_lines_combination_is_a_loop_in_diagram(
    diagram_lines: InternalAndExternalLines,
):
    """
    It checks if the given lines combination from list_of_all_possible_lines_combinations is a loop.
    The combination of lines is a loop <==> in the list of all vertices of the given lines
    (line = (vertex1, vertex2)), each vertex is repeated TWICE, i.e. each vertex is the end
    of the previous line and the start of the next one.

    ARGUMENTS:

    diagram_lines is given by the functions get_list_with_propagators_from_nickel_index()

    Note: for some technical reasons, we will assign new momentums (k and q, according to the list_of_momentums)
    to propagators containing the D_v kernel, i.e. to propagators_with_helicity. Since each loop in the diagram
    contains such helical propagator, we can describe the entire loop structure of the diagram by assigning a
    new momentum to it in each loop.

    OUTPUT DATA EXAMPLE: (digit corresponds to the line from diagram_lines):

    [(0, 1, 2), (2, 3, 4), (0, 1, 3, 4)]
    """
    list_of_all_possible_lines_combinations = get_all_possible_line_combinations(diagram_lines)

    dict_with_diagram_internal_lines = diagram_lines.dict_internal_propagators

    i = 0
    while i < len(list_of_all_possible_lines_combinations):
        list_of_list_of_vertices_for_ith_combination = [
            dict_with_diagram_internal_lines[k][0] for k in list_of_all_possible_lines_combinations[i]
        ]  # for a i-th combination from list_of_all_possible_lines_combinations we get a list of lines
        # (each digit from list_of_all_possible_lines_combinations is the key of the
        # dict_with_diagram_internal_lines, i.e. line)
        # the output is ((vertex,vertex), (vertex,vertex), ...)
        ordered_list_vith_all_diagram_vertices = list(
            itertools.chain.from_iterable(list_of_list_of_vertices_for_ith_combination)
        )  # converting a list of lists to a list of vertices
        list_with_number_of_occurrences = list(
            Counter(ordered_list_vith_all_diagram_vertices).values()
        )  # counting numbers of occurrences of the vertex in a list
        condition_to_be_a_loop = all(
            list_with_number_of_occurrences[i] == 2 for i in range(len(list_with_number_of_occurrences))
        )  # ith element of list_of_all_possible_lines_combinations is a loop <==>
        # each vertex in list_with_number_of_occurrences is repeated TWICE

        if condition_to_be_a_loop == True:
            i += 1  # this configuration give us a loop for the corresponding diagram
        else:
            del list_of_all_possible_lines_combinations[i]
    return list_of_all_possible_lines_combinations


def put_momentums_and_frequencies_to_propagators_with_helicity(
    diagram_lines: InternalAndExternalLines,
    list_of_propagators_with_helicity: list,
    list_of_momentums: list,
    list_of_frequencies: list,
):
    """
    It assigning momentum (according to the list_of_momentums) to helicity propagators in the concret diagram.
    This function uses the information that there can only be one helical propagator in each loop.

    ARGUMENTS:

    diagram_lines is given by the function get_list_with_propagators_from_nickel_index(),
    list_of_propagators_with_helicity, list_of_momentums, list_of_frequencies (see global variables)

    OUTPUT DATA EXAMPLE:

    dict_with_momentums_for_propagators_with_helicity = {1: k, 3: q}

    dict_with_frequencies_for_propagators_with_helicity = {1: w_k, 3: w_q}
    """
    list_of_all_internal_propagators = diagram_lines.dict_internal_propagators

    dict_with_momentums_for_propagators_with_helicity = dict()
    dict_with_frequencies_for_propagators_with_helicity = dict()
    for i in list_of_all_internal_propagators:
        vertices_and_fields_in_propagator = list_of_all_internal_propagators[i]
        # selected one particular propagator from the list
        fields_in_propagator = vertices_and_fields_in_propagator[1]
        # selected information about which fields define the propagator
        length = len(dict_with_momentums_for_propagators_with_helicity)
        # sequentially fill in the empty dictionary for corresponding diagram according to
        # list_of_momentums and set_of_propagators_with_helicity
        if fields_in_propagator in list_of_propagators_with_helicity:
            for j in range(len(list_of_momentums)):
                # len(list_of_momentums) = len(list_of_frequencies)
                if length == j:
                    dict_with_momentums_for_propagators_with_helicity.update({i: list_of_momentums[j]})
                    dict_with_frequencies_for_propagators_with_helicity.update({i: list_of_frequencies[j]})

    independent_args_in_helical_lines = IndependentMomentumsInHelicalPropagators(
        dict_with_momentums_for_propagators_with_helicity, dict_with_frequencies_for_propagators_with_helicity
    )
    return independent_args_in_helical_lines


def get_usual_QFT_loops(list_of_loops: list, dict_with_momentums_for_propagators_with_helicity: dict):
    """
    It selects from the list_of_loops only those that contain one heicity propagator
    (through which the momentum k or q flows), i.e. each new loop corresponds one new
    momentum and no more (we exclude loops in which the law of conservation of momentum does not hold)

    ARGUMENTS:

    list_of_loops is given by the function list_of_all_possible_lines_combinations()),

    dict_with_momentums_for_propagators_with_helicity (is given by the function

    put_momentums_and_frequencies_to_propagators_with_helicity())

    OUTPUT DATA EXAMPLE: (digit corresponds to the line):

    list_of_usual_QFT_loops = [(0, 1, 2), (2, 3, 4)]
    """

    i = 0
    list_of_usual_QFT_loops = copy.copy(list_of_loops)
    while i < len(list_of_usual_QFT_loops):
        test_loop = list_of_usual_QFT_loops[i]
        number_of_helicity_propagators = list(
            map(
                lambda x: test_loop.count(x),
                dict_with_momentums_for_propagators_with_helicity,
            )
        )  # calculate the number of helicity propagators in a loop
        if number_of_helicity_propagators.count(1) != 1:
            # delete those loops that contain two (and more) helical propagators
            # note that each loop in the diagram contains at least one such propagator
            del list_of_usual_QFT_loops[i]
        else:
            i += 1
    return list_of_usual_QFT_loops
