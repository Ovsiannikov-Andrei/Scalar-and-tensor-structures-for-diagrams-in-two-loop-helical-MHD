#!/usr/bin/python3

import os
import sys
import copy
import itertools
import sympy as sym
# from sympy import re, im, I, E, symbols
from sympy import *
from functools import reduce
from collections import Counter
import time

# -----------------------------------------------------------------------------------------------------------------#
#                                                 Global variables
# -----------------------------------------------------------------------------------------------------------------#

stupen = 1  # proportionality of the tensor structure to the external momentum p

number_int_vert = 4  # the number of internal (three-point) vertecies in diagram

vertex_begin = 0     # 
vertex_end = 3       # 

hyb = [p, k, q] = symbols("p k q")      # symbols for momentums (p denotes an external momentum)
[w, w_k, w_q] = symbols("w, w_k, w_q")  # symbols for frequencies (w denotes an external frequency)
P = Function("P")                       # Transverse projection operator
H = Function("H")                       # Helical term
kd = Function("kd")                     # Kronecker delta function
hyb = Function("hyb")                   # defines momentum (hybnost = momentum) as follows: hyb(k, 1) is $k_1$
lcs = Function("lcs")                   # Levi-Civita symbol

[I, A, z, nu, vo, uo, rho] = symbols("I A z nu vo uo rho")

# I is an imaginary unit
# A is a model parameter (the model action includes a term ~ A Bbv ).
# z ????
# nu ???
# v_0 ???
# u_0^(-1) is a "magnetic Prandtl number"
# rho is a gyrotropy parameter, |rho| < 1

[go, d, eps] = symbols("go d eps")  # coupling constant, dimension

# g_0 is a bare coupling constant
# d is a space dimension
# eps determines a degree of model deviation from logarithmicity

[s, b, B] = symbols("s b B")

# the index of field connected to field and the external momentum p_s ???

propagators_with_helicity = [["v", "v"], ["v", "b"], ["b", "v"], ["b", "b"]]

momentums_for_helicity_propagators = [k, q]
frequencies_for_helicity_propagators = [w_k, w_q]

# these propagators contain the kernel D_v and will determine the loops in the diagram 
# (for technical reasons, it is convenient for us to give these lines new momentums and frequencies
# (k, q and w_k, w_q)) first loop - (k, w_k), second loop - (q, w_q)

# -----------------------------------------------------------------------------------------------------------------#
#                                                Auxiliary functions
# -----------------------------------------------------------------------------------------------------------------#

#------------------------------------------------------------------------------------------------------------------#
#                    Create a file with a name and write the Nickel index of the diagram into it
#------------------------------------------------------------------------------------------------------------------#

def get_information_from_Nickel_index(
    graf
    ): # given Nickel index (graf) creates a file name with the output data

    Nickel_index = "".join(graf.split(sep = 'SC = ')[0])

    Symmetry_factor = " ".join(graf.split(sep = 'SC = ')[1])

    Nickel_topology = " ".join(graf.split(sep = 'SC = ')[0].split(sep = ":")[0].split(sep = "|"))
    # topological part of the Nickel index

    Nickel_lines = " ".join(graf.split(sep = 'SC = ')[0].split(sep = ":")[1].split(sep = "|"))
    # line structure in the diagram corresponding to Nickel_topology

    return [f"Diagram {Nickel_topology.strip()} {Nickel_lines.strip()}.txt", Nickel_index.strip(), Symmetry_factor.strip()]

# Nickel index example: e12|23|3|e|:0B_bB_vv|vB_bb|bV|0b|
# File name example: "Diagram e12 23 3 e 0B_bB_vv vB_bb bV 0b.txt" (all "|" are replaced by a space, ":" is removed)

def get_helical_propagators(
    fields_for_propagators_with_helicity
    ): # glues separate fields into propagators according to the propagators_with_helicity
    dimension = len(fields_for_propagators_with_helicity) 
    list_of_propagators_with_helicity = [0] * dimension
    for i in range(dimension):
        list_of_propagators_with_helicity[i] = (
            fields_for_propagators_with_helicity[i][0] + fields_for_propagators_with_helicity[i][1])
    return list_of_propagators_with_helicity
# Function application example: 
# list_of_propagators_with_helicity = ['vv', 'vb', 'bv', 'bb']

def get_list_with_propagators_from_nickel_index(
    nickel,
    ):  # arranges the propagators into a list of inner and outer lines with fields
    s1 = 0                  # numbers individual blocks |...| in the topological part of the Nickel index 
                            # (all before the symbol :), i.e. vertices of the diagram
    s2 = nickel.find(":")   # runs through the part of the Nickel index describing the lines (after the symbol :)
    propagator_internal = []
    propagator_external = []
    for i in nickel[: nickel.find(":")]:
        if i == "e":
            propagator_external += [[(-1, s1), ["0", nickel[s2 + 2]]]]
            s2 += 3
        elif i != "|":
            propagator_internal += [[(s1, int(i)), [nickel[s2 + 1], nickel[s2 + 2]]]]
            s2 += 3
        else:
            s1 += 1
    return [propagator_internal, propagator_external]

# Function application example: 
# propagator(e12|23|3|e|:0B_bB_vv|vB_bb|bV|0b|) = 
# [
# [[(0, 1), ['b', 'B']], [(0, 2), ['v', 'v']], [(1, 2), ['v', 'B']], [(1, 3), ['b', 'b']], [(2, 3), ['b', 'V']]],
# [[(-1, 0), ['0', 'B']], [(-1, 3), ['0', 'b']]]
# ]
# I.e. vertex 0 is connected to vertex 1 by a line b---B, vertex 0 is connected to vertex 2 by a line v---v, etc.

#------------------------------------------------------------------------------------------------------------------#
#                        Get a loop structure of the diagram (which lines form loops)
#------------------------------------------------------------------------------------------------------------------#

def get_list_as_dictionary(
    list
    ): # turns the list into a dictionary, keys are the numbers of the list elements
    dictionary = dict()
    for x in range(len(list)):
        dictionary.update(
            {x: list[x]}
        ) 
    return dictionary

def list_of_all_possible_lines_combinations(
    dict_with_internal_lines
    ):  # give us all possible combinations of lines (propagators) 
        # (each digit in output list = key from dict_with_internal_lines, i.e. line in diagram)
    list_of_loops = list()
    for i in range(len(dict_with_internal_lines) - 1):
        ordered_list_of_r_cardinality_subsets = list(
            itertools.combinations(dict_with_internal_lines.keys(), r = i + 2)
            ) 
            # returns an ordered list of ordered subsets of the given set, 
            # starting with subsets of cardinality 2
        [
            list_of_loops.append(x) for x in ordered_list_of_r_cardinality_subsets
        ]
    return list_of_loops 
# Function application example (digit corresponds to the line):
# list_of_loops = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4), 
# (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4), (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4), 
# (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4), (0, 1, 2, 3, 4)]

def check_if_the_given_lines_combination_is_a_loop_in_diagram(
    list_of_all_possible_lines_combinations, dict_with_diagram_internal_lines
    ):  # сheck if the given lines combination from list_of_all_possible_lines_combinations() is a loop
        # the combination of lines is a loop <==> in the list of all vertices of the given lines
        # (line = (vertex1, vertex2)), each vertex is repeated TWICE, i.e. each vertex is the end 
        # of the previous line and the start of the next one
    i = 0
    while i < len(list_of_all_possible_lines_combinations): 
        list_of_list_of_vertices_for_ith_combination = [
            dict_with_diagram_internal_lines[k][0] for k in list_of_all_possible_lines_combinations[i]
            ]   # for a i-th combination from list_of_all_possible_lines_combinations we get a list of lines 
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
            i += 1 # this configuration give us a loop for the corresponding diagram
        else:
            del list_of_all_possible_lines_combinations[i]
    return list_of_all_possible_lines_combinations
# Function application example (digit corresponds to the line):
# [(0, 1, 2), (2, 3, 4), (0, 1, 3, 4)]

# Important note: for some technical reasons, we will assign new momentums (k and q, 
# according to the list_of_momentums) to propagators containing the D_v kernel, i.e. to propagators_with_helicity.
# Since each loop in the diagram contains such helical propagator, we can describe the entire loop structure of the
# diagram by assigning a new momentum to it in each loop

def put_momentums_and_frequencies_to_propagators_with_helicity(
    set_of_all_internal_propagators, set_of_propagators_with_helicity, list_of_momentums, list_of_frequencies
    ): # assigning momentum (according to the list_of_momentums) to helicity propagators in concret diagram
       # this function uses the information that there can only be one helical propagator in each loop
    dict_with_momentums_for_propagators_with_helicity = dict()
    # assuming that the variable set_of_all_internal_propagators is given by the function
    # get_list_with_propagators_from_nickel_index()
    dict_with_frequencies_for_propagators_with_helicity = dict()
    for i in set_of_all_internal_propagators:  
        vertices_and_fields_in_propagator = set_of_all_internal_propagators[i]
        # selected one particular propagator from the list
        fields_in_propagator = vertices_and_fields_in_propagator[1]
        # selected information about which fields define the propagator
        length = len(dict_with_momentums_for_propagators_with_helicity)
        # sequentially fill in the empty dictionary for corresponding diagram according to 
        # list_of_momentums and set_of_propagators_with_helicity
        if fields_in_propagator in set_of_propagators_with_helicity:
            for j in range(len(list_of_momentums)):
                # len(list_of_momentums) = len(list_of_frequencies)
                if  length == j:
                    dict_with_momentums_for_propagators_with_helicity.update(
                        {i: list_of_momentums[j]}
                        )
                    dict_with_frequencies_for_propagators_with_helicity.update(
                        {i: list_of_frequencies[j]}
                        )

    return [dict_with_momentums_for_propagators_with_helicity,
            dict_with_frequencies_for_propagators_with_helicity]
# Function application example:
# dict_with_momentums_for_propagators_with_helicity = {1: k, 3: q}
# dict_with_frequencies_for_propagators_with_helicity = {1: w_k, 3: w_q}

def get_usual_QFT_loops(
    list_of_loops, dict_with_momentums_for_propagators_with_helicity
    ): # selects from the list of all possible loops of the diagram only those that contain one 
       # heicity propagator (through which the momentum k or q flow), i.e. each new loop corresponds
       # one new momentum and no more (exclude different exotic cases)
    i = 0 
    list_of_usual_QFT_loops = copy.copy(list_of_loops) 
    while i < len(list_of_usual_QFT_loops): 
        test_loop = list_of_usual_QFT_loops[i]
        number_of_helicity_propagators = list(
            map(lambda x: test_loop.count(x), dict_with_momentums_for_propagators_with_helicity)
            ) # calculate the number of helicity propagators in a loop
        if number_of_helicity_propagators.count(1) != 1:
            # delete those loops that contain two (and more) helical propagators
            # note that each loop in the diagram contains at least one such propagator
            del list_of_usual_QFT_loops[i] 
        else:
            i += 1
    return list_of_usual_QFT_loops
# Function application example (digit corresponds to the line):
# list_of_usual_QFT_loops = [(0, 1, 2), (2, 3, 4)]

#------------------------------------------------------------------------------------------------------------------#
#                       Get a distribution over momentums and frequencies flowing over lines
#------------------------------------------------------------------------------------------------------------------#

def get_momentum_and_frequency_distribution(internal_lines, momentums_in_helical_propagators, 
    frequencies_in_helical_propagators, external_momentum, external_frequency, begin_vertex, end_vertex
    ):

    length = len(internal_lines)

    momentums_for_all_propagators = [symbols(f'k_{i}') for i in range(length)]
    frequencies_for_all_propagators = [symbols(f'w_{i}') for i in range(length)]

    distribution_of_arbitrary_momentums = dict()
    distribution_of_arbitrary_frequencies = dict()

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

    for vertex in range(number_int_vert):
        if vertex == begin_vertex:
            momentum_conservation_law[vertex] += -external_momentum
            frequency_conservation_law[vertex] += -external_frequency
        elif vertex == end_vertex:
            momentum_conservation_law[vertex] += external_momentum
            frequency_conservation_law[vertex] += external_frequency
        for line_number in range(length):
            momentum = distribution_of_arbitrary_momentums[line_number]
            frequency = distribution_of_arbitrary_frequencies[line_number]
            line = internal_lines[line_number][0]
            if vertex in line:
                if line.index(vertex) % 2 == 0:
                    momentum_conservation_law[vertex] += momentum
                    frequency_conservation_law[vertex] += frequency
                else: 
                    momentum_conservation_law[vertex] += -momentum
                    frequency_conservation_law[vertex] += -frequency

    list_of_momentum_conservation_laws = [momentum_conservation_law[i] for i in range(number_int_vert)]
    list_of_arbitrary_momentums = [
        momentums_for_all_propagators[i] for i in range(length) if i not in momentums_in_helical_propagators
        ]
    list_of_frequency_conservation_laws = [frequency_conservation_law[i]for i in range(number_int_vert)]
    list_of_arbitrary_frequencies = [
        frequencies_for_all_propagators[i] for i in range(length) if i not in frequencies_in_helical_propagators
        ]

    list_of_arbitrary_momentums = list(filter(lambda x: x != 0, list_of_arbitrary_momentums))
    list_of_arbitrary_frequencies = list(filter(lambda x: x != 0, list_of_arbitrary_frequencies))

    define_arbitrary_momentums = sym.solve((list_of_momentum_conservation_laws), (list_of_arbitrary_momentums))
    define_arbitrary_frequencies = sym.solve((list_of_frequency_conservation_laws), (list_of_arbitrary_frequencies))

    momentum_distribution = dict()
    frequency_distribution = dict()

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
# Function application example:
# [{0: -k + p, 1: k, 2: -k + p - q, 3: q, 4: p - q}, {0: w - w_k, 1: w_k, 2: w - w_k - w_q, 3: w_q, 4: w - w_q}]

def get_momentum_and_frequency_distribution_at_zero_p_and_w(
    internal_lines, momentum_distribution, frequency_distribution, external_momentum, external_frequency,
    momentums_in_helical_propagators, frequencies_in_helical_propagators
    ):

    momentum_distribution_at_zero_external_momentum = dict()
    frequency_distribution_at_zero_external_frequency = dict()

    length = len(internal_lines)

    list_with_momentums = [0] * length
    list_with_frequencies = [0] * length

    for i in range(length):
        if i not in momentums_in_helical_propagators:
            list_with_momentums[i] += momentum_distribution[i].subs(external_momentum, 0)
            momentum_distribution_at_zero_external_momentum.update({i: list_with_momentums[i]})
        else:
            momentum_distribution_at_zero_external_momentum.update({i: momentums_in_helical_propagators[i]})
        if i not in frequencies_in_helical_propagators:
            list_with_frequencies[i] += frequency_distribution[i].subs(external_frequency, 0)
            frequency_distribution_at_zero_external_frequency.update({i: list_with_frequencies[i]})
        else:
            frequency_distribution_at_zero_external_frequency.update({i: frequencies_in_helical_propagators[i]})

    return [momentum_distribution_at_zero_external_momentum, frequency_distribution_at_zero_external_frequency]
# Function application example:
# [{0: -k, 1: k, 2: -k - q, 3: q, 4: -q}, {0: -w_k, 1: w_k, 2: -w_k - w_q, 3: w_q, 4: -w_q}]

def momentum_and_frequency_distribution_at_vertexes(
    external_lines, internal_lines, number_of_all_vertices, external_momentum, external_frequency,
    momentum_distribution_at_zero_external_momentum, frequency_distribution_at_zero_external_frequency
    ):

    data_for_vertexes_distribution = [0] * (number_of_all_vertices * 3)
    for line in external_lines:  # deploy external momentum
        end_vertex = line[0][1]
        outflowing_field = line[1][1]
        if outflowing_field == "B":
            data_for_vertexes_distribution[3 * end_vertex] = [
                -1, "B", external_momentum, external_frequency]
            indexB = 3 * end_vertex  # save the index of the external field b
        else:
            data_for_vertexes_distribution[3 * end_vertex] = [
                -1, outflowing_field, -external_momentum, -external_frequency]
            indexb = 3 * end_vertex  # save the index of the outer field b

    for propagator_key in internal_lines:  
        line = internal_lines[propagator_key]
        begin_vertex = line[0][0]
        end_vertex = line[0][1]
        inflowing_field = line[1][0]
        outflowing_field = line[1][1]
        inflowing_data = [
            propagator_key, inflowing_field, -momentum_distribution_at_zero_external_momentum[propagator_key],
            -frequency_distribution_at_zero_external_frequency[propagator_key]
            ]
        outflowing_data = [
            propagator_key, outflowing_field, momentum_distribution_at_zero_external_momentum[propagator_key],
            frequency_distribution_at_zero_external_frequency[propagator_key]
            ]
        if data_for_vertexes_distribution[begin_vertex * 3] == 0:
            data_for_vertexes_distribution[begin_vertex * 3] = inflowing_data
        elif data_for_vertexes_distribution[begin_vertex * 3 + 1] == 0:
            data_for_vertexes_distribution[begin_vertex * 3 + 1] = inflowing_data
        else:
            data_for_vertexes_distribution[begin_vertex * 3 + 2] = inflowing_data
        if data_for_vertexes_distribution[end_vertex * 3] == 0:
            data_for_vertexes_distribution[end_vertex * 3] = outflowing_data
        elif data_for_vertexes_distribution[end_vertex * 3 + 1] == 0:
            data_for_vertexes_distribution[end_vertex * 3 + 1] = outflowing_data
        else:
            data_for_vertexes_distribution[end_vertex * 3 + 2] = outflowing_data

    frequency_and_momentum_distribution_at_vertexes = dict()
    for i in range(number_of_all_vertices):
        frequency_and_momentum_distribution_at_vertexes['vertex', i] = data_for_vertexes_distribution[3*i:3*(i + 1)]

    data_for_vertexes_momentum_distribution = [0] * (number_of_all_vertices * 3)
    for j in range(len(data_for_vertexes_distribution)):
        data_for_vertexes_momentum_distribution[j] = data_for_vertexes_distribution[j][0:3]

    return [indexB, indexb, data_for_vertexes_distribution, 
    frequency_and_momentum_distribution_at_vertexes, data_for_vertexes_momentum_distribution]
# Function application example:
# indexB = 0
# indexb = 9
# data_for_vertexes_distribution = [
# [-1, 'B', p, w], [0, 'b', k, w_k], [1, 'v', -k, -w_k], [0, 'B', -k, -w_k], [2, 'v', k + q, w_k + w_q], 
# [3, 'b', -q, -w_q], [1, 'v', k, w_k], [2, 'B', -k - q, -w_k - w_q], [4, 'b', q, w_q], [-1, 'b', -p, -w], 
# [3, 'b', q, w_q], [4, 'V', -q, -w_q]]
# ]
# frequency_and_momentum_distribution_at_vertexes = {
# ('vertex', 0): [[-1, 'B', p, w], [0, 'b', k, w_k], [1, 'v', -k, -w_k]], ('vertex', 1): [[0, 'B', -k, -w_k],
# [2, 'v', k + q, w_k + w_q], [3, 'b', -q, -w_q]], ('vertex', 2): [[1, 'v', k, w_k], [2, 'B', -k - q, -w_k - w_q],
#  [4, 'b', q, w_q]], ('vertex', 3): [[-1, 'b', -p, -w], [3, 'b', q, w_q], [4, 'V', -q, -w_q]]
# }
# data_for_vertexes_momentum_distribution = [
# [-1, 'B', p], [0, 'b', k], [1, 'v', -k], [0, 'B', -k], [2, 'v', k + q], [3, 'b', -q], [1, 'v', k], 
# [2, 'B', -k - q], [4, 'b', q], [-1, 'b', -p], [3, 'b', q], [4, 'V', -q]
# ]

#------------------------------------------------------------------------------------------------------------------#
#                                       Do smth                                       
#------------------------------------------------------------------------------------------------------------------#

def define_propagator_product_numerator(
    P_data, H_data, numerator, space, fields_in_propagator, momentum_arg, frequency_arg, in1, in2
    ):
    projector_argument_list = P_data
    helical_argument_list = H_data
    product_of_tensor_operators = numerator
    interspace = space

    match fields_in_propagator:
        case ["v", "v"]:
            product_of_tensor_operators *= (P(momentum_arg, in1, in2)
            + I * rho * H(momentum_arg, in1, in2))
            interspace = (interspace + f"Pvv[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["v", "V"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + f"PvV[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["V", "v"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + f"PbB[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["b", "B"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + f"PbB[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["B", "b"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + f"PBb[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["v", "b"]:
            product_of_tensor_operators *= (P(momentum_arg, in1, in2)
            + I * rho * H(momentum_arg, in1, in2))
            interspace = (interspace + f"Pvb[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg,  in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["b", "v"]:
            product_of_tensor_operators *= (P(momentum_arg, in1, in2)
            + I * rho * H(momentum_arg, in1, in2))
            interspace = (interspace + f"Pbv[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["b", "b"]:
            product_of_tensor_operators *= (P(momentum_arg, in1, in2)
            + I * rho * H(momentum_arg, in1, in2))
            interspace = (interspace + f"Pbb[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["V", "b"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + "PVb[{momentum_arg}, {frequency_arg}]*") 
            projector_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["b", "V"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + f"PbV[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["B", "v"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + f"PBv[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case ["v", "B"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + f"PvB[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            return product_of_tensor_operators, projector_argument_list, helical_argument_list, interspace
        case _:
            return sys.exit("Nickel index contains unknown propagator type")

def get_propagator_product_numerator(
    product_of_tensor_operators, projector_argument_list, helical_argument_list, 
    propagator_product_for_Wolphram_Mathematica, distribution_of_momentums_over_vertices, 
    set_with_internal_lines, momentum_distribution_with_zero_p, frequency_distribution_with_zero_w, 
    define_propagator_product_numerator
    ):
    indexy = list(map(lambda x: x[0], distribution_of_momentums_over_vertices))
    for i in set_with_internal_lines:
        line = set_with_internal_lines[i]
        in1 = indexy.index(i)
        indexy[in1] = len(set_with_internal_lines)
        in2 = indexy.index(i)
        fields_in_propagator = line[1]
        all_structures_in_numerator = define_propagator_product_numerator(
            projector_argument_list, helical_argument_list, product_of_tensor_operators, 
            propagator_product_for_Wolphram_Mathematica, fields_in_propagator, 
            momentum_distribution_with_zero_p[i],
            frequency_distribution_with_zero_w[i], in1, in2
            )
        product_of_tensor_operators = all_structures_in_numerator[0]
        projector_argument_list = all_structures_in_numerator[1]
        helical_argument_list = all_structures_in_numerator[2]
        propagator_product_for_Wolphram_Mathematica = all_structures_in_numerator[3]
    return [product_of_tensor_operators, projector_argument_list, 
    helical_argument_list, propagator_product_for_Wolphram_Mathematica[:-1]]
# Function application example:
# product_of_tensor_operators = (I*rho*H(k, w_k, 2, 6) + P(k, w_k, 2, 6))*(I*rho*H(q, w_q, 5, 10) + 
# P(q, w_q, 5, 10))*P(-k, -w_k, 1, 3)*P(-q, -w_q, 8, 11)*P(-k - q, -w_k - w_q, 4, 7)
# projector_argument_list = [
# [-k, -w_k, 1, 3], [k, w_k, 2, 6], [-k - q, -w_k - w_q, 4, 7], [q, w_q, 5, 10], [-q, -w_q, 8, 11]
# ]
# helical_argument_list = [[k, w_k, 2, 6], [q, w_q, 5, 10]]
# propagator_product_for_Wolphram_Mathematica[:-1] = PbB[-k, -w_k]*Pvv[k, w_k]*PvB[-k - q, -w_k - w_q]*
# Pbb[q, w_q]*PbV[-q, -w_q]

def adding_vertex_factors_to_product_of_propagators(
    product_of_tensor_operators, Kronecker_delta_structure, momentum_structure, number_of_vertices, distribution_of_momentums_over_vertices
    ):

    ordered_list_of_fields_flowing_from_vertices = list(map(lambda x: x[1], distribution_of_momentums_over_vertices))  

    for vertex_number in range(number_of_vertices):  
        vertex_triple = ordered_list_of_fields_flowing_from_vertices[
            3 * vertex_number : 3 * (vertex_number + 1)]  # triple for vertex
        sorted_vertex_triple = sorted(vertex_triple, reverse=False) # ascending sort
        if sorted_vertex_triple == ["B", "b", "v",]:  
            in1 = 3 * vertex_number + vertex_triple.index("B")
            in2 = 3 * vertex_number + vertex_triple.index("b")
            in3 = 3 * vertex_number + vertex_triple.index("v")
            vertex_factor_Bbv = I * (
                hyb(distribution_of_momentums_over_vertices[in1][2], in3) * kd(in1, in2)
                - A * hyb(distribution_of_momentums_over_vertices[in1][2], in2) * kd(in1, in3))
            product_of_tensor_operators = product_of_tensor_operators * vertex_factor_Bbv
            Kronecker_delta_structure.append([in1, in2]) 
            Kronecker_delta_structure.append([in1, in3])
            momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3]) 
            momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])
        elif sorted_vertex_triple == ["V", "v", "v"]:
            in1 = 3 * vertex_number + vertex_triple.index("V")
            index_set = [3*vertex_number, 3*vertex_number + 1, 3*vertex_number + 2]
            index_set.remove(in1)
            in2 = index_set[0]
            in3 = index_set[1]
            vertex_factor_Vvv = I * (
                hyb(distribution_of_momentums_over_vertices[in1][2], in2) * kd(in1, in3)
                + hyb(distribution_of_momentums_over_vertices[in1][2], in3) * kd(in1, in2))
            product_of_tensor_operators = product_of_tensor_operators * vertex_factor_Vvv
            Kronecker_delta_structure.append([in1, in3])
            Kronecker_delta_structure.append([in1, in2])
            momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])
            momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3])
        elif sorted_vertex_triple == ["V", "b", "b"]:
            in1 = 3 * vertex_number + vertex_triple.index("V")
            index_set = [3*vertex_number, 3*vertex_number + 1, 3*vertex_number + 2]
            index_set.remove(in1)
            in2 = index_set[0]
            in3 = index_set[1]
            vertex_factor_Vbb = I * (
                hyb(distribution_of_momentums_over_vertices[in1][2], in2) * kd(in1, in3)
                + hyb(distribution_of_momentums_over_vertices[in1][2], in3) * kd(in1, in2)) 
            product_of_tensor_operators = product_of_tensor_operators * vertex_factor_Vbb
            Kronecker_delta_structure.append([in1, in3])
            Kronecker_delta_structure.append([in1, in2])
            momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in2])
            momentum_structure.append([distribution_of_momentums_over_vertices[in1][2], in3])
        else:
            sys.exit("Unknown vertex type")
    return product_of_tensor_operators, Kronecker_delta_structure, momentum_structure

#------------------------------------------------------------------------------------------------------------------#
#                                       Do smth                                       
#------------------------------------------------------------------------------------------------------------------#

def dosad(
    zoznam, ind_hod, struktura, pozicia
    ): # ?????????
    if ind_hod in zoznam:
        return zoznam
    elif ind_hod not in struktura:
        return list()
    elif zoznam[pozicia] != -1:
        return list()
    elif len(zoznam) - 1 == pozicia:
        return zoznam[:pozicia] + [ind_hod]
    else:
        return zoznam[:pozicia] + [ind_hod] + zoznam[pozicia + 1 :]

#------------------------------------------------------------------------------------------------------------------#
#                                       Define the main body of this program                                        
#------------------------------------------------------------------------------------------------------------------#

def get_output_data(
    graf
    ): 

    #--------------------------------------------------------------------------------------------------------------#
    #                    Create a file with a name and write the Nickel index of the diagram into it
    #--------------------------------------------------------------------------------------------------------------#

    output_file_name = get_information_from_Nickel_index(
        graf
        )[0] # according to the given Nickel index of the diagram, create the name of the file with the results
    nickel_index = get_information_from_Nickel_index(
        graf
        )[1] # get Nickel index from the line with the data
    symmetry_coefficient = get_information_from_Nickel_index(
        graf
        )[2] # get symmetry factor from the line with the data
    
    if not os.path.isdir("Results"):
        os.mkdir("Results") # create the Results folder if it doesn't already exist

    Fey_graphs = open(
        f"Results/{output_file_name}", "w"
        ) # creating a file with all output data for the corresponding diagram

    Fey_graphs.write(
        f"Nickel index of the Feynman diagram: {nickel_index} \n"
        ) # write the Nickel index to the file

    Fey_graphs.write(
        f"\nDiagram symmetry factor: {symmetry_coefficient} \n"
        ) # write the symmetry coefficient to the file
    
    Fey_graphs.write(
        f"\nNotation: \n"
        f"1. Fields: v is a random vector velocity field, b is a vector magnetic field, "
        "B and V are auxiliary vector fields (according to Janssen - De Dominicis approach)\n"
        f"2. Propagators: vv = <vv>, vB = <vB>, etc.\n"
        f"3. Momentums and frequencies: {p, w} denotes external momentum and frequency, "
        "{k, q} and {w_k, w_q} denote momentums and frequencies flowing along the loops in the diagram.\n"
        f"4. Loop structure: arguments {k, q} and {w_k, w_q} are always assigned to propagators containing" 
        f"the D_v kernel (so-called helical propagators): {get_helical_propagators(propagators_with_helicity)}\n"
        ) # write the list of used notation   

    #--------------------------------------------------------------------------------------------------------------#
    #                Define a loop structure of the diagram (which lines form loops) and write it into file
    #--------------------------------------------------------------------------------------------------------------#

    internal_lines = get_list_with_propagators_from_nickel_index(graf)[0]  # list with diagram internal lines

    dict_with_internal_lines = get_list_as_dictionary(
        internal_lines
        ) # put the list of all internal lines in the diagram to a dictionary

    Fey_graphs.write(
        f"\nPropagators in the diagram (digit key corresponds to the line): \n{dict_with_internal_lines} \n"
        ) # write the dictionary with all internal lines to the file

    list_of_all_loops_in_diagram = check_if_the_given_lines_combination_is_a_loop_in_diagram(
        list_of_all_possible_lines_combinations(dict_with_internal_lines), dict_with_internal_lines
        ) # get list of all loops in the diagram (this function works for diagrams with any number of loops) 

    momentums_in_helical_propagators = put_momentums_and_frequencies_to_propagators_with_helicity(
        dict_with_internal_lines, propagators_with_helicity, 
        momentums_for_helicity_propagators, frequencies_for_helicity_propagators
        )[0] # create a dictionary for momentums flowing in lines containing kernel D_v, i.e. in 
             # helicity propagators (hybnost == momentum)

    loop = get_usual_QFT_loops(
        list_of_all_loops_in_diagram, momentums_in_helical_propagators
        ) # select only those loops that contain only one helical propagator (usual QFT loops)

    Fey_graphs.write(
        f"\nLoops in the diagram for a given internal momentum (digit coresponds to the line): \n{loop} \n"
        ) # write the loop structure of the diagram to the file

    #--------------------------------------------------------------------------------------------------------------#
    #                      Get a distribution over momentums and frequencies flowing over lines
    #--------------------------------------------------------------------------------------------------------------#



    # The beginning of the momentum distribution. In this case, momentum flows into the diagram via field B and flows out through field b.
    # If the momentum flows into the vertex is with (+) if the outflow is with (-).
    # In propagator the line follows nickel index. For example line (1,2) is with (+) momentum and line (2,1) is with (-) momentum - starting point is propagators vv or propagators with kernel D_v

    frequencies_in_helical_propagators = put_momentums_and_frequencies_to_propagators_with_helicity(
        dict_with_internal_lines, propagators_with_helicity, 
        momentums_for_helicity_propagators, frequencies_for_helicity_propagators
        )[1]
    
    momentum_and_frequency_distribution = get_momentum_and_frequency_distribution(
        dict_with_internal_lines, momentums_in_helical_propagators, frequencies_in_helical_propagators, 
        p, w, vertex_begin, vertex_end
    )

    momentum_distribution = momentum_and_frequency_distribution[0]

    frequency_distribution = momentum_and_frequency_distribution[1]

    propagator_args_distribution_at_zero_p_and_w = get_momentum_and_frequency_distribution_at_zero_p_and_w(
    dict_with_internal_lines, momentum_distribution, frequency_distribution, p, w,
    momentums_for_helicity_propagators, frequencies_for_helicity_propagators
    )

    momentum_distribution_at_zero_external_momentum = propagator_args_distribution_at_zero_p_and_w[0]

    frequency_distribution_at_zero_external_frequency = propagator_args_distribution_at_zero_p_and_w[1]

    Fey_graphs.write(
        f"\nMomentum propagating along the lines (digit key coresponds to the line): \n{momentum_distribution}\n"
    )

    Fey_graphs.write(
        f"\nFrequency propagating along the lines (digit key coresponds to the line): \n{frequency_distribution}\n"
    )

    external_lines = get_list_with_propagators_from_nickel_index(graf)[1] # list with diagram external lines

    distribution_of_diagram_parameters_over_vertices = momentum_and_frequency_distribution_at_vertexes(
    external_lines, dict_with_internal_lines, number_int_vert, p, w,
    momentum_distribution_at_zero_external_momentum, frequency_distribution_at_zero_external_frequency
    )

    indexB =  distribution_of_diagram_parameters_over_vertices[0]

    indexb =  distribution_of_diagram_parameters_over_vertices[1]
    
    frequency_and_momentum_distribution_at_vertexes = distribution_of_diagram_parameters_over_vertices[3]

    moznost = distribution_of_diagram_parameters_over_vertices[4]

    Fey_graphs.write(
        f"\nMomentum and frequency distribution at the vertices:"
        f" \n{frequency_and_momentum_distribution_at_vertexes} \n"
    )  

    # [[index of propagator, field, momentum]]

    #--------------------------------------------------------------------------------------------------------------#
    #                      Do smth
    #--------------------------------------------------------------------------------------------------------------#


    # --------------------------------------------------------------------------------------
    # The previous part is the prepartion for the writing the structure from the diagram.

    # Tenzor = 1  # writing the tensor structure - this is the quantity where the tensor structure is constructed
    # I write into the Tenzor, the projection part of the propagators (vv, Vv = v'v, Bb = b'b )
    # indexy = list(map(lambda x: x[0], moznost))
    # P_structure = ([])  
    # I save the structures so that I don't have to guess through all possible combinations (faster running of the program) [ [momentum, index 1, index 2]]
    # H_structure = ([])  
    # I save the helical structures so that I don't have to guess through all possible combinations (faster running of the program) [ [momentum, index 1, index 2]]
    # H_{ij} (k) = \epsilon_{i, j, l} k_l/ k = H(k, i, j) - it is part with levi-civita symbol and momentum
    # P_{i, j} (k) = P(k, i, j)
    
    Tenzor = 1

    P_structure = ([])

    H_structure = ([])

    propagator_product_for_WfMath = ''

    tensor_structure_of_propagator_product_numerator = get_propagator_product_numerator(
        Tenzor, P_structure, H_structure, propagator_product_for_WfMath, moznost, dict_with_internal_lines, 
        momentum_distribution_at_zero_external_momentum, frequency_distribution_at_zero_external_frequency, 
        define_propagator_product_numerator)

    Tenzor = tensor_structure_of_propagator_product_numerator[0]

    P_structure = tensor_structure_of_propagator_product_numerator[1]

    H_structure = tensor_structure_of_propagator_product_numerator[2]

    propagator_product_for_WfMath = tensor_structure_of_propagator_product_numerator[3]

    Fey_graphs.write(
        f"\nPropagator product for the Wolfram Mathematica file: \n{propagator_product_for_WfMath}\n"
    )

    # I save the kronecker delta so that I don't have to guess through all possible combinations (faster running of the program) [ [index 1, index 2]] ... kd(index 1, index 2)
    # I save the momemntum and their index (faster running of the program) [ [ k, i] ] ... k_i = hyb(k, i) 
    # polia - the ordered list: [ V, v, v, b, V, v,... ] - the first three fields corespond the 0 vertex
    # part to add vertices - all vertieces have
    # B_i*v_j*Bbv_{ijl}b_l, Bbv_{ijl} = I*(k_j*delta_{ij} - A*k_l*delta_{ij})  
    # Vvv = Vbb

    kd_structure = ([])
    hyb_structure = ([])

    whole_tensor_structure_of_integrand_numerator = adding_vertex_factors_to_product_of_propagators(
        Tenzor, kd_structure, hyb_structure, number_int_vert, moznost)
    
    Tenzor = whole_tensor_structure_of_integrand_numerator[0]

    kd_structure = whole_tensor_structure_of_integrand_numerator[1]
    
    hyb_structure = whole_tensor_structure_of_integrand_numerator[2]
    
    # ----------------------------------------------------------------------------------------------
    # The program start here. The previous part is only for the reason that I don't have to write the whole structure from the diagram.
    t = time.time()  # it is only used to calculate the calculation time -- can be omitted

    Fey_graphs.write(
        f"\nTensor structure of the diagram before calculation: \n{Tenzor} \n"
    )

    print(f"{Tenzor}\n")

    Tenzor = expand(Tenzor)  # The final tesor structure from the diagram.

    Tenzor = rho * Tenzor.coeff(rho**stupen)  # What I need from the Tenzor structure
    Tenzor = expand(Tenzor.subs(I**5, I))  # calculate the imaginary unit
    # Tenzor = Tenzor.subs(A, 1)              # It depends on which part we want to calculate from the vertex Bbv
    # print(Tenzor)

    print("step 0:", round(time.time() - t, 1), "sec")


    for in2 in kd_structure:
        structurep = list()
        for (
            in1
        ) in (
            P_structure
        ):  # calculation via Kronecker's delta function: P(k, i, j) kd(i, l) = P(k, l, j)
            if in1[1] == in2[0]:
                Tenzor = Tenzor.subs(
                    P(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                    P(in1[0], in2[1], in1[2]),
                )
                structurep.append([in1[0], in2[1], in1[2]])
            elif in1[1] == in2[1]:
                Tenzor = Tenzor.subs(
                    P(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                    P(in1[0], in2[0], in1[2]),
                )
                structurep.append([in1[0], in2[0], in1[2]])
            elif in1[2] == in2[0]:
                Tenzor = Tenzor.subs(
                    P(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                    P(in1[0], in1[1], in2[1]),
                )
                structurep.append([in1[0], in1[1], in2[1]])
            elif in1[2] == in2[1]:
                Tenzor = Tenzor.subs(
                    P(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                    P(in1[0], in1[1], in2[0]),
                )
                structurep.append([in1[0], in1[1], in2[0]])
            if Tenzor.coeff(kd(in2[0], in2[1])) == 0:
                # del kd_structure[0] # it deletes the kronecker delta from the list if it is no longer in the tensor structure
                break
        P_structure = (
            P_structure + structurep
        )  # it adds all newly created structures to the list
        structureh = list()
        for (
            in1
        ) in (
            H_structure
        ):  # calculation via Kronecker's delta function: H(k, i, j) kd(i, l) = H(k, l, j)
            if in1[1] == in2[0]:
                Tenzor = Tenzor.subs(
                    H(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                    H(in1[0], in2[1], in1[2]),
                )
                structureh.append([in1[0], in2[1], in1[2]])
            elif in1[1] == in2[1]:
                Tenzor = Tenzor.subs(
                    H(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                    H(in1[0], in2[0], in1[2]),
                )
                structureh.append([in1[0], in2[0], in1[2]])
            elif in1[2] == in2[0]:
                Tenzor = Tenzor.subs(
                    H(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                    H(in1[0], in1[1], in2[1]),
                )
                structureh.append([in1[0], in1[1], in2[1]])
            elif in1[2] == in2[1]:
                Tenzor = Tenzor.subs(
                    H(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                    H(in1[0], in1[1], in2[0]),
                )
                structureh.append([in1[0], in1[1], in2[0]])
            if Tenzor.coeff(kd(in2[0], in2[1])) == 0:
                # del kd_structure[0]
                break
        H_structure = H_structure + structureh


    print("step 1:", round(time.time() - t, 1), "sec")

    i = 0
    while i < len(P_structure):  # discard from the Tensor structure what is zero for the projection operator P_ij (k) * k_i = 0
        in1 = P_structure[i]
        if Tenzor.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[2]), 0)
        if Tenzor.coeff(P(in1[0], in1[1], in1[2])) == 0:
            P_structure.remove(in1)
        else:
            if (
                in1[0] == -k or in1[0] == -q
            ):  # Replace in the tensor structure in the projection operators:  P(-k,i,j) = P(k,i,j)
                Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]), P(-in1[0], in1[1], in1[2]))
                P_structure[i][0] = -in1[0]
            i += 1

    i = 0
    while i < len(H_structure):  # discard from the Tensor structure what is zero for the helical operator H_ij (k) * k_i = 0
        in1 = H_structure[i]
        if Tenzor.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[2]), 0)
        if Tenzor.coeff(H(in1[0], in1[1], in1[2])) == 0:
            H_structure.remove(in1)
        else:
            i += 1

    print("step 2:", round(time.time() - t, 1), "sec")

    i = 0
    while (len(H_structure) > i):  # sipmplify in the Tenzor part H_{ij} (k) P_{il} (k) =  H_{il} (k)
        in1 = H_structure[i]
        for in2 in P_structure:
            if (
                in1[0] == in2[0]
                and Tenzor.coeff(H(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2])) != 0
            ):
                if in1[1] == in2[1]:
                    Tenzor = Tenzor.subs(
                        H(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        H(in1[0], in2[2], in1[2]),
                    )
                    H_structure += [[in1[0], in2[2], in1[2]]]
                elif in1[1] == in2[2]:
                    Tenzor = Tenzor.subs(
                        H(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        H(in1[0], in2[1], in1[2]),
                    )
                    H_structure += [[in1[0], in2[1], in1[2]]]
                elif in1[2] == in2[1]:
                    Tenzor = Tenzor.subs(
                        H(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        H(in1[0], in1[1], in2[2]),
                    )
                    H_structure += [[in1[0], in1[1], in2[2]]]
                elif in1[2] == in2[2]:
                    Tenzor = Tenzor.subs(
                        H(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        H(in1[0], in1[1], in2[1]),
                    )
                    H_structure += [[in1[0], in1[1], in2[1]]]
        if Tenzor.coeff(H(in1[0], in1[1], in1[2])) == 0:
            H_structure.remove(in1)
        else:
            i += 1

    print("step 3:", round(time.time() - t, 1), "sec")

    i = 0
    while (len(P_structure) > i):  # sipmplify in the Tenzor part  P_{ij} (k) P_{il} (k) =  P_{il} (k)
        in1 = P_structure[i]
        structurep = list()
        for j in range(i + 1, len(P_structure)):
            in2 = P_structure[j]
            if (
                in1[0] == in2[0]
                and Tenzor.coeff(P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2])) != 0
            ):
                if in1[1] == in2[1]:
                    Tenzor = Tenzor.subs(
                        P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        P(in1[0], in2[2], in1[2]),
                    )
                    structurep.append([in1[0], in2[2], in1[2]])
                elif in1[1] == in2[2]:
                    Tenzor = Tenzor.subs(
                        P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        P(in1[0], in2[1], in1[2]),
                    )
                    structurep.append([in1[0], in2[1], in1[2]])
                elif in1[2] == in2[1]:
                    Tenzor = Tenzor.subs(
                        P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        P(in1[0], in1[1], in2[2]),
                    )
                    structurep.append([in1[0], in1[1], in2[2]])
                elif in1[2] == in2[2]:
                    Tenzor = Tenzor.subs(
                        P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        P(in1[0], in1[1], in2[1]),
                    )
                    structurep.append([in1[0], in1[1], in2[1]])
        if Tenzor.coeff(P(in1[0], in1[1], in1[2])) == 0:
            P_structure.remove(in1)
        else:
            i += 1
        P_structure = (
            P_structure + structurep
        )  # it add all newly created structures to the list

    print("step 4:", round(time.time() - t, 1), "sec")

    for i in hyb_structure:  # replace: hyb(-k+q, i) = -hyb(k, i) + hyb(q, i)
        k_c = i[0].coeff(k)
        q_c = i[0].coeff(q)
        if k_c != 0 or q_c != 0:
            Tenzor = Tenzor.subs(hyb(i[0], i[1]), (k_c * hyb(k, i[1]) + q_c * hyb(q, i[1])))


    kd_structure = list()
    for (i) in (P_structure):  # Define transverse projection operator P(k,i,j) = kd(i,j) - hyb(k,i)*hyb(k,j)/k^2
        k_c = i[0].coeff(k)
        q_c = i[0].coeff(q)
        Tenzor = Tenzor.subs(
            P(i[0], i[1], i[2]),
            kd(i[1], i[2])
            - (k_c * hyb(k, i[1]) + q_c * hyb(q, i[1]))
            * (k_c * hyb(k, i[2]) + q_c * hyb(q, i[2]))
            / (k_c**2 * k**2 + q_c**2 * q**2 + 2 * k_c * q_c * k * q * z),
        )
        kd_structure.append([i[1], i[2]])

    print("step 5:", round(time.time() - t, 1), "sec")

    Tenzor = expand(Tenzor)

    for (in1) in (H_structure):  # discard from the Tensor structure what is zero for the helical operator H_{ij} (k) * k_i = 0
        clen = Tenzor.coeff(H(in1[0], in1[1], in1[2]))
        if clen.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        if clen.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        if in1[0] == k and clen.coeff(hyb(q, in1[1]) * hyb(q, in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(q, in1[1]) * hyb(q, in1[2]), 0
            )
        if in1[0] == q and clen.coeff(hyb(k, in1[1]) * hyb(k, in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(k, in1[1]) * hyb(k, in1[2]), 0
            )

    print("step 6:", round(time.time() - t, 1), "sec")

    inkd = 0
    while (inkd == 0):  # calculation part connected with the kronecker delta function: kd(i,j) *hyb(k,i) = hyb(k,j)
        for (
            in1
        ) in (
            kd_structure
        ):  # beware, I not treat the case if there remains a delta function with indexes of external fields !!
            clen = Tenzor.coeff(kd(in1[0], in1[1]))
            if clen.coeff(hyb(k, in1[0])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(k, in1[0]), hyb(k, in1[1]))
            if clen.coeff(hyb(k, in1[1])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(k, in1[1]), hyb(k, in1[0]))
            if clen.coeff(hyb(q, in1[0])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(q, in1[0]), hyb(q, in1[1]))
            if clen.coeff(hyb(q, in1[1])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(q, in1[1]), hyb(q, in1[0]))
            if clen.coeff(hyb(p, in1[0])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(p, in1[0]), hyb(p, in1[1]))
            if clen.coeff(hyb(p, in1[1])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(p, in1[1]), hyb(p, in1[0]))
            if Tenzor.coeff(kd(in1[0], in1[1])) == 0:
                kd_structure.remove(in1)
                inkd += 1
        if inkd != 0:
            inkd = 0
        else:
            inkd = 1

    print("step 7:", round(time.time() - t, 1), "sec")

    i = 0
    while len(H_structure) > i:  # calculation for helical term
        in1 = H_structure[i]
        clen = Tenzor.coeff(H(in1[0], in1[1], in1[2]))
        if (
            clen.coeff(hyb(in1[0], in1[1])) != 0
        ):  # I throw out the part:  H (k,i,j) hyb(k,i) = 0
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        if clen.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        if (
            in1[0] == k and clen.coeff(hyb(q, in1[1]) * hyb(q, in1[2])) != 0
        ):  # I throw out the part:  H (k,i,j) hyb(q,i) hyb(q, j) = 0
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(q, in1[1]) * hyb(q, in1[2]), 0
            )
        if in1[0] == q and clen.coeff(hyb(k, in1[1]) * hyb(k, in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(k, in1[1]) * hyb(k, in1[2]), 0
            )
        for (
            in2
        ) in (
            kd_structure
        ):  # it puts together the Kronecker delta and the helical term: H(k,i,j)*kd(i,l) = H(k,l,j)
            if clen.coeff(kd(in2[0], in2[1])) != 0:
                if in1[1] == in2[0]:
                    Tenzor = Tenzor.subs(
                        H(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                        H(in1[0], in2[1], in1[2]),
                    )
                    if [in1[0], in2[1], in1[2]] is not H_structure:
                        H_structure.append([in1[0], in2[1], in1[2]])
                elif in1[1] == in2[1]:
                    Tenzor = Tenzor.subs(
                        H(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                        H(in1[0], in2[0], in1[2]),
                    )
                    if [in1[0], in2[1], in1[2]] is not H_structure:
                        H_structure.append([in1[0], in2[0], in1[2]])
                elif in1[2] == in2[0]:
                    Tenzor = Tenzor.subs(
                        H(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                        H(in1[0], in1[1], in2[1]),
                    )
                    if [in1[0], in2[1], in1[2]] is not H_structure:
                        H_structure.append([in1[0], in1[1], in2[1]])
                elif in1[2] == in2[1]:
                    Tenzor = Tenzor.subs(
                        H(in1[0], in1[1], in1[2]) * kd(in2[0], in2[1]),
                        H(in1[0], in1[1], in2[0]),
                    )
                    if [in1[0], in2[1], in1[2]] is not H_structure:
                        H_structure.append([in1[0], in1[1], in2[0]])
        for in2 in kd_structure:
            if Tenzor.coeff(kd(in2[0], in2[1])) == 0:
                kd_structure.remove(in2)
        i += 1

    print("step 8:", round(time.time() - t, 1), "sec")

    p_structure = list()  # list of indeces for momentum p in Tenzor
    k_structure = list()  # list of indeces for momentum k in Tenzor
    q_structure = list()  # list of indeces for momentum q in Tenzor
    for in1 in range(len(moznost)):  # It combines quantities with matching indices.
        Tenzor = Tenzor.subs(hyb(k, in1) ** 2, k**2)
        Tenzor = Tenzor.subs(hyb(q, in1) ** 2, q**2)
        Tenzor = Tenzor.subs(
            hyb(q, in1) * hyb(k, in1), k * q * z
        )  # k.q = k q z, where z = cos(angle) = k . q/ |k| /|q|
        if (
            Tenzor.coeff(hyb(p, in1)) != 0
        ):  # H( , j, s) hyb( ,j) hyb( ,s) hyb( , indexb) hyb(p, i) hyb(q, i) hyb(q, indexB) = 0 or   H( , j, indexb) hyb( ,j) hyb(p, i) hyb(q, i) hyb(q, indexB) = 0
            Tenzor = Tenzor.subs(hyb(p, in1) * hyb(q, in1) * hyb(q, indexB), 0)
            Tenzor = Tenzor.subs(hyb(p, in1) * hyb(k, in1) * hyb(k, indexB), 0)
            Tenzor = Tenzor.subs(hyb(p, in1) * hyb(q, in1) * hyb(q, indexb), 0)
            Tenzor = Tenzor.subs(hyb(p, in1) * hyb(k, in1) * hyb(k, indexb), 0)
            p_structure += [in1]
        if Tenzor.coeff(hyb(q, in1)) != 0:
            q_structure += [in1]
        if Tenzor.coeff(hyb(k, in1)) != 0:
            k_structure += [in1]

    Tenzor = Tenzor.subs(
        hyb(q, indexb) * hyb(q, indexB), 0
    )  # delete zero values in the Tenzor: H( ,i,j) hyb(p, i) hyb( ,j) hyb(k, indexB) hyb(k, indexb) = 0
    Tenzor = Tenzor.subs(hyb(k, indexb) * hyb(k, indexB), 0)


    print("step 9:", round(time.time() - t, 1), "sec")

    # calculation of H structure - For this particular case, one of the external indices (p_s, b_i or B_j) is paired with a helicity term.
    # we will therefore use the information that, in addition to the helicity term H( , i,j), they can be multiplied by a maximum of three internal momenta.
    # For examle: H(k, indexb, j) hyb(q, j) hyb(k, indexB) hyb(q, i) hyb(p, i) and thus in this step I will calculate all possible combinations for this structure.
    # In this case, helical term H(k, i, j) = epsilon(i,j,s) k_s /k

    i = 0
    while i < len(H_structure):  # I go through all of them helical term H( , , )
        in1 = H_structure[i]
        while (
            Tenzor.coeff(H(in1[0], in1[1], in1[2])) == 0
        ):  # if the H( , , ) structure is no longer in the Tenzor, I throw it away
            H_structure.remove(in1)
            in1 = H_structure[i]
        if (
            in1[0] == k
        ):  # it create a list where momenta are stored in the positions and indexes pf momenta. - I have for internal momenta k or q
            kombinacia = in1 + [
                q,
                -1,
                p,
                -1,
                k,
                -1,
                q,
                -1,
            ]  # [ k, indexH, indexH, q, -1, p, -1,  k, -1, q, -1 ]
        else:
            kombinacia = in1 + [k, -1, p, -1, k, -1, q, -1]
        if (
            indexB == in1[1]
        ):  # it looks for whether the H helicity term contains an idex corresponding to the externa field b or B
            kombinacia[4] = in1[2]
        elif indexB == in1[2]:
            kombinacia[4] = in1[1]
        elif indexb == in1[1]:
            kombinacia[4] = in1[2]
        elif indexb == in1[2]:
            kombinacia[4] = in1[1]
        kombinacia_old = [
            kombinacia
        ]  # search whether the index B or b is in momenta not associated with the helicity term
        kombinacia_new = list()
        kombinacia_new.append(
            dosad(kombinacia_old[0], indexB, k_structure, 8)
        )  # it create and put the field index B in to the list on the position 8: hyb(k,indexB)
        kombinacia = dosad(
            kombinacia_old[0], indexB, q_structure, 10
        )  # it create and put the field index B in to the list on the position 10: hyb(q,indexB)
        if kombinacia not in kombinacia_new:
            kombinacia_new.append(kombinacia)
        kombinacia_old = kombinacia_new
        kombinacia_new = list()
        for (
            in2
        ) in (
            kombinacia_old
        ):  # # it create and put the field index b in to the list with index
            kombinacia_new.append(dosad(in2, indexb, k_structure, 8))
            kombinacia = dosad(in2, indexb, q_structure, 10)
            if kombinacia not in kombinacia_new:
                kombinacia_new.append(kombinacia)
            if list() in kombinacia_new:
                kombinacia_new.remove(list())
        kombinacia_old = kombinacia_new
        kombinacia_new = (
            list()
        )  #  I know which indexes are free. I know where the fields B or b are located.
        for (
            in2
        ) in (
            kombinacia_old
        ):  # I have free two indecies and I start summing in the tensor structure
            if (
                in2[4] == -1 and in2[0] == k
            ):  # it calculate if there is H(k,...,...) and the indecies of the external fields are outside
                if (
                    in2[1] in p_structure and in2[2] in q_structure
                ):  # H(k, i, j) hyb(q, j) hyb(p, i) hyb(k, indexb) hyb(q, indexB) = ... or  H(k, i, j) hyb(q, j) hyb(p, i) hyb(k, indexB) hyb(q, indexb)
                    Tenzor = Tenzor.subs(
                        H(k, in2[1], in2[2])
                        * hyb(q, in2[2])
                        * hyb(p, in2[1])
                        * hyb(k, in2[8])
                        * hyb(q, in2[10]),
                        hyb(p, s)
                        * lcs(s, in2[10], in2[8])
                        * q**2
                        * k
                        * (1 - z**2)
                        / d
                        / (d + 2),
                    )
                if in2[2] in p_structure and in2[1] in q_structure:
                    Tenzor = Tenzor.subs(
                        H(k, in2[1], in2[2])
                        * hyb(q, in2[1])
                        * hyb(p, in2[2])
                        * hyb(k, in2[8])
                        * hyb(q, in2[10]),
                        -hyb(p, s)
                        * lcs(s, in2[10], in2[8])
                        * q**2
                        * k
                        * (1 - z**2)
                        / d
                        / (d + 2),
                    )
            if in2[4] == -1 and in2[0] == q:  #
                if in2[1] in p_structure and in2[2] in k_structure:
                    Tenzor = Tenzor.subs(
                        H(q, in2[1], in2[2])
                        * hyb(k, in2[2])
                        * hyb(p, in2[1])
                        * hyb(k, in2[8])
                        * hyb(q, in2[10]),
                        -hyb(p, s)
                        * lcs(s, in2[10], in2[8])
                        * q
                        * k**2
                        * (1 - z**2)
                        / d
                        / (d + 2),
                    )
                if in2[2] in p_structure and in2[1] in k_structure:
                    Tenzor = Tenzor.subs(
                        H(q, in2[1], in2[2])
                        * hyb(k, in2[1])
                        * hyb(p, in2[2])
                        * hyb(k, in2[8])
                        * hyb(q, in2[10]),
                        hyb(p, s)
                        * lcs(s, in2[10], in2[8])
                        * q
                        * k**2
                        * (1 - z**2)
                        / d
                        / (d + 2),
                    )
            if (
                in2[8] == -1 and in2[0] == k
            ):  # # H(k, indexb, j) hyb(q, j) hyb(p, i) hyb(k, i) hyb(q, indexB) = ... or  H(k, indexB, j) hyb(q, j) hyb(p, i) hyb(k, i) hyb(q, indexb)
                for in3 in p_structure:
                    if in2[1] == indexb or in2[1] == indexB:
                        Tenzor = Tenzor.subs(
                            H(k, in2[1], in2[2])
                            * hyb(q, in2[2])
                            * hyb(p, in3)
                            * hyb(k, in3)
                            * hyb(q, in2[10]),
                            -hyb(p, s)
                            * lcs(s, in2[10], in2[1])
                            * q**2
                            * k
                            * (1 - z**2)
                            / d
                            / (d + 2),
                        )
                    if in2[2] == indexb or in2[2] == indexB:
                        Tenzor = Tenzor.subs(
                            H(k, in2[1], in2[2])
                            * hyb(q, in2[1])
                            * hyb(p, in3)
                            * hyb(k, in3)
                            * hyb(q, in2[10]),
                            hyb(p, s)
                            * lcs(s, in2[10], in2[2])
                            * q**2
                            * k
                            * (1 - z**2)
                            / d
                            / (d + 2),
                        )
            if in2[8] == -1 and in2[0] == q:
                for in3 in p_structure:
                    if in2[1] == indexb or in2[1] == indexB:
                        Tenzor = Tenzor.subs(
                            H(q, in2[1], in2[2])
                            * hyb(k, in2[2])
                            * hyb(p, in3)
                            * hyb(k, in3)
                            * hyb(q, in2[10]),
                            hyb(p, s)
                            * lcs(s, in2[10], in2[1])
                            * q
                            * k**2
                            * (1 - z**2)
                            / d
                            / (d + 2),
                        )
                    if in2[2] == indexb or in2[2] == indexB:
                        Tenzor = Tenzor.subs(
                            H(q, in2[1], in2[2])
                            * hyb(k, in2[1])
                            * hyb(p, in3)
                            * hyb(k, in3)
                            * hyb(q, in2[10]),
                            -hyb(p, s)
                            * lcs(s, in2[10], in2[2])
                            * q
                            * k**2
                            * (1 - z**2)
                            / d
                            / (d + 2),
                        )
            if in2[10] == -1 and in2[0] == k:
                for in3 in p_structure:
                    if in2[1] == indexb or in2[1] == indexB:
                        Tenzor = Tenzor.subs(
                            H(k, in2[1], in2[2])
                            * hyb(q, in2[2])
                            * hyb(p, in3)
                            * hyb(k, in2[8])
                            * hyb(q, in3),
                            -hyb(p, s)
                            * lcs(s, in2[1], in2[8])
                            * q**2
                            * k
                            * (1 - z**2)
                            / d
                            / (d + 2),
                        )
                    if in2[2] == indexb or in2[2] == indexB:
                        Tenzor = Tenzor.subs(
                            H(k, in2[1], in2[2])
                            * hyb(q, in2[1])
                            * hyb(p, in3)
                            * hyb(k, in2[8])
                            * hyb(q, in3),
                            hyb(p, s)
                            * lcs(s, in2[2], in2[8])
                            * q**2
                            * k
                            * (1 - z**2)
                            / d
                            / (d + 2),
                        )
            if in2[10] == -1 and in2[0] == q:
                for in3 in p_structure:
                    if in2[1] == indexb or in2[1] == indexB:
                        Tenzor = Tenzor.subs(
                            H(q, in2[1], in2[2])
                            * hyb(k, in2[2])
                            * hyb(k, in2[8])
                            * hyb(p, in3)
                            * hyb(q, in3),
                            hyb(p, s)
                            * lcs(s, in2[1], in2[8])
                            * q
                            * k**2
                            * (1 - z**2)
                            / d
                            / (d + 2),
                        )
                    if in2[2] == indexb or in2[2] == indexB:
                        Tenzor = Tenzor.subs(
                            H(q, in2[1], in2[2])
                            * hyb(k, in2[1])
                            * hyb(k, in2[8])
                            * hyb(p, in3)
                            * hyb(q, in3),
                            -hyb(p, s)
                            * lcs(s, in2[2], in2[8])
                            * q
                            * k**2
                            * (1 - z**2)
                            / d
                            / (d + 2),
                        )
        i += 1


    print("step 10:", round(time.time() - t, 1), "sec")

    for (in1) in (H_structure):  # calculate the structure where there are two external momentums: H(momentum, i, indexB)* p(i) hyb( , indexb) and other combinations except H(momentum, indexB, indexb) hyb(p, i) hyb(k, i)
        if Tenzor.coeff(H(in1[0], in1[1], in1[2])) != 0:
            if in1[1] in p_structure and in1[2] == indexb:
                if in1[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, in1[1], in1[2]) * hyb(p, in1[1]) * hyb(k, indexB),
                        hyb(p, s) * lcs(s, indexb, indexB) * k / d,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, in1[1], in1[2]) * hyb(p, in1[1]) * hyb(q, indexB),
                        hyb(p, s) * lcs(s, indexb, indexB) * q * z / d,
                    )
                else:
                    Tenzor = Tenzor.subs(
                        H(q, in1[1], in1[2]) * hyb(p, in1[1]) * hyb(q, indexB),
                        hyb(p, s) * lcs(s, indexb, indexB) * q / d,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, in1[1], in1[2]) * hyb(p, in1[1]) * hyb(k, indexB),
                        hyb(p, s) * lcs(s, indexb, indexB) * k * z / d,
                    )
            if in1[2] in p_structure and in1[1] == indexb:
                if in1[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, in1[1], in1[2]) * hyb(p, in1[2]) * hyb(k, indexB),
                        -hyb(p, s) * lcs(s, indexb, indexB) * k / d,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, in1[1], in1[2]) * hyb(p, in1[2]) * hyb(q, indexB),
                        -hyb(p, s) * lcs(s, indexb, indexB) * q * z / d,
                    )
                else:
                    Tenzor = Tenzor.subs(
                        H(q, in1[1], in1[2]) * hyb(p, in1[2]) * hyb(q, indexB),
                        -hyb(p, s) * lcs(s, indexb, indexB) * q / d,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, in1[1], in1[2]) * hyb(p, in1[2]) * hyb(k, indexB),
                        -hyb(p, s) * lcs(s, indexb, indexB) * k * z / d,
                    )
            if in1[1] in p_structure and in1[2] == indexB:
                if in1[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, in1[1], in1[2]) * hyb(p, in1[1]) * hyb(k, indexb),
                        -hyb(p, s) * lcs(s, indexb, indexB) * k / d,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, in1[1], in1[2]) * hyb(p, in1[1]) * hyb(q, indexb),
                        -hyb(p, s) * lcs(s, indexb, indexB) * q * z / d,
                    )
                else:
                    Tenzor = Tenzor.subs(
                        H(q, in1[1], in1[2]) * hyb(p, in1[1]) * hyb(q, indexb),
                        -hyb(p, s) * lcs(s, indexb, indexB) * q / d,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, in1[1], in1[2]) * hyb(p, in1[1]) * hyb(k, indexb),
                        -hyb(p, s) * lcs(s, indexb, indexB) * k * z / d,
                    )
            if in1[2] in p_structure and in1[1] == indexB:
                if in1[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, in1[1], in1[2]) * hyb(p, in1[2]) * hyb(k, indexb),
                        hyb(p, s) * lcs(s, indexb, indexB) * k / d,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, in1[1], in1[2]) * hyb(p, in1[2]) * hyb(q, indexb),
                        hyb(p, s) * lcs(s, indexb, indexB) * q * z / d,
                    )
                else:
                    Tenzor = Tenzor.subs(
                        H(q, in1[1], in1[2]) * hyb(p, in1[2]) * hyb(q, indexb),
                        hyb(p, s) * lcs(s, indexb, indexB) * q / d,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, in1[1], in1[2]) * hyb(p, in1[2]) * hyb(k, indexb),
                        hyb(p, s) * lcs(s, indexb, indexB) * k * z / d,
                    )


    # lcs( i, j, l) - Levi-Civita symbol
    Tenzor = Tenzor.subs(lcs(s, indexb, indexB), -lcs(s, indexB, indexb))  #

    Tenzor = simplify(Tenzor)

    print("step 11:", round(time.time() - t, 1), "sec")

    result = str(Tenzor)
    result = result.replace("**", "^")
    Fey_graphs.write(
        f"\nTensor structure of the diagram after calculation: \n"
    )
    Fey_graphs.write(f"\n{result} \n")

    print("Tensor structure of the diagram after calculation:", Tenzor)

    # print(pretty(Tenzor, use_unicode=False))

    # Fey_graphs.write("\n"+ pretty(Tenzor, use_unicode=False) + "\n")

    # Fey_graphs.write("\n"+ latex(Tenzor) + "\n")

    Fey_graphs.close()

with open('Two-loop MHD diagrams.txt') as MHD_diagrams_file:

    for graf in MHD_diagrams_file.readlines():

        get_output_data(graf)