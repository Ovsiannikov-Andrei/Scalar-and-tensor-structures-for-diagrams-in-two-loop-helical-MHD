from Functions.Data_classes import *

# ------------------------------------------------------------------------------------------------------------------#
#                      We create a file and start write the information about diagram into it
# ------------------------------------------------------------------------------------------------------------------#


def get_information_from_Nickel_index(line_with_info: str, diagram_number: int):
    """
    Generates a file name with results for each particular diagram
    using the data from the file "Two-loop MHD diagramms".

    ARGUMENTS:

    line_with_info -- Nickel index of the diagram + symmetry factor
    diagram_number -- ordinal number of the line with the Nickel index from the file with the list of all indices

    OUTPUT DATA EXAMPLE:

    File name example: "Diagram__e12_23_3_e__0B_bB_vv__vB_bb__bV__0b.txt"
    (all "|" are replaced by __, ":" is replaced by __)

    Nickel index examples: e12|23|3|e|:0B_bB_vv|vB_bb|bV|0b|, e12|e3|33||:0B_bV_vb|0b_bV|Bv_vv||

    Symmetry factor example: 1
    """

    # separating the Nickel index from the symmetry factor of the diagram
    nickel_index = "".join(line_with_info.split(sep="SC = ")[0])
    symmetry_factor = " ".join(line_with_info.split(sep="SC = ")[1])

    # topological part of the Nickel index
    nickel_topology = "_".join(line_with_info.split(sep="SC = ")[0].rstrip().split(sep=":")[0].split(sep="|"))[:-1]

    # line structure in the diagram corresponding to Nickel_topology
    nickel_lines = "__".join(line_with_info.split(sep="SC = ")[0].rstrip().split(sep=":")[1].split(sep="|"))[:-1]

    nickel_index_info = NickelIndexInfo(
        f"{diagram_number}.Diagram__{nickel_topology.strip()}__{nickel_lines.strip()}.txt",
        nickel_index.strip(),
        str(symmetry_factor.strip()),
    )

    return nickel_index_info


def get_list_with_propagators_from_nickel_index(nickel_index: str):
    """
    Arranges the propagators into a list of inner and outer lines with fields. The list is constructed as follows:
    vertex 0 is connected to vertex 1 by a line b---B, vertex 0 is connected to vertex 2 by a line v---v, etc.

    ARGUMENTS:

    nickel_index -- Nickel index of the diagram.
    It is defined by the function get_information_from_Nickel_index()

    Note:

    Works only for diagrams with triplet vertices

    OUTPUT DATA EXAMPLE:

    propagator(e12|23|3|e|:0B_bB_vv|vB_bb|bV|0b|) =
    [
    [[(0, 1), ['b', 'B']], [(0, 2), ['v', 'v']], [(1, 2), ['v', 'B']], [(1, 3), ['b', 'b']], [(2, 3), ['b', 'V']]],
    [[(-1, 0), ['0', 'B']], [(-1, 3), ['0', 'b']]]
    ]

    """

    s1 = 0
    # numbers individual blocks |...| in the topological part of the Nickel index
    # (all before the symbol :), i.e. vertices of the diagram

    s2 = nickel_index.find(":")
    # runs through the part of the Nickel index describing the lines (after the symbol :)

    propagator_internal = []
    propagator_external = []

    for i in nickel_index[: nickel_index.find(":")]:
        if i == "e":
            propagator_external += [[(-1, s1), ["0", nickel_index[s2 + 2]]]]
            s2 += 3
        elif i != "|":
            propagator_internal += [[(s1, int(i)), [nickel_index[s2 + 1], nickel_index[s2 + 2]]]]
            s2 += 3
        else:
            s1 += 1

    def get_list_as_dictionary(list: list):
        """
        Turns the list into a dictionary, keys are digits

        ARGUMENTS:

        Some list
        """
        dictionary = dict()
        for x in range(len(list)):
            dictionary.update({x: list[x]})
        return dictionary

    dict_internal_propagators = get_list_as_dictionary(propagator_internal)
    dict_external_propagators = get_list_as_dictionary(propagator_external)

    diagram_lines = InternalAndExternalLines(
        propagator_internal, propagator_external, dict_internal_propagators, dict_external_propagators
    )

    return diagram_lines
