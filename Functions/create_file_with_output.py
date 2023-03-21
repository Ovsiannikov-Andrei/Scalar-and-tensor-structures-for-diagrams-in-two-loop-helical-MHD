# ------------------------------------------------------------------------------------------------------------------#
#                      We create a file and start write the information about diagram into it
# ------------------------------------------------------------------------------------------------------------------#


def get_information_from_Nickel_index(graf, diagram_number):
    """
    Generates a file name with results for each particular diagram
    using the data from the file "Two-loop MHD diagramms".

    ARGUMENTS:

    graf -- Nickel index of the diagram + symmetry factor
    diagram_number -- ordinal number of the line with the Nickel index from the file with the list of all indices

    OUTPUT DATA EXAMPLE:

    File name example: "Diagram__e12-23-3-e+0B_bB_vv-vB_bb-bV-0b.txt"
    (all "|" are replaced by -, ":" is replaced by +)

    Nickel index examples: e12|23|3|e|:0B_bB_vv|vB_bb|bV|0b|, e12|e3|33||:0B_bV_vb|0b_bV|Bv_vv||

    Symmetry factor example: 1
    """

    Nickel_index = "".join(graf.split(sep="SC = ")[0])
    Symmetry_factor = " ".join(graf.split(sep="SC = ")[1])
    # separating the Nickel index from the symmetry factor of the diagram

    Nickel_topology = "_".join(graf.split(sep="SC = ")[0].rstrip().split(sep=":")[0].split(sep="|"))[:-1]
    # topological part of the Nickel index

    Nickel_lines = "__".join(graf.split(sep="SC = ")[0].rstrip().split(sep=":")[1].split(sep="|"))[:-1]
    # line structure in the diagram corresponding to Nickel_topology

    return [
        f"{diagram_number}. Diagram__{Nickel_topology.strip()}__{Nickel_lines.strip()}.txt",
        Nickel_index.strip(),
        Symmetry_factor.strip(),
    ]


def get_list_with_propagators_from_nickel_index(nickel):
    """
    Arranges the propagators into a list of inner and outer lines with fields. The list is constructed as follows:
    vertex 0 is connected to vertex 1 by a line b---B, vertex 0 is connected to vertex 2 by a line v---v, etc.

    ARGUMENTS:

    nickel -- Nickel index of the diagram.
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

    s2 = nickel.find(":")
    # runs through the part of the Nickel index describing the lines (after the symbol :)

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
