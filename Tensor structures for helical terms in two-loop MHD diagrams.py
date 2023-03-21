#!/usr/bin/python3


# A detailed description of most of the notation introduced in this program can be found in the articles:

# [1] Adzhemyan, L.T., Vasil'ev, A.N. & Gnatich, M. Turbulent dynamo as spontaneous symmetry breaking.
# Theor Math Phys 72, 940–950 (1987). https://doi.org/10.1007/BF01018300

# [2] Hnatič, M.; Honkonen, J.; Lučivjanský, T. Symmetry Breaking in Stochastic Dynamics and Turbulence.
# Symmetry 2019, 11, 1193. https://doi.org/10.3390/sym11101193

# [3] D. Batkovich, Y. Kirienko, M. Kompaniets, and S. Novikov, GraphState - A tool for graph identification
# and labelling, arXiv:1409.8227, program repository: https://bitbucket.org/mkompan/graph_state/downloads

# ATTENTION!!! Already existing names of variables and functions should NOT be changed!


import os

from Functions.create_file_with_general_notation import *
from Functions.DIAGRAM_DESCRIPTION import *
from Functions.DIAGRAM_CALCULATION import *

# ------------------------------------------------------------------------------------------------------------------#
#                                        Computing two-loop MHD diagrams
# ------------------------------------------------------------------------------------------------------------------#


def main():
    """
    The program reads the Nickel indices line by line from a special external file (0),
    performs calculations and writes the output data about the diagram to the created file.

    The output data includes the topology of the diagram, the distribution of momenta and frequencies, and
    the diagram integrand (the product of tensor operators and everything else separately). All integrands are
    calculated up to the level of taking integrals over frequencies and calculating tensor convolutions.
    """

    if not os.path.isdir("Results"):
        # create the Results folder if it doesn't already exist
        os.mkdir("Results")

    create_file_with_info_and_supplementary_matherials()
    # create a file with decoding of all notations and additional information

    number_of_counted_diagrams = 0  # counter to count already processed diagrams

    print(f"PROGRAM START")

    d_default = 3  #  coordinate space dimension (assumed to be 3 by default)
    eps_default = 0.5  # value of the eps regularization parameter (default is assumed to be 0.5)
    A_MHD = 1  # value of the model parameter A (MHD corresponds to A = 1)

    print(f"\nDefault parameters: ")
    print(
        f"Coordinate space dimension d = {d_default}, regularization parameter eps = {eps_default} "
        f"(regularization in d = 4 - 2*eps space) and A = {A_MHD}"
    )

    calc_with_uo = input("\nDo you want to perform calculations for specific values of the Prandtl number? (y/n) ")

    list_with_uo_values = []

    if calc_with_uo == "y":
        entered_list_with_uo = input("Enter a list of desired magnetic Prandtl numbers separated by spaces: ").split()
        list_with_uo_values = [float(i) for i in entered_list_with_uo]
    else:
        list_with_uo_values.append("uo")

    with open("Two-loop MHD diagrams.txt", "r") as MHD_diagrams_file:

        for diagram in MHD_diagrams_file.readlines():

            print(f"\nCALCULATION {number_of_counted_diagrams + 1} BEGIN")

            diagram_data = get_info_about_diagram(diagram, number_of_counted_diagrams + 1)

            output_file_name = diagram_data[0]
            # create the name of the file with results
            momentums_at_vertices = diagram_data[1]
            # momentum distribution at the vertices
            indexb = diagram_data[2]
            # index of the inflowing field
            indexB = diagram_data[3]
            # index of the outflowing field
            P_structure = diagram_data[4]
            # save the Projector operator arguments
            H_structure = diagram_data[5]
            # save the Helical operator arguments
            kd_structure = diagram_data[6]
            # save the Kronecker delta arguments
            momentum_structure = diagram_data[7]
            # save all momentums and their components
            Integrand_tensor_part = diagram_data[8]
            # save the tensor structure (product of the tensor operators)
            Integrand_scalar_part = diagram_data[9]
            # save the scalar function
            diagram_convergent_criterion = diagram_data[10]
            # corresponding integral is convergent (True/False)

            diagram_integrand_data = diagram_integrand_calculation(
                output_file_name,
                momentums_at_vertices,
                indexb,
                indexB,
                P_structure,
                H_structure,
                kd_structure,
                momentum_structure,
                Integrand_tensor_part,
                Integrand_scalar_part,
                diagram_convergent_criterion,
            )

            scalar_part_without_repl = diagram_integrand_data[0]
            # scalar part of the integrand reduced to a common denominator and partially simplified.
            # Here, the replacement of momentums k, q -- > B*k/nuo, B*q/nuo has not yet been carried out.
            # This is the last expression here, which makes sense for both divergent and convergent diagrams.

            if diagram_convergent_criterion == True:
                # Below, all diagrams are assumed to be convergent.
                scalar_part_depending_only_on_uo = diagram_integrand_data[1]
                # scalar part of the integrand depending only on uo (replacing  k, q -- > B*k/nuo, B*q/nuo)
                field_and_nuo_factor = diagram_integrand_data[2]
                # factor by which the scalar part of the integrand is multiplied (all dependence on |B| and nuo)
                tensor_convolution = diagram_integrand_data[3]
                # computed tensor structure corresponding to the rotor (and only) terms

                integrand_for_numeric_calculation = preparing_diagram_for_numerical_integration(
                    output_file_name,
                    tensor_convolution,
                    eps_default,
                    d_default,
                    A_MHD,
                    list_with_uo_values,
                    field_and_nuo_factor,
                    scalar_part_depending_only_on_uo,
                )

            print(f"\nCALCULATION {number_of_counted_diagrams + 1} END \n")

            number_of_counted_diagrams += 1

    print(f"Number of counted diagrams: {number_of_counted_diagrams}")


# ------------------------------------------------------------------------------------------------------------------#
#                                                    Entry point
# ------------------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    main()
