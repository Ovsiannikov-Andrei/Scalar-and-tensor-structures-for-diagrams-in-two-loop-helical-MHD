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

from Functions.Data_classes import *
from Functions.create_file_with_general_notation import *
from Functions.DIAGRAM_DESCRIPTION import *
from Functions.DIAGRAM_CALCULATION import *
from Functions.preparing_for_numerical_integration import *

# ------------------------------------------------------------------------------------------------------------------#
#                                        Computing two-loop MHD diagrams
# ------------------------------------------------------------------------------------------------------------------#


def main():
    """
    The program reads the Nickel indices line by line from a special external file "Two-loop MHD self-energy diagrams.txt",
    performs calculations and writes the output data about the diagram to the created file.

    The output data includes the topology of the diagram, the distribution of momenta and frequencies, and
    the diagram integrand (the product of tensor operators and everything else separately). All integrands are
    calculated up to the level of taking integrals over frequencies and calculating tensor convolutions.
    """

    # create folder with all info about structure of diagrams, if it doesn't already exist
    if not os.path.isdir("Details about the diagrams"):
        os.mkdir("Details about the diagrams")

    # create the Final Results folder for integrends for numerical calculations,
    # if it doesn't already exist
    if not os.path.isdir("Final Results"):
        os.mkdir("Final Results")
    if not os.path.isdir("Final Results/UV-finite diagrams"):
        os.mkdir("Final Results/UV-finite diagrams")
    if not os.path.isdir("Final Results/UV-infinite diagrams"):
        os.mkdir("Final Results/UV-infinite diagrams")

    # create a file with decoding of all notations and additional information
    create_file_with_info_and_supplementary_matherials()

    # counter to count already processed diagrams
    number_of_counted_diagrams = 0

    print(f"PROGRAM START")

    # default value of coordinate space dimension
    d_default = 3
    # default value of the regularization parameter eps
    eps_default = 0
    # default value of the model parameter A (MHD corresponds to A = 1)
    A_MHD = 1
    # one-loop reciprocal magnetic Prandtl number at the fixed point
    uo_default = round((sqrt(43 / 3) - 1) / 2, 3)

    print(
        f"""\nDefault parameters: 
coordinate space dimension d = {d_default}, 
one-loop reciprocal magnetic Prandtl number at the fixed point uo = {uo_default}, 
regularization parameter epsilon = {eps_default}, and model type A = {A_MHD} (MHD). \n"""
    )

    model_parameters = input(
        f"""Would you like to change other default parameter values or obtain integrand expressions 
for uo values different from the default? (y/n) """
    )

    list_with_uo_values = []

    if model_parameters == "y":
        d_input = input(f"Enter a space dimension: d = ")
        eps_input = input(f"Enter value of the regularization parameter: epsilon = ")
        A_input = input(f"Enter model type (valid values are A = 0, A = 1): A = ")
        entered_list_with_uo = input(f"Enter a list of desired magnetic Prandtl numbers separated by spaces: ").split()
        list_with_uo_values = [float(i) for i in entered_list_with_uo]
    else:
        d_input = d_default
        eps_input = eps_default
        A_input = A_MHD
        list_with_uo_values.append(uo_default)

    output_in_WfMath_format = input(
        f"\nWould you also like to get results in a format suitable for use in Wolfram Mathematica? (y/n) "
    )

    with open("Two-loop MHD self-energy diagrams.txt", "r") as MHD_diagrams_file:
        for Nickel_index in MHD_diagrams_file.readlines():
            print(f"\nCALCULATION {number_of_counted_diagrams + 1} BEGIN")

            diagram_data = get_info_about_diagram(Nickel_index, output_in_WfMath_format, number_of_counted_diagrams + 1)

            diagram_integrand_data = diagram_integrand_calculation(diagram_data, output_in_WfMath_format)

            preparing_diagram_for_numerical_integration(
                diagram_data.output_file_name,
                diagram_integrand_data,
                eps_input,
                d_input,
                A_input,
                uo_default,
                list_with_uo_values,
                output_in_WfMath_format,
                diagram_data.expression_UV_convergence_criterion,
            )

            print(f"\nCALCULATION {number_of_counted_diagrams + 1} END \n")

            number_of_counted_diagrams += 1

    print(f"Number of counted diagrams: {number_of_counted_diagrams}")


# ------------------------------------------------------------------------------------------------------------------#
#                                                    Entry point
# ------------------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    main()
