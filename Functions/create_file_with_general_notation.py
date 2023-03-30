from Functions.get_diagram_integrand import *

# ------------------------------------------------------------------------------------------------------------------#
#                               Create a file with general notation and information
# ------------------------------------------------------------------------------------------------------------------#


def get_propagators_from_list_of_fields(fields_for_propagators):
    """
    Glues separate fields indexes from propagators_with_helicity into the list of propagators

    OUTPUT DATA EXAMPLE:

    list_of_propagators_with_helicity = ['vv', 'vb', 'bv', 'bb']
    """
    dimension = len(fields_for_propagators)
    list_of_propagators = [0] * dimension
    for i in range(dimension):
        list_of_propagators[i] = fields_for_propagators[i][0] + fields_for_propagators[i][1]
    return list_of_propagators


def create_file_with_info_and_supplementary_matherials():
    """
    Creates a file with general information and supplementary matherials
    """
    with open("General_notation.txt", "w+") as Notation_file:

        Notation_file.write(
            f"A detailed description of most of the notation introduced in this program can be found in the articles:\n"
            f"[1] Adzhemyan, L.T., Vasil'ev, A.N., Gnatich, M. Turbulent dynamo as spontaneous symmetry breaking. \n"
            f"Theor Math Phys 72, 940-950 (1987). https://doi.org/10.1007/BF01018300 \n"
            f"[2] Hnatic, M., Honkonen, J., Lucivjansky, T. Symmetry Breaking in Stochastic Dynamics and Turbulence. \n"
            f"Symmetry 2019, 11, 1193. https://doi.org/10.3390/sym11101193 \n"
            f"[3] D. Batkovich, Y. Kirienko, M. Kompaniets S. Novikov, GraphState - A tool for graph identification\n"
            f"and labelling, arXiv:1409.8227, program repository: https://bitbucket.org/mkompan/graph_state/downloads\n"
        )

        Notation_file.write(
            f"\nGeneral remarks: \n"
            f"0. Detailed information regarding the definition of the Nickel index can be found in [3]. \n"
            f"1. Fields: v is a random vector velocity field, b is a vector magnetic field, "
            f"B and V are auxiliary vector fields \n"
            f"(according to Janssen - De Dominicis approach).\n"
            f"2. List of non-zero propagators: {get_propagators_from_list_of_fields(all_nonzero_propagators)}\n"
            f"3. Momentums and frequencies: {p, w} denotes external (inflowing) momentum and frequency, "
            f"{momentums_for_helicity_propagators} and {frequencies_for_helicity_propagators} \n"
            f"denotes momentums and frequencies flowing along the loops in the diagram.\n"
            f"4. Vertices in the diagram are numbered in ascending order from 0 to {number_int_vert - 1}.\n"
            f"5. Loop structure: for technical reasons, it is convenient to give new momentums "
            f"{momentums_for_helicity_propagators} and frequencies {frequencies_for_helicity_propagators} \n"
            f"to propagators containing D_v kernel (see definition below): "
            f"{get_propagators_from_list_of_fields(propagators_with_helicity)} \n"
            f"The first loop corresponds to the pair {k, w_k} and the second to the pair {q, w_q}.\n"
        )  # write some notes

        Notation_file.write(
            f"\nDefinitions of non-zero elements of the propagator matrix in the "
            f"momentum-frequency representation:\n"
        )

        [i, j, l] = symbols("i j l", integer=True)

        all_fields_glued_into_propagators = get_propagators_from_list_of_fields(all_nonzero_propagators)

        info_about_propagators = list()
        for m in range(len(all_nonzero_propagators)):
            info_about_propagators.append(
                define_propagator_product(([]), ([]), 1, "", 1, all_nonzero_propagators[m], k, w_k, i, j)
            )
            propagator_without_tensor_structure = info_about_propagators[m][4]
            tensor_structure_of_propagator = info_about_propagators[m][0]
            Notation_file.write(
                f"\n{all_fields_glued_into_propagators[m]}{k, q, i, j} = "
                f"{propagator_without_tensor_structure*tensor_structure_of_propagator}\n"
            )  # write the propagator definition into file

        Notation_file.write(
            f"\nVertex factors: \n"
            f"\nvertex_factor_Bbv(k, i, j, l) = {vertex_factor_Bbv(k, i, j, l).doit()}\n"
            f"\nvertex_factor_Vvv(k, i, j, l) = {vertex_factor_Vvv(k, i, j, l).doit()}\n"
            f"\nvertex_factor_Vbb(k, i, j, l) = {vertex_factor_Vvv(k, i, j, l).doit()}\n"
            f"\nHere arguments i, j, l are indices of corresonding fields in corresponding vertex \n"
            f"(for example, in the Bbv-vertex i denotes index of B, i - index of b, and l - index ob v).\n"
        )  # write vertex factors

        Notation_file.write(
            f"\nUsed notation: \n"
            f"""\nHereinafter, unless otherwise stated, the symbol {k} denotes the vector modulus. 
{A} parametrizes the model type: model of linearized NavierStokes equation ({A} = -1), kinematic MHD turbulence 
({A} = 1), model of a passive vector field advected by a given turbulent environment ({A} = 0). 
Index {s} reserved to denote the component of the external momentum {p}, {d} the spatial dimension of the system 
(its physical value is equal to 3), function sc_prod(. , .) denotes the standard dot product of vectors in R**d
(its arguments are always vectors!). 

Function hyb(k, i) denotes i-th component of vector {k} (i = 1, ..., d), functions kd(i, j) and lcs(i, j, l) 
denotes the Kronecker delta and Levi-Civita symbols.

We assume that the integration in the diagram is carried out in the spherical coordinate system for vectors 
{k} and {q}: k_1 = k*sin(theta_1)*cos(phi_1), k_2 = k*sin(theta_1)*sin(phi_1), k_3 = k*cos(theta_1), 
q_1 = k*sin(theta_2)*cos(phi_2), q_2 = k*sin(theta_2)*sin(phi_2), q_3 = k*cos(theta_2), where theta_1, theta_2 are
angles between vector {B} and {k}, or {q} respectively, vector {B} is proportional to the magnetic induction.
Corresponding measure is: dmu = 2*(pi - u)*k^2*q^2*sin(theta_1)*sin(theta_2)*dk*dq*dtheta_1*d_theta_2*du, where 
2*u = (phi_2 - phi_1) (all diagrams depends only from cos(phi_1 - phi_2)). We also introduce following notation for
cosines: {z} = cos(angle between {k} and {q}) = sin(theta_1)*sin(theta_2)*cos(phi_1 - phi_2) + cos(theta_1)*cos(theta_2), 
{z_k} = cos(theta_1), {z_q} = cos(theta_2).

The rest of the model numeric parameters are {nuo} -- renormalized kinematic viscosity, {go} -- coupling constant,
{mu} -- renormalization mass, {uo} -- renormalized reciprocal magnetic Prandtl number, {rho} -- gyrotropy 
parameter (abs(rho) < 1), and {eps} determines a degree of model deviation from logarithmicity (0 < eps =< 2). \n
"""
            f"\nD_v(k) = {D_v(k).doit()}\n"
            f"\nalpha(k, w) = {alpha(nuo, k, w).doit()}\n"
            f"\nalpha_star(k, w) = alpha*(k, w) = {alpha_star(nuo, k, w).doit()}\n"
            f"\nbeta(k, w) = {beta(nuo, k, w).doit()}\n"
            f"\nbeta_star(k, w) = beta*(k, w) = {beta_star(nuo, k, w).doit()}\n"
            f"\nf_1(B, k, w) = {f_1(B, nuo, k).doit().doit()}\n"
            f"\nf_2(B, k, w) = {f_2(B, nuo, k).doit().doit()}\n"
            f"\nchi_1(k, w) = {chi_1(k, w).doit()}\n"
            f"\nchi_2(k, w) = {chi_2(k, w).doit()}\n"
            f"\nxi(k, w) = {xi(k, w).doit()}\n"
            f"\nxi_star(k, w) = xi*(k, w) = {xi_star(k, w).doit()}\n"
        )

        Notation_file.write(
            f"\nEach diagram in the corresponding file is defined by a formula of the following form: \n"
            f"\nDiagram = symmetry_factor*integral_with_measure*integrand*tensor_structure, \n"
            f"""\nwhere integrand is a part of the product of propagators (no tensor operators in the numerator) 
and tensor_structure is a corresponding product of tensor operators. \n"""
        )

    Notation_file.close()
