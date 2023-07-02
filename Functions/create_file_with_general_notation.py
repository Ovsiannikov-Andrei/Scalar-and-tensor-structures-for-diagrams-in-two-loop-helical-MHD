from Functions.Data_classes import *
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
    Create a file with general information and supplementary matherials
    """
    with open("General_notation.txt", "w+") as Notation_file:
        Notation_file.write(
            f"""A detailed description of most of the notation introduced in this program can be found in the articles:
[1] Adzhemyan, L.T., Vasil'ev, A.N., Gnatich, M. Turbulent dynamo as spontaneous symmetry breaking. 
Theor Math Phys 72, 940-950 (1987). https://doi.org/10.1007/BF01018300 
[2] Hnatic, M., Honkonen, J., Lucivjansky, T. Symmetry Breaking in Stochastic Dynamics and Turbulence. 
Symmetry 2019, 11, 1193. https://doi.org/10.3390/sym11101193 
[3] D. Batkovich, Y. Kirienko, M. Kompaniets S. Novikov, GraphState - A tool for graph identification
and labelling, arXiv:1409.8227, program repository: https://bitbucket.org/mkompan/graph_state/downloads\n """
        )

        Notation_file.write(
            f"""\nIn general, the contribution of each diagram corresponding to the Nickel index from the file 
Two-loop MHD diagrams.txt can be symbolically written as

Diagram = s*int(dmu*integrand), integrand = T_ij*F,

where s is a symmetry factor of the diagram, dmu is a measure of integration, and T_ij*F is the integrand that 
this program is dedicated to calculating. The calculation of the integrand is divided into two stages: 
the calculation of the tensor structure T_ij and the calculation of the scalar part F. The structure T_ij includes 
tensor structures from each propagator (ie the operators P_ij, P_ij + i*rho*H_ij, as well as vertex factors), 
F contains everything that remains from the product of propagators.

See below for further explanation.\n"""
        )

        Notation_file.write(
            f"""\nGeneral remarks about the procedure for restoring the diagram by the Nickel index: 
0. Detailed information regarding the definition of the Nickel index can be found in [3]. 
1. Fields: v is a random vector velocity field, b is a vector magnetic field, 
B and V are auxiliary vector fields (according to Janssen - De Dominicis approach).
2. List of non-zero propagators: {get_propagators_from_list_of_fields(all_nonzero_propagators)}
3. Momentums and frequencies: {p, w} denotes external (inflowing to diagram through field B) momentum 
and frequency, {momentums_for_helicity_propagators} and {frequencies_for_helicity_propagators} \
denotes momentums and frequencies flowing along the lines containing 
the kernel {D_v}.
4. Vertices in the diagram are numbered in ascending order from 0 to {number_int_vert - 1} in the order 
they occur in the Nickel index.
5. Loop structure: for technical reasons, it is convenient to assign new momentums {momentums_for_helicity_propagators} and frequencies 
{frequencies_for_helicity_propagators} to propagators containing D_v kernel (see definition below): 
{get_propagators_from_list_of_fields(propagators_with_helicity)}
The first loop corresponds to the pair {k, w_k} and the second to the pair {q, w_q}.\n"""
        )  # write some notes

        Notation_file.write(
            f"\nDefinitions of non-zero elements of the propagator matrix in the "
            f"momentum-frequency representation:\n"
        )

        [i, j, l] = symbols("i j l", integer=True)
        empty_integrand_scalar_and_tensor_parts = IntegrandPropagatorProduct()

        all_fields_glued_into_propagators = get_propagators_from_list_of_fields(all_nonzero_propagators)

        info_about_propagators = list()
        for m in range(len(all_nonzero_propagators)):
            propagator_product = define_propagator_product(
                empty_integrand_scalar_and_tensor_parts, all_nonzero_propagators[m], k, w_k, i, j
            )
            info_about_propagators.append([propagator_product.scalar_part, propagator_product.tensor_part])

            propagator_without_tensor_structure = info_about_propagators[m][0]
            tensor_structure_of_propagator = info_about_propagators[m][1]
            Notation_file.write(
                f"\n{all_fields_glued_into_propagators[m]}{k, q, i, j} = "
                f"{propagator_without_tensor_structure*tensor_structure_of_propagator}"
            )  # write the propagator definition into file

        Notation_file.write(
            f"""\n\nVertex factors:

vertex_factor_Bbv(k, i, j, l) = {vertex_factor_Bbv(k, i, j, l).doit()}
vertex_factor_Vvv(k, i, j, l) = {vertex_factor_Vvv(k, i, j, l).doit()}
vertex_factor_Vbb(k, i, j, l) = {vertex_factor_Vvv(k, i, j, l).doit()}

Here arguments i, j, l are indices of corresonding fields in corresponding vertex 
(for example, in the Bbv-vertex i denotes index of {B}, i - index of b, and l - index ob v).\n"""
        )  # write vertex factors

        Notation_file.write(
            f"\nUsed notation: \n"
            f"""\nHereinafter, unless otherwise stated, the symbol {k} denotes the vector modulus. 
{A} parametrizes the model type: model of linearized NavierStokes equation ({A} = -1), kinematic MHD turbulence 
({A} = 1), and model of a passive vector field advected by a given turbulent environment ({A} = 0). 
Index {s} reserved to denote the component of the external momentum {p}, {d} the spatial dimension of the system 
(its physical value is equal to 3), function {sc_prod}(. , .) denotes the standard dot product of vectors in R**d
(its arguments are always vectors!). 

Function {hyb(k, i)} denotes i-th component of vector {k} (i = 1, ..., {d}), functions {kd(i, j)} and {lcs(i, j, l)} 
denotes the Kronecker delta and Levi-Civita symbols.

The rest of the model numeric parameters are {nuo} -- renormalized kinematic viscosity, {go} -- coupling constant,
{mu} -- renormalization mass, {uo} -- renormalized reciprocal magnetic Prandtl number, {rho} -- gyrotropy 
parameter (abs({rho}) < 1), and {eps} determines a degree of model deviation from logarithmicity (0 < {eps} =< 2).

{D_v(k)} = {D_v(k).doit()}
{alpha(nuo, k, w)} = {alpha(nuo, k, w).doit()}
{alpha_star(nuo, k, w)} = {alpha_star(nuo, k, w).doit()}
{beta(nuo, k, w)} = {beta(nuo, k, w).doit()}
{beta_star(nuo, k, w)} = {beta_star(nuo, k, w).doit()}
{f_1(B, nuo, k)} = {f_1(B, nuo, k).doit().doit()}
{f_2(B, nuo, k)} = {f_2(B, nuo, k).doit().doit()}
{chi_1(k, w)} = {chi_1(k, w).doit()}
{chi_2(k, w)} = {chi_2(k, w).doit()}
{xi(k, w)} = {xi(k, w).doit()}
{xi_star(k, w)} = {xi_star(k, w).doit()}\n"""
        )

        Notation_file.write(
            f"""\nRemarks about the integration procedure:
0. We assume that the integration in the diagram is carried out in the spherical coordinate system 
for vectors {k} and {q}: 

k_1 = k*sin(theta_1)*cos(phi_1), 
k_2 = k*sin(theta_1)*sin(phi_1), 
k_3 = k*cos(theta_1), 
q_1 = k*sin(theta_2)*cos(phi_2), 
q_2 = k*sin(theta_2)*sin(phi_2), 
q_3 = k*cos(theta_2), 

where theta_1, theta_2 are angles between vector {B} and {k}, or {q} respectively. Vector {B} is a vector 
proportional to the magnetic induction. Corresponding measure of integration is: 

dmu = 2*(pi - u)*k**2*q**2*sin(theta_1)*sin(theta_2)*dk*dq*dtheta_1*dtheta_2*du, 

where 2*u = (phi_2 - phi_1) (all diagrams depends only from cos(phi_1 - phi_2)). 
We also introduce following notation for cosines: 

{z} = cos(angle between {k} and {q}) = sin(theta_1)*sin(theta_2)*cos(phi_1 - phi_2) + cos(theta_1)*cos(theta_2), 
{z_k} = cos(theta_1), 
{z_q} = cos(theta_2).

1. After the replacement of arguments k, q --> |B|*k/nuo, |B|*q/nuo, the expression for the diagram is converted 
as follows 

Diagram = s*C*int(dmu*T1_ij*F1), T_ij = C_T*T1_ij, F = C_F*F1, C = C_mu*C_T*C_F, C_mu = (|B|/nuo)**6, C_T = (|B|/nuo)**2

where F1 = F1(k, q, z, z_k, z_q, uo) is a function that depends only on the parameter uo and integration variables, 
and C_F is a dimensional multiplier depending on |B|, nuo and all other dimensional parameters.
"""
        )

    Notation_file.close()
