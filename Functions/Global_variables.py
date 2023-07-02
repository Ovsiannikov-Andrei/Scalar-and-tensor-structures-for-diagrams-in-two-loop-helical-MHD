from sympy import *

# -----------------------------------------------------------------------------------------------------------------#
#                                               Global variables and symbols
# -----------------------------------------------------------------------------------------------------------------#

[p, k, q] = symbols("p k q", real=True)
"""
p denotes an external (inflowing) momentum
k and q denote momentums flowing through loops
"""

[w, w_k, w_q] = symbols("w, w_k, w_q", real=True)
"""
w denotes an external (inflowing) frequency
w_k and w_q denote frequencies flowing through loops
"""

[A, s, d] = symbols("A s d", integer=True)
"""
A parametrizes the model type: model of linearized NavierStokes equation (A = -1), 
kinematic MHD turbulence (A = 1), model of a passive vector field advected by a given turbulent 
environment (A = 0)

Note: this program works only for the cases A = 0, A = 1.

s reserved to denote the component of the external momentum p 
d the spatial dimension of the system, its physical value is equal to 3
"""

[z, z_k, z_q, Lambda_cutoff, B] = symbols("z z_k z_q Lambda_cutoff B", real=True)
"""
z = cos(angle between k and q) = dot_product(k, q)/ (abs(k) * abs(q))
z_k = cos(angle between k and B) = dot_product(k, B)/ (abs(k) * abs(B))
z_q = cos(angle between q and B) = dot_product(q, B)/ (abs(q * abs(B))

B is a field proportional to the magnetic induction vector

Attention! 
Do not confuse the field B (in the notation [1] it is theta') and <B> is a 
spontaneously arising constant magnetic field (in the notation [1] it is c)

Lambda is a momentum dimension cutoff parameter
"""

[nuo, nu, mu, uo, u, rho] = symbols("nuo nu mu uo u rho", positive=True)
"""
nuo is a bare kinematic viscosity, nu is a renormalized kinematic viscosity, mu is a renormalization mass, 
uo is a bare reciprocal magnetic Prandtl number, u is a renormalized reciprocal magnetic Prandtl number,
and rho is a gyrotropy parameter (abs(rho) < 1)
"""

[go, g, eps] = symbols("go, g eps", real=True)
"""
go is a bare coupling constant, g is a renormalized coupling constant, 
eps determines a degree of model deviation from logarithmicity (0 < eps =< 2)
"""

[a1, a2, a3] = symbols("a1 a2 a3", positive=True)
"""
a1, a2, ... are some additional symbols needed to define the properties of the functions defined below
"""

"""
Due to the appearance of <B> != 0, additional propagators appear in MHD. 
Here v is phi, V is phi', b is theta and B is theta' (in the notation [1]).
"""
all_nonzero_propagators = [
    ["v", "v"],
    ["v", "V"],
    ["V", "v"],
    ["b", "B"],
    ["B", "b"],
    ["v", "b"],
    ["b", "v"],
    ["b", "b"],
    ["V", "b"],
    ["b", "V"],
    ["B", "v"],
    ["v", "B"],
]
"""
The set all_nonzero_propagators contains all possible nonzero propagators.
"""
propagators_with_helicity = [["v", "v"], ["v", "b"], ["b", "v"], ["b", "b"]]
"""
The set propagators_with_helicity consists of propagators containing the core D_v (see below). 
In this program, this set is used to define the loop structure of the diagram.
"""
momentums_for_helicity_propagators = [k, q]
frequencies_for_helicity_propagators = [w_k, w_q]
""" 
For technical reasons, it is convenient for us to give to propagators from propagators_with_helicity 
new momentums (momentums_for_helicity_propagators) and frequencies (frequencies_for_helicity_propagators). 
The first loop corresponds to the pair (k, w_k) and the second to the pair (q, w_q).
"""

number_int_vert = 4
""" 
Parameter number_int_vert is a total number of internal (three-point) vertecies in diagram
"""
stupen = 1
""" 
Parameter stupen denotes the desired degree of rho.  
"""
