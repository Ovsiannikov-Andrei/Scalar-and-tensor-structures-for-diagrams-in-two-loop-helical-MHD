#!/usr/bin/python3

import os
import sys
import copy
import itertools
import sympy as sym
from sympy import Number
from sympy import I, E, pi
from sympy import re, im, conjugate, Add, factorial
from sympy import *
from sympy import symbols, solve_linear
from functools import reduce
from collections import Counter
import time


# A detailed description of most of the notation introduced in this program can be found in the articles:

# 1. Adzhemyan, L.T., Vasil'ev, A.N. & Gnatich, M. Turbulent dynamo as spontaneous symmetry breaking.
# Theor Math Phys 72, 940–950 (1987). https://doi.org/10.1007/BF01018300

# 2. Hnatič, M.; Honkonen, J.; Lučivjanský, T. Symmetry Breaking in Stochastic Dynamics and Turbulence.
# Symmetry 2019, 11, 1193. https://doi.org/10.3390/sym11101193

# 3. D. Batkovich, Y. Kirienko, M. Kompaniets, and S. Novikov, GraphState - A tool for graph identification
# and labelling, arXiv:1409.8227, program repository: https://bitbucket.org/mkompan/graph_state/downloads

# ATTENTION!!! Already existing names of variables and functions should NOT be changed!

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

s reserved to denote the component of the external momentum p 
d the spatial dimension of the system, its physical value is equal to 3
"""

[z, theta] = symbols("z theta", real=True)
"""
z = cos(angle between k and q) = dot_product(k, q)/ (abs(k) * abs(q))
theta is proportional to the magnetic induction vector
"""

[nu, mu, u, rho] = symbols("nu mu u rho", positive=True)
"""
nu is a renormalized kinematic viscosity,  mu is a renormalization mass, 
u is a renormalized reciprocal magnetic Prandtl number, and  rho is a gyrotropy parameter (abs(rho) < 1)
"""

[g, eps] = symbols("g eps", real=True)
"""
g is a coupling constant, eps determines a degree of model deviation from logarithmicity (0 < eps =< 2)
"""

all_nonzero_propagators = [
    ["v", "v"], ["v", "V"], ["V", "v"], ["b", "B"], ["B", "b"], ["v", "b"],
    ["b", "v"], ["b", "b"], ["V", "b"], ["b", "V"], ["B", "v"], ["v", "B"]
]
propagators_with_helicity = [["v", "v"], ["v", "b"], ["b", "v"], ["b", "b"]]
momentums_for_helicity_propagators = [k, q]
frequencies_for_helicity_propagators = [w_k, w_q]
"""
The set all_nonzero_propagators contains all possible nonzero propagators.
The set propagators_with_helicity consists of propagators containing the core D_v (see below). 
In this program, this set is used to define the loop structure of the diagram.

For technical reasons, it is convenient for us to give to propagators from propagators_with_helicity 
new momentums (momentums_for_helicity_propagators) and frequencies (frequencies_for_helicity_propagators). 
The first loop corresponds to the pair (k, w_k) and the second to the pair (q, w_q).
"""

number_int_vert = 4
""" 
Parametr number_int_vert is a total number of internal (three-point) vertecies in diagram
"""
stupen = 1
""" 
Parameter stupen denotes the desired degree of rho.  
"""

# -----------------------------------------------------------------------------------------------------------------#
#                                       Custom functions in subclass Function in SymPy
# -----------------------------------------------------------------------------------------------------------------#

# All functions here are given in momentum-frequency representation


class D_v(Function):
    """ 
    D_v(k) = g*mu**(2*eps)*nu**3*k**(4 - d - 2*eps)

    ARGUMENTS:

    k -- absolute value of momentum

    PARAMETERS:

    g -- coupling constant, nu -- renormalized kinematic viscosity, d -- space dimension, 
    mu -- renormalization mass, eps -- determines a degree of model deviation from logarithmicity

    PROPERTIES:

    D_v(k) = D_v(-k)
    """
    @classmethod
    def eval(cls, k):
        if k.could_extract_minus_sign():
            return cls(-k)  # function is even with respect to k by definition

    def doit(self, deep=True, **hints):
        k = self.args[0]

        if deep:
            k = k.doit(deep=deep, **hints)
        return g*mu**(2*eps)*nu**3*k**(4 - d - 2*eps)


class alpha(Function):
    """ 
    alpha(k, w) = I*w + nu*k**2 

    ARGUMENTS:

    k -- momentum, w - frequency

    PARAMETERS:

    nu -- renormalized kinematic viscosity

    PROPERTIES:

    alpha(k, w) = alpha(-k, w)
    """
    @classmethod
    def eval(cls, k, w):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, w)

    def doit(self, deep=True, **hints):
        k, w = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
        return I*w + nu*k**2


class alpha_star(Function):
    """ 
    alpha_star(k, w) = conjugate(alpha(k, w)) = -I*w + nu*k**2 

    ARGUMENTS:

    k -- momentum, w - frequency

    PARAMETERS:

    nu -- renormalized kinematic viscosity

    PROPERTIES:

    alpha_star(k, w) = alpha_star(-k, w),

    alpha_star(k, w) = alpha(k, -w)
    """
    @classmethod
    def eval(cls, k, w):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, w)
        if w.could_extract_minus_sign():
            return alpha(k, -w)

    def doit(self, deep=True, **hints):
        k, w = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
        return -I*w + nu*k**2


class beta(Function):
    """ 
    beta(k, w) = I*w + uo*nu*k**2

    ARGUMENTS:

    k -- momentum, w - frequency

    PARAMETERS:

    u -- renormalized reciprocal magnetic Prandtl number, nu -- renormalized kinematic viscosity

    PROPERTIES:

    beta(k, w) = beta(-k, w)
    """
    @classmethod
    def eval(cls, k, w):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, w)

    def doit(self, deep=True, **hints):
        k, w = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
        return I*w + u*nu*k**2


class beta_star(Function):
    """ 
    beta_star(k, w) = conjugate(beta(k, w)) = -I*w + uo*nu*k**2

    ARGUMENTS:

    k -- momentum, w - frequency

    PARAMETERS:

    u -- renormalized reciprocal magnetic Prandtl number, nu -- renormalized kinematic viscosity

    PROPERTIES:

    beta_star(k, w) = beta_star(-k, w),

    beta_star(k, w) = beta(k, -w)    
    """
    @classmethod
    def eval(cls, k, w):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, w)
        if w.could_extract_minus_sign():
            return beta(k, -w)

    def doit(self, deep=True, **hints):
        k, w = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
        return -I*w + u*nu*k**2


class sc_prod(Function):
    """ 
    This auxiliary function denotes the standard dot product of vectors in R**d

    ARGUMENTS:

    theta -- external magnetic field, k -- momentum

    PROPERTIES:

    sc_prod(theta, k) = beta_star(theta, -k)   
    """
    @classmethod
    def eval(cls, theta, k):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return -cls(theta, -k)


class Discriminant(Function):
    """ 
    This auxiliary function is introduced for the convenience of the subsequent calculation of 
    integrals over frequencies (using the residue theorem). It defines the discriminant 
    that occurs when solving the quadratic equation xi(k, w) = 0 with respect to w (see below)

    under_root_expression(k, A) = 4*A*sc_prod(theta, k)**2 - k**4*nu**2*(uo - 1)**2

    ARGUMENTS:
    A - digit parameter of the model, k -- momentum

    PARAMETERS:

    u -- renormalized reciprocal magnetic Prandtl number, nu -- renormalized kinematic viscosity
    sc_prod(theta, k) -- dot product of external magnetic field theta and momentum k, 

    PROPERTIES:

    under_root_expression(k, A) = under_root_expression(-k, A)   
    """

    @classmethod
    def eval(cls, k, A):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, A)

    def doit(self, deep=True, **hints):
        k, A = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            A = A.doit(deep=deep, **hints)
        return 4*A*sc_prod(theta, k)**2 - k**4*nu**2*(u - 1)**2


class f_1(Function):
    """ 
    This auxiliary function is introduced for the convenience of the subsequent calculation of 
    integrals over frequencies (using the residue theorem). It returns the first root of the equation
    xi(k, w) = 0 with respect to w (see below).

    f_1(k, A) = (sqrt(under_root_expression(k, A)) + I*k**2*nu*uo + I*k**2*nu)/2

    ARGUMENTS:
    A - digit parameter of the model, k -- momentum

    PARAMETERS:

    u -- renormalized reciprocal magnetic Prandtl number, nu -- renormalized kinematic viscosity
    sc_prod(theta, k) -- dot product of external magnetic field theta and momentum k, 

    PROPERTIES:

    f_1(k, A) = f_1(-k, A)   
    """
    @classmethod
    def eval(cls, k, A):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, A)

    def doit(self, deep=True, **hints):
        k, A = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            A = A.doit(deep=deep, **hints)
        return (sqrt(Discriminant(k, A)) + I*k**2*nu*u + I*k**2*nu)/2


class f_2(Function):
    """ 
    This auxiliary function is introduced for the convenience of the subsequent calculation of 
    integrals over frequencies (using the residue theorem). It returns the second root of the equation
    xi(k, w) = 0 with respect to w (see below). 

    f_2(k, A) = (- sqrt(under_root_expression(k, A)) + I*k**2*nu*uo + I*k**2*nu)/2

    Note:
    f_2(k, A) differs from f_1(k, A) by sign before the square root

    ARGUMENTS:
    A - digit parameter of the model, k -- momentum

    PARAMETERS:

    u -- renormalized reciprocal magnetic Prandtl number, nu -- renormalized kinematic viscosity
    sc_prod(theta, k) -- dot product of external magnetic field theta and momentum k, 

    PROPERTIES:

    f_2(k, A) = f_2(-k, A),   

    conjugate(f_2(k, A)) = - f_1(k, A)  ==>  conjugate(f_1(k, A)) = - f_2(k, A)
    """
    @classmethod
    def eval(cls, k, A):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, A)

    def doit(self, deep=True, **hints):
        k, A = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            A = A.doit(deep=deep, **hints)
        return (-sqrt(Discriminant(k, A)) + I*k**2*nu*u + I*k**2*nu)/2


class chi_1(Function):
    """ 
    This auxiliary function is introduced for the convenience of the subsequent calculation of 
    integrals over frequencies (using the residue theorem). It defines the first monomial in the 
    decomposition of the square polynomial xi(k, w) (with respect to w) into irreducible factors (see below). 

    chi_1(k, w) = (w - f_1(k, A))

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    chi_1(k, w) = chi_1(-k, w)   
    """
    @classmethod
    def eval(cls, k, w):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, w)

    def doit(self, deep=True, **hints):
        k, w = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
        return (w - f_1(k, A))


class chi_2(Function):
    """ 
    This auxiliary function is introduced for the convenience of the subsequent calculation of 
    integrals over frequencies (using the residue theorem). It defines the second monomial in the 
    decomposition of the square polynomial xi(k, w) (with respect to w) into irreducible factors (see below). 

    chi_2(k, w) = (w - f_2(k, A))

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    chi_2(k, w) = chi_2(-k, w),  

    conjugate(chi_2(k, w)) = w - conjugate(f_2(k, A)) = w + f_1(k, A) = -(- w - f_1(k, A)) = -chi_1(k, -w),

    chi_2(k, -w) = (- w - f_2(k, A)) = -(w + f_2(k, A)) = -w + conjugate(f_1(k, A)) = -conjugate(chi_1(k, w))
    """
    @classmethod
    def eval(cls, k, w):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, w)

    def doit(self, deep=True, **hints):
        k, w = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
        return (w - f_2(k, A))


class xi(Function):
    """ 
    The function xi(k, w) is defined by the equality

    xi(k, w) = A*sc_prod(theta, k)**2 + alpha(k, w)*beta(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PARAMETERS:

    A - digit parameter of the model

    Representations for xi(k, w) (decomposition into irreducible ones):

    xi(k, w) = -chi_1(k, w)*chi_2(k, w) = -(w - f_1(k, A))*(w - f_2(k, A)) (see definitions above)

    PROPERTIES:

    xi(k, w) = xi(-k, w)  
    """
    @classmethod
    def eval(cls, k, w):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, w)

    def doit(self, deep=True, **hints):
        k, w = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
        return -chi_1(k, w)*chi_2(k, w)


class xi_star(Function):
    """ 
    The function xi_star(k, w) is defined by the equality

    xi_star(k, w) = conjugate(xi(k, w)) = A*sc_prod(theta, k)**2 + alpha_star(k, w)*beta_star(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PARAMETERS:

    A - digit parameter of the model

    Representations for xi_star(k, w) (decomposition into irreducible ones):

    xi_star(k, w) = -conjugate(chi_2(k, w))*conjugate(chi_1(k, w)) = -chi_2(k, -w)*chi_1(k, -w)

    PROPERTIES:

    xi(k, w) = xi(-k, w),

    xi_star(k, w) =  -(w + f_1(k, A))*(w + f_2(k, A)) = -(- w - f_1(k, A))*(- w - f_2(k, A)) = xi(k, -w),

    i.e. using the symmetry properties xi_star(k, w) is also expressed in terms of chi_1 and chi_2  
    """
    @classmethod
    def eval(cls, k, w):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, w)
        if w.could_extract_minus_sign():
            return xi(k, -w)  # xi_star(k, w) = xi(k, -w)

    def doit(self, deep=True, **hints):
        k, w = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
        # express the result in terms of the functions chi_1 and chi_2
        return -chi_2(k, -w)*chi_1(k, -w)


class kd(Function):
    """ 
    kd(index1, index2) defines the Kronecker delta function

    ARGUMENTS:
    index1 and index2 -- positive integers
    """


class hyb(Function):
    """ 
    hyb(k, index) returns the momentum index-th component

    ARGUMENTS:
    k -- momentum, index -- positive integer enumerates the components k
    """


class lcs(Function):
    """ 
    lcs(index1, index2, index3) defines the Levi-Civita symbol

    ARGUMENTS:
    index1, index2, index3 -- positive integers
    """


class vertex_factor_Bbv(Function):
    """ 
    The vertex_factor_Bbv(k, index_B, index_b, index_v) function determines the corresponding 
    vertex multiplier (Bbv) of the diagram.

    vertex_factor_Bbv(k, index_B, index_b, index_v) = I * (hyb(k, index_v) * kd(index_B, index_b) - 
    A * hyb(k, index_b) * kd(index_B, index_v))

    ARGUMENTS:
    k -- momentum, index_B, index_b, index_v -- positive integers
    """
    @classmethod
    def eval(cls, k, index_B, index_b, index_v):
        if isinstance(k, Number) and all(isinstance(m, Integer) for m in [index_B, index_b, index_v]):
            return I * (hyb(k, index_v) * kd(index_B, index_b) - A * hyb(k, index_b) * kd(index_B, index_v))

    def doit(self, deep=True, **hints):
        k, index_B, index_b, index_v = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            index_B = index_B.doit(deep=deep, **hints)
            index_b = index_b.doit(deep=deep, **hints)
            index_v = index_v.doit(deep=deep, **hints)

        return I * (hyb(k, index_v) * kd(index_B, index_b) - A * hyb(k, index_b) * kd(index_B, index_v))


class vertex_factor_Vvv(Function):
    """ 
    The vertex_factor_Vvv(k, index_V, index1_v, index2_v) function determines the corresponding 
    vertex multiplier (Vvv) of the diagram.

    vertex_factor_Vvv(k, index_V, index1_v, index2_v) = I * (hyb(k, index1_v) * kd(index_V, index2_v) + 
    hyb(k, index2_v) * kd(index_V, index1_v))

    ARGUMENTS:
    k -- momentum, index_V, index1_v, index2_v -- positive integers
    """
    @classmethod
    def eval(cls, k, index_V, index1_v, index2_v):
        if isinstance(k, Number) and all(isinstance(m, Integer) for m in [index_V, index1_v, index2_v]):
            return I * (hyb(k, index1_v) * kd(index_V, index2_v) + hyb(k, index2_v) * kd(index_V, index1_v))

    def doit(self, deep=True, **hints):
        k, index_V, index1_v, index2_v = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            index_V = index_V.doit(deep=deep, **hints)
            index1_v = index1_v.doit(deep=deep, **hints)
            index2_v = index2_v.doit(deep=deep, **hints)

        return I * (hyb(k, index1_v) * kd(index_V, index2_v) + hyb(k, index2_v) * kd(index_V, index1_v))


class P(Function):
    """ 
    Transverse projection operator
    """


class H(Function):
    """ 
    Helical term
    """

# -----------------------------------------------------------------------------------------------------------------#
#                                                Auxiliary functions
# -----------------------------------------------------------------------------------------------------------------#

# The diagram e12|23|3|e|:0B_bB_vv|vB_bb|bV|0b| is considered as an example of calculating the output data
# for all functions below.

# ------------------------------------------------------------------------------------------------------------------#
#                      We create a file and start write the information about diagram into it
# ------------------------------------------------------------------------------------------------------------------#


def get_information_from_Nickel_index(
    graf
):
    """
    Generates a file name with results for each particular diagram
    using the data from the file "Two-loop MHD diagramms".

    ARGUMENTS:

    graf -- Nickel index of the diagram + symmetry factor

    OUTPUT DATA EXAMPLE:

    File name example: "Diagram__e12-23-3-e+0B_bB_vv-vB_bb-bV-0b.txt" 
    (all "|" are replaced by -, ":" is replaced by +)

    Nickel index examples: e12|23|3|e|:0B_bB_vv|vB_bb|bV|0b|, e12|e3|33||:0B_bV_vb|0b_bV|Bv_vv||

    Symmetry factor example: 1
    """

    Nickel_index = "".join(graf.split(sep='SC = ')[0])
    Symmetry_factor = " ".join(graf.split(sep='SC = ')[1])
    # separating the Nickel index from the symmetry factor of the diagram

    Nickel_topology = "_".join(graf.split(sep='SC = ')[0].rstrip()
                               .split(sep=":")[0].split(sep="|"))[:-1]
    # topological part of the Nickel index

    Nickel_lines = "__".join(graf.split(sep='SC = ')[0].rstrip()
                            .split(sep=":")[1].split(sep="|"))[:-1]
    # line structure in the diagram corresponding to Nickel_topology

    return [f"Diagram__{Nickel_topology.strip()}__{Nickel_lines.strip()}.txt",
            Nickel_index.strip(), Symmetry_factor.strip()]


def get_list_with_propagators_from_nickel_index(
    nickel,
):
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
    """
    numbers individual blocks |...| in the topological part of the Nickel index 
    (all before the symbol :), i.e. vertices of the diagram
    """
    s2 = nickel.find(":")
    """
    runs through the part of the Nickel index describing the lines (after the symbol :)
    """
    propagator_internal = []
    propagator_external = []
    for i in nickel[: nickel.find(":")]:
        if i == "e":
            propagator_external += [[(-1, s1), ["0", nickel[s2 + 2]]]]
            s2 += 3
        elif i != "|":
            propagator_internal += [[(s1, int(i)),
                                     [nickel[s2 + 1], nickel[s2 + 2]]]]
            s2 += 3
        else:
            s1 += 1
    return [propagator_internal, propagator_external]

# ------------------------------------------------------------------------------------------------------------------#
#                       We get  a loop structure of the diagram (which lines form loops)
# ------------------------------------------------------------------------------------------------------------------#


def get_list_as_dictionary(
    list
):
    """
    Turns the list into a dictionary, keys are digits

    ARGUMENTS:

    Some list
    """
    dictionary = dict()
    for x in range(len(list)):
        dictionary.update(
            {x: list[x]}
        )
    return dictionary


def get_line_keywards_to_dictionary(
    some_dictionary
):
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


def list_of_all_possible_lines_combinations(
    dict_with_internal_lines
):
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

    list_of_loops = list()
    for i in range(len(dict_with_internal_lines) - 1):
        ordered_list_of_r_cardinality_subsets = list(
            itertools.combinations(dict_with_internal_lines.keys(), r=i + 2)
        )
        # returns an ordered list of ordered subsets of the given set,
        # starting with subsets of cardinality 2
        [
            list_of_loops.append(x) for x in ordered_list_of_r_cardinality_subsets
        ]
    return list_of_loops


def check_if_the_given_lines_combination_is_a_loop_in_diagram(
    list_of_all_possible_lines_combinations, dict_with_diagram_internal_lines
):
    """
    It checks if the given lines combination from list_of_all_possible_lines_combinations is a loop.
    The combination of lines is a loop <==> in the list of all vertices of the given lines
    (line = (vertex1, vertex2)), each vertex is repeated TWICE, i.e. each vertex is the end 
    of the previous line and the start of the next one.

    ARGUMENTS:

    list_of_all_possible_lines_combinations is given by the function list_of_all_possible_lines_combinations(),

    dict_with_diagram_internal_lines is given by the functions get_list_with_propagators_from_nickel_index()
    and get_list_as_dictionary()

    Note: for some technical reasons, we will assign new momentums (k and q, according to the list_of_momentums) 
    to propagators containing the D_v kernel, i.e. to propagators_with_helicity. Since each loop in the diagram 
    contains such helical propagator, we can describe the entire loop structure of the diagram by assigning a 
    new momentum to it in each loop.

    OUTPUT DATA EXAMPLE: (digit corresponds to the line from dict_with_diagram_internal_lines):

    [(0, 1, 2), (2, 3, 4), (0, 1, 3, 4)]
    """

    i = 0
    while i < len(list_of_all_possible_lines_combinations):
        list_of_list_of_vertices_for_ith_combination = [
            dict_with_diagram_internal_lines[k][0] for k in list_of_all_possible_lines_combinations[i]
        ]   # for a i-th combination from list_of_all_possible_lines_combinations we get a list of lines
        # (each digit from list_of_all_possible_lines_combinations is the key of the
        # dict_with_diagram_internal_lines, i.e. line)
        # the output is ((vertex,vertex), (vertex,vertex), ...)
        ordered_list_vith_all_diagram_vertices = list(
            itertools.chain.from_iterable(
                list_of_list_of_vertices_for_ith_combination)
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
    set_of_all_internal_propagators, set_of_propagators_with_helicity, list_of_momentums, list_of_frequencies
):
    """
    It assigning momentum (according to the list_of_momentums) to helicity propagators in the concret diagram.
    This function uses the information that there can only be one helical propagator in each loop.

    ARGUMENTS:

    set_of_all_internal_propagators (is given by the function get_list_with_propagators_from_nickel_index()),

    set_of_propagators_with_helicity, list_of_momentums, list_of_frequencies (see global variables)

    OUTPUT DATA EXAMPLE:

    dict_with_momentums_for_propagators_with_helicity = {1: k, 3: q}

    dict_with_frequencies_for_propagators_with_helicity = {1: w_k, 3: w_q}
    """
    dict_with_momentums_for_propagators_with_helicity = dict()
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
                if length == j:
                    dict_with_momentums_for_propagators_with_helicity.update(
                        {i: list_of_momentums[j]}
                    )
                    dict_with_frequencies_for_propagators_with_helicity.update(
                        {i: list_of_frequencies[j]}
                    )

    return [dict_with_momentums_for_propagators_with_helicity,
            dict_with_frequencies_for_propagators_with_helicity]


def get_usual_QFT_loops(
    list_of_loops, dict_with_momentums_for_propagators_with_helicity
):
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
            map(lambda x: test_loop.count(x),
                dict_with_momentums_for_propagators_with_helicity)
        )  # calculate the number of helicity propagators in a loop
        if number_of_helicity_propagators.count(1) != 1:
            # delete those loops that contain two (and more) helical propagators
            # note that each loop in the diagram contains at least one such propagator
            del list_of_usual_QFT_loops[i]
        else:
            i += 1
    return list_of_usual_QFT_loops

# ------------------------------------------------------------------------------------------------------------------#
#                      We get a distribution over momentums and frequencies flowing over lines
# ------------------------------------------------------------------------------------------------------------------#


def get_momentum_and_frequency_distribution(
    internal_lines, momentums_in_helical_propagators, frequencies_in_helical_propagators, 
    external_momentum, external_frequency, begin_vertex, end_vertex, number_int_vert
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
    momentums_for_all_propagators = [symbols(f'k_{i}') for i in range(length)]
    frequencies_for_all_propagators = [
        symbols(f'w_{i}') for i in range(length)]

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

    list_of_momentum_conservation_laws = [
        momentum_conservation_law[i] for i in range(number_int_vert)]
    list_of_arbitrary_momentums = [
        momentums_for_all_propagators[i] for i in range(length) if i not in momentums_in_helical_propagators
    ]
    list_of_frequency_conservation_laws = [
        frequency_conservation_law[i]for i in range(number_int_vert)]
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
    internal_lines, momentum_distribution, frequency_distribution, external_momentum, external_frequency,
    momentums_in_helical_propagators, frequencies_in_helical_propagators
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
            list_with_momentums[i] += momentum_distribution[i].subs(
                external_momentum, 0)
            momentum_distribution_at_zero_external_momentum.update(
                {i: list_with_momentums[i]})
        else:
            momentum_distribution_at_zero_external_momentum.update(
                {i: momentums_in_helical_propagators[i]})
        # frequency distribution at zero external frequency
        if i not in frequencies_in_helical_propagators:
            list_with_frequencies[i] += frequency_distribution[i].subs(
                external_frequency, 0)
            frequency_distribution_at_zero_external_frequency.update(
                {i: list_with_frequencies[i]})
        else:
            frequency_distribution_at_zero_external_frequency.update(
                {i: frequencies_in_helical_propagators[i]})

    return [momentum_distribution_at_zero_external_momentum, frequency_distribution_at_zero_external_frequency]


def momentum_and_frequency_distribution_at_vertexes(
    external_lines, internal_lines, number_of_all_vertices, external_momentum, external_frequency,
    momentum_distribution_at_zero_external_momentum, frequency_distribution_at_zero_external_frequency
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
                -1, "B", external_momentum, external_frequency]
            indexB = 3 * end_vertex  # save the index of the external field b
        else:
            data_for_vertexes_distribution[3 * end_vertex] = [
                -1, outflowing_field, -external_momentum, -external_frequency]
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
            propagator_key, outflowing_field, -
            momentum_distribution_at_zero_external_momentum[propagator_key],
            -frequency_distribution_at_zero_external_frequency[propagator_key]
        ]
        inflowing_data = [
            propagator_key, inflowing_field, momentum_distribution_at_zero_external_momentum[
                propagator_key],
            frequency_distribution_at_zero_external_frequency[propagator_key]
        ]

        if data_for_vertexes_distribution[begin_vertex * 3] == 0:
            data_for_vertexes_distribution[begin_vertex * 3] = outflowing_data
        elif data_for_vertexes_distribution[begin_vertex * 3 + 1] == 0:
            data_for_vertexes_distribution[begin_vertex *
                                           3 + 1] = outflowing_data
        else:
            data_for_vertexes_distribution[begin_vertex *
                                           3 + 2] = outflowing_data

        if data_for_vertexes_distribution[end_vertex * 3] == 0:
            data_for_vertexes_distribution[end_vertex * 3] = inflowing_data
        elif data_for_vertexes_distribution[end_vertex * 3 + 1] == 0:
            data_for_vertexes_distribution[end_vertex * 3 + 1] = inflowing_data
        else:
            data_for_vertexes_distribution[end_vertex * 3 + 2] = inflowing_data

    # change the keywords in the dictionary from numeric to string (for convenience)
    frequency_and_momentum_distribution_at_vertexes = dict()
    for i in range(number_of_all_vertices):
        frequency_and_momentum_distribution_at_vertexes['vertex',
                                                        i] = data_for_vertexes_distribution[3*i:3*(i + 1)]

    data_for_vertexes_momentum_distribution = [
        0] * (number_of_all_vertices * 3)
    for j in range(len(data_for_vertexes_distribution)):
        data_for_vertexes_momentum_distribution[j] = data_for_vertexes_distribution[j][0:3]

    return [indexB, indexb, data_for_vertexes_distribution,
            frequency_and_momentum_distribution_at_vertexes, data_for_vertexes_momentum_distribution]

# ------------------------------------------------------------------------------------------------------------------#
#                    Obtaining the integrand for the diagram (rational function and tensor part)
# ------------------------------------------------------------------------------------------------------------------#


def define_propagator_product(
    empty_P_data, empty_H_data, empty_numerator, empty_space, empty_propagator_data,
    fields_in_propagator, momentum_arg, frequency_arg, in1, in2
):
    """
    The function contains all the information about the propagators of the model.
    It is supposed to apply it to the list of propagators of each specific diagram 
    to obtain the corresponding integrand.

    ARGUMENTS:

    empty_P_data = ([]) -- list where information (momentum, frequency, indices) about
    projectors is stored,

    empty_H_data = ([]) -- list where information (momentum, frequency, indices) about 
    Helical terms is stored,

    empty_numerator = 1 -- factor by which the corresponding index structure of the propagator 
    is multiplied, 

    empty_space = "" -- empty string space where momentum and frequency arguments are stored,

    empty_propagator_data = 1 -- factor by which the corresponding propagator is multiplied 
    (without index structure),

    fields_in_propagator -- argument(["field1", "field2"]) passed to the function,

    momentum_arg, frequency_arg -- propagator arguments,

    in1, in2 -- indices of the propagator tensor structure
    """
    projector_argument_list = empty_P_data
    helical_argument_list = empty_H_data
    product_of_tensor_operators = empty_numerator
    interspace = empty_space
    product_of_propagators = empty_propagator_data

    match fields_in_propagator:
        case ["v", "v"]:
            product_of_tensor_operators *= (P(momentum_arg, in1, in2)
                                            + I * rho * H(momentum_arg, in1, in2))
            interspace = (
                interspace + f"Pvv[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                beta(momentum_arg, frequency_arg)*beta_star(momentum_arg, frequency_arg) *
                D_v(momentum_arg)/(xi(momentum_arg, frequency_arg) *
                                   xi_star(momentum_arg, frequency_arg))
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["v", "V"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (
                interspace + f"PvV[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                beta_star(momentum_arg, frequency_arg) /
                xi_star(momentum_arg, frequency_arg)
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["V", "v"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (
                interspace + f"PbB[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                beta_star(momentum_arg, frequency_arg) /
                xi_star(momentum_arg, frequency_arg)
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["b", "B"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (
                interspace + f"PbB[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                alpha_star(momentum_arg, frequency_arg) /
                xi_star(momentum_arg, frequency_arg)
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["B", "b"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (
                interspace + f"PBb[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                alpha_star(momentum_arg, frequency_arg) /
                xi_star(momentum_arg, frequency_arg)
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["v", "b"]:
            product_of_tensor_operators *= (P(momentum_arg, in1, in2)
                                            + I * rho * H(momentum_arg, in1, in2))
            interspace = (
                interspace + f"Pvb[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg,  in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                I*A*beta(momentum_arg, frequency_arg)*sc_prod(theta, momentum_arg) *
                D_v(momentum_arg)/(xi(momentum_arg, frequency_arg) *
                                   xi_star(momentum_arg, frequency_arg))
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["b", "v"]:
            product_of_tensor_operators *= (P(momentum_arg, in1, in2)
                                            + I * rho * H(momentum_arg, in1, in2))
            interspace = (
                interspace + f"Pbv[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                I*A*beta(momentum_arg, frequency_arg)*sc_prod(theta, momentum_arg) *
                D_v(momentum_arg)/(xi(momentum_arg, frequency_arg) *
                                   xi_star(momentum_arg, frequency_arg))
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["b", "b"]:
            product_of_tensor_operators *= (P(momentum_arg, in1, in2)
                                            + I * rho * H(momentum_arg, in1, in2))
            interspace = (
                interspace + f"Pbb[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            helical_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                A**2*sc_prod(theta, momentum_arg)**2*D_v(momentum_arg) /
                (xi(momentum_arg, frequency_arg) *
                 xi_star(momentum_arg, frequency_arg))
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["V", "b"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (interspace + "PVb[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                I*A*sc_prod(theta, momentum_arg) /
                xi_star(momentum_arg, frequency_arg)
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["b", "V"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (
                interspace + f"PbV[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                I*A*sc_prod(theta, momentum_arg) /
                xi_star(momentum_arg, frequency_arg)
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["B", "v"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (
                interspace + f"PBv[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                I*sc_prod(theta, momentum_arg) /
                xi_star(momentum_arg, frequency_arg)
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case ["v", "B"]:
            product_of_tensor_operators *= P(momentum_arg, in1, in2)
            interspace = (
                interspace + f"PvB[{momentum_arg}, {frequency_arg}]*")
            projector_argument_list += [[momentum_arg, in1, in2]]
            product_of_propagators *= (
                I*sc_prod(theta, momentum_arg) /
                xi_star(momentum_arg, frequency_arg)
            )

            return [product_of_tensor_operators, projector_argument_list,
                    helical_argument_list, interspace, product_of_propagators]
        case _:
            return sys.exit("Nickel index contains unknown propagator type")


def get_propagator_product(
    distribution_of_momentums_over_vertices, set_with_internal_lines,
    empty_P_data, empty_H_data, empty_numerator,
    empty_space, empty_propagator_data,
    momentum_distribution_with_zero_p, frequency_distribution_with_zero_w
):
    """
    This function applies the function define_propagator_product() to the list 
    with propagators of a particular diagram.

    ARGUMENTS:

    distribution_of_momentums_over_vertices is given by the function 
    momentum_and_frequency_distribution_at_vertexes(),

    set_with_internal_lines is given by the function get_list_with_propagators_from_nickel_index()

    empty_P_data = ([]) -- list where information (momentum, frequency, indices) about
    projectors is stored (this argument is passed to the function define_propagator_product()),

    empty_H_data = ([]) -- list where information (momentum, frequency, indices) about 
    Helical terms is stored (this argument is passed to the function define_propagator_product()),

    empty_numerator = 1 -- factor by which the corresponding index structure of the propagator 
    is multiplied (this argument is passed to the function define_propagator_product()), 

    empty_space = "" -- empty string space where momentum and frequency arguments are stored 
    (this argument is passed to the function define_propagator_product()),

    empty_propagator_data = 1 -- factor by which the corresponding propagator is multiplied 
    (without index structure) (this argument is passed to the function define_propagator_product()),

    momentum_distribution_with_zero_p, frequency_distribution_with_zero_w are given by the function
    get_momentum_and_frequency_distribution_at_zero_p_and_w()

    OUTPUT DATA EXAMPLE:

    product_of_tensor_operators = (I*rho*H(k, w_k, 2, 6) + P(k, w_k, 2, 6))*(I*rho*H(q, w_q, 5, 10) + 
    P(q, w_q, 5, 10))*P(-k, -w_k, 1, 3)*P(-q, -w_q, 8, 11)*P(-k - q, -w_k - w_q, 4, 7)

    projector_argument_list = [
    [-k, -w_k, 1, 3], [k, w_k, 2, 6], [-k - q, -w_k - w_q, 4, 7], [q, w_q, 5, 10], [-q, -w_q, 8, 11]
    ]

    helical_argument_list = [[k, w_k, 2, 6], [q, w_q, 5, 10]]

    propagator_product_for_Wolphram_Mathematica[:-1] = PbB[-k, -w_k]*Pvv[k, w_k]*PvB[-k - q, -w_k - w_q]*
    Pbb[q, w_q]*PbV[-q, -w_q]
    """

    # according to list distribution_of_momentums_over_vertices (vertices ordered) returns a list of indices
    # (see note in the description of momentum_and_frequency_distribution_at_vertexes())
    indexy = list(map(lambda x: x[0], distribution_of_momentums_over_vertices))

    for i in set_with_internal_lines:
        line = set_with_internal_lines[i]
        """
        lines are numbered by digits 
        the indexes in the indexy list are also digits, each of which occurs twice, 
        i.e. forms a line between the corresponding vertices 
        (each three indices in indexy belong to one vertex)
        """
        # .index(i) function returns the position of the first encountered element in the list
        in1 = indexy.index(i)
        # select the first (of two) index corresponding to line i (i encodes line in set_with_internal_lines)
        indexy[in1] = len(set_with_internal_lines)
        # rewrite in1 with a large enough number
        in2 = indexy.index(i)
        # select the secondgt (of two) index corresponding to line i (i encodes line in set_with_internal_lines)
        fields_in_propagator = line[1]
        momentum_arg = momentum_distribution_with_zero_p[i]
        frequency_arg = frequency_distribution_with_zero_w[i]

        all_structures_in_numerator = define_propagator_product(
            empty_P_data, empty_H_data, empty_numerator,
            empty_space, empty_propagator_data, fields_in_propagator,
            momentum_arg, frequency_arg, in1, in2
        )

        empty_numerator = all_structures_in_numerator[0]
        empty_P_data = all_structures_in_numerator[1]
        empty_H_data = all_structures_in_numerator[2]
        empty_space = all_structures_in_numerator[3]
        empty_propagator_data = all_structures_in_numerator[4]

    return [empty_numerator, empty_P_data,
            empty_H_data, empty_space[:-1],  # delete last symbol "*"
            empty_propagator_data]


def adding_vertex_factors_to_product_of_propagators(
    product_of_tensor_operators, Kronecker_delta_structure, momentum_structure,
    number_of_vertices, distribution_of_momentums_over_vertices
):
    """
    This function adds tensor vertex factors to the product of the tensor parts of the propagators. 
    Thus, this function completes the definition of the tensor part of the integrand of the corresponding diagram

    ARGUMENTS:

    product_of_tensor_operators is given by the function get_propagator_product(),

    Kronecker_delta_structure = ([]) -- list where information (indices) about
    vertex factors is stored,

    momentum_structure = ([]) -- list where information (momentums) about
    vertex factors is stored,

    number_of_vertices = 4 -- see global variables,

    distribution_of_momentums_over_vertices is given by the function 
    momentum_and_frequency_distribution_at_vertexes()
    """
    # according to list distribution_of_momentums_over_vertices (vertices ordered) returns a list of fields
    ordered_list_of_fields_flowing_from_vertices = list(
        map(lambda x: x[1], distribution_of_momentums_over_vertices))

    for vertex_number in range(number_of_vertices):

        vertex_triple = ordered_list_of_fields_flowing_from_vertices[
            3 * vertex_number: 3 * (vertex_number + 1)]  # field triple for corresponding vertex
        sorted_vertex_triple = sorted(
            vertex_triple, reverse=False)  # ascending sort

        match sorted_vertex_triple:
            case ["B", "b", "v",]:
                in1 = 3 * vertex_number + vertex_triple.index("B")
                in2 = 3 * vertex_number + vertex_triple.index("b")
                in3 = 3 * vertex_number + vertex_triple.index("v")
                Bbv = vertex_factor_Bbv(
                    distribution_of_momentums_over_vertices[in1][2], in1, in2, in3).doit()

                product_of_tensor_operators = product_of_tensor_operators * Bbv
                Kronecker_delta_structure.append([in1, in2])
                Kronecker_delta_structure.append([in1, in3])
                momentum_structure.append(
                    [distribution_of_momentums_over_vertices[in1][2], in3])
                momentum_structure.append(
                    [distribution_of_momentums_over_vertices[in1][2], in2])

            case ["V", "v", "v"]:
                in1 = 3 * vertex_number + vertex_triple.index("V")
                # since the two fields are the same, we don't know in advance what position in2 is in
                index_set = [3*vertex_number, 3 *
                             vertex_number + 1, 3*vertex_number + 2]
                index_set.remove(in1)
                in2 = index_set[0]
                in3 = index_set[1]
                Vvv = vertex_factor_Vvv(
                    distribution_of_momentums_over_vertices[in1][2], in1, in2, in3).doit()

                product_of_tensor_operators = product_of_tensor_operators * Vvv
                Kronecker_delta_structure.append([in1, in3])
                Kronecker_delta_structure.append([in1, in2])
                momentum_structure.append(
                    [distribution_of_momentums_over_vertices[in1][2], in2])
                momentum_structure.append(
                    [distribution_of_momentums_over_vertices[in1][2], in3])

            case ["V", "b", "b"]:
                in1 = 3 * vertex_number + vertex_triple.index("V")
                # since the two fields are the same, we don't know in advance what position in2 is in
                index_set = [3*vertex_number, 3 *
                             vertex_number + 1, 3*vertex_number + 2]
                index_set.remove(in1)
                in2 = index_set[0]
                in3 = index_set[1]
                Vbb = vertex_factor_Vvv(distribution_of_momentums_over_vertices[in1][2], in1, in2, in3
                                        ).doit()  # vertex_factor_Vvv = vertex_factor_Vbb by definiton

                product_of_tensor_operators = product_of_tensor_operators * Vbb
                Kronecker_delta_structure.append([in1, in3])
                Kronecker_delta_structure.append([in1, in2])
                momentum_structure.append(
                    [distribution_of_momentums_over_vertices[in1][2], in2])
                momentum_structure.append(
                    [distribution_of_momentums_over_vertices[in1][2], in3])
            case _:
                sys.exit("Unknown vertex type")
    return product_of_tensor_operators, Kronecker_delta_structure, momentum_structure

# ------------------------------------------------------------------------------------------------------------------#
#                                          Сomputing integrals over frequencies
# ------------------------------------------------------------------------------------------------------------------#

# Integrals over frequency are calculated using the residue theorem


def get_info_how_to_close_contour(
    rational_function, variable1
):
    """
    This function evaluates the complexity of calculating integrals of two variables using residues. 
    The estimation is made on the basis of counting the number of poles lying in the upper/lower
    complex half-plane for each variable, respectively. To do this, for each variable, 
    two counters corresponding to the closing of the loop up down are created. 
    Their minimum gives the desired criterion.

    ARGUMENTS:

    rational_function -- a rational function whose denominator consists of the product of the powers
    of the functions chi_1 and chi_2,

    variable1 -- variable over which the integral is first calculated

    Note:

    This function uses information about the structure of the singularities of the functions 1/chi_1(k, w) 
    and 1/chi_2(k, w) with respect to the variable w. Under the condition that u is positive, 
    it can be shown that the position of the pole is determined only by the sign of w in chi_1, chi_2
    (the functions f_1, f_2 do not change the sign of the im(w) for any k)

    OUTPUT DATA EXAMPLE:

    computational_complexity_when_closing_UP_contour_in_plane_of_variable1 = 6

    computational_complexity_when_closing_DOWN_contour_in_plane_of_variable1 = 2

    computational_complexity_when_closing_UP_contour_in_plane_of_variable2 = 4

    computational_complexity_when_closing_DOWN_contour_in_plane_of_variable2 = 2
    """

    denominator = fraction(rational_function)[1]
    denominator_structure = factor_list(denominator)[1]
    number_of_multipliers = len(denominator_structure)

    computational_complexity_when_closing_UP_contour_in_plane_of_variable1 = 0
    computational_complexity_when_closing_DOWN_contour_in_plane_of_variable1 = 0

    computational_complexity_when_closing_UP_contour_in_plane_of_variable2 = 0
    computational_complexity_when_closing_DOWN_contour_in_plane_of_variable2 = 0

    for pole in range(number_of_multipliers):
        pole_equation = denominator_structure[pole][0]
        multiplicity = denominator_structure[pole][1]
        frequency = pole_equation.args[1]
        if frequency.has(variable1) == True:
            if frequency.could_extract_minus_sign() == True:
                # if this condition is fit, then the pole lies in the lower half-plane
                # (see note in the function description)
                computational_complexity_when_closing_DOWN_contour_in_plane_of_variable1 += multiplicity
            else:
                computational_complexity_when_closing_UP_contour_in_plane_of_variable1 += multiplicity
        else:
            if frequency.could_extract_minus_sign() == True:
                # if this condition is fit, then the pole lies in the lower half-plane
                # (see note in the function description)
                computational_complexity_when_closing_DOWN_contour_in_plane_of_variable2 += multiplicity
            else:
                computational_complexity_when_closing_UP_contour_in_plane_of_variable2 += multiplicity

    return [[computational_complexity_when_closing_UP_contour_in_plane_of_variable1,
            computational_complexity_when_closing_DOWN_contour_in_plane_of_variable1],
            [computational_complexity_when_closing_UP_contour_in_plane_of_variable2,
            computational_complexity_when_closing_DOWN_contour_in_plane_of_variable2]]


def calculate_residues_sum(
        rational_function, main_variable, parameter):
    """
    This function calculates the sum of the residues (symbolically, at the level of functions f_1, f_2)
    of a rational function (without 2*pi*I factor) with respect to one given variable.

    GENERAL REQUIREMENTS FOR A RATIONAL FUNCTION:

    The denominator of rational_function must be the product of the powers of the chi_1.doit() and chi_2.doit(), 
    i.e. be expressed in terms of functions f_1, f_2. 
    The result of calculating the residues sum is also expressed in terms of f_1 and f_2.

    ARGUMENTS:

    rational_function -- a rational function whose denominator consists of the product of the powers
    of the functions chi_1.doit() and chi_2.doit(),

    main_variable -- variable over which the residues are calculated

    parameter -- other variables (which may not be, then this argument does not affect anything)

    Note:

    Calculating the residue from a standard formula requires a pole reduction operation 
    by multiplying the fraction by the corresponding expression. Here raises a problem:
    ((x-a)/(-x + a)).subs(x, a) = 0. In order not to use the simplify() function 
    (it takes a very long time), you need to bring the expression exactly to the form
    (x-a) /(-x + a) = -(x-a)/(-x + a), then -((x-a)/(x - a)).subs(x, a) = -1

    OUTPUT DATA EXAMPLE:

    As an example, here is represented the result for a simpler function than the characteristic diagram integrand. 
    Nevertheless, the reduced function contains all the characteristic features of the corresponding integrands.

    rational_function = (w_k - 2)/(chi_1(k, w_k)**2*chi_1(q, -w_q-w_k)**2*chi_2(k, -w_k))).doit()

    sum_of_residues_in_upper_half_plane = 
    2*(f_1(k, A) - 2)/((-f_1(k, A) - f_2(k, A))*(-w_q - f_1(k, A) - f_1(q, A))**3) + 
    1/((-f_1(k, A) - f_2(k, A))*(-w_q - f_1(k, A) - f_1(q, A))**2) + 
    (f_1(k, A) - 2)/((-f_1(k, A) - f_2(k, A))**2*(-w_q - f_1(k, A) - f_1(q, A))**2)

    sum_of_residues_in_lower_half_plane = 
    (-w_q - f_1(q, A) - 2)/((-w_q - f_1(k, A) - f_1(q, A))**2*(w_q + f_1(q, A) - f_2(k, A))**2)
    + 1/((-w_q - f_1(k, A) - f_1(q, A))**2*(w_q + f_1(q, A) - f_2(k, A))) - 
    2*(-w_q - f_1(q, A) - 2)/((-w_q - f_1(k, A) - f_1(q, A))**3*(w_q + f_1(q, A) - f_2(k, A))) - 
    (-f_2(k, A) - 2)/((-f_1(k, A) - f_2(k, A))**2*(-w_q - f_1(q, A) + f_2(k, A))**2)

    sum_of_residues_in_upper_half_plane + sum_of_residues_in_lower_half_plane = 0, 
    since the residue of the corresponding function at infinity is 0
    """
    numerator = fraction(rational_function)[0]
    denominator = fraction(rational_function)[1]
    denominator_structure = factor_list(denominator)[1]
    number_of_multipliers = len(denominator_structure)

    residues_in_upper_half_plane = list()
    residues_in_lower_half_plane = list()

    for pole in range(number_of_multipliers):
        pole_equation = denominator_structure[pole][0]
        multiplicity = denominator_structure[pole][1]

        if pole_equation.has(main_variable) == True:
            # we solve a linear equation of the form w - smth = 0
            # solve_linear() function, unlike solve(), is less sensitive to the type of input data
            solution_of_pole_equation = solve_linear(
                pole_equation, symbols=[main_variable])[1]
            # see note in description
            if denominator.has(main_variable - solution_of_pole_equation) == True:
                calcullable_function = (
                    (numerator*(main_variable - solution_of_pole_equation)**multiplicity)/denominator).diff(
                    main_variable, multiplicity - 1)
                residue_by_variable = calcullable_function.subs(
                    main_variable, solution_of_pole_equation)
                # residue cannot be zero
                if residue_by_variable == 0:
                    sys.exit(
                        "An error has been detected. The residue turned out to be zero.")

            else:
                calcullable_function = (
                    (numerator*(-main_variable + solution_of_pole_equation)**multiplicity)/denominator).diff(
                    main_variable, multiplicity - 1)
                residue_by_variable = (-1)**multiplicity*calcullable_function.subs(
                    main_variable, solution_of_pole_equation)
                # residue cannot be zero
                if residue_by_variable == 0:
                    sys.exit(
                        "An error has been detected. The residue turned out to be zero.")

            if solution_of_pole_equation.subs(parameter, 0).could_extract_minus_sign() == True:
                residues_in_lower_half_plane.append(residue_by_variable)
            else:
                residues_in_upper_half_plane.append(residue_by_variable)
        # Error handling in this function takes several orders of magnitude longer than the
        # execution of the program itself

        # if simplify(sum(residues_in_upper_half_plane) + sum(residues_in_lower_half_plane)) != 0:
            # the residue at infinity is usually 0
        #     sys.exit("Error when calculating residues")

    sum_of_residues_in_upper_half_plane = sum(residues_in_upper_half_plane)
    sum_of_residues_in_lower_half_plane = sum(residues_in_lower_half_plane)

    return sum_of_residues_in_upper_half_plane, sum_of_residues_in_lower_half_plane


def calculating_frequency_integrals_in_two_loop_diagrams(
    integrand_with_denominator_from_xi_functions, frequency1, frequency2
):
    """
    This function calculates integrals over frequencies using the residue theorem

    GENERAL REQUIREMENTS FOR AN INTEGRAND:

    The integrand_with_denominator_from_xi_functions denominator must be the product of the xi_1 and xi_2 powers.
    (Further, the .doit() operation is applied to integrand_with_denominator_from_xi_functions)

    ARGUMENTS:

    integrand_with_denominator_from_xi_functions -- a rational function whose denominator consists of the 
    product of the powers of the functions xi_1 and xi_2,

    frequency1 -- variable over which the integral is calculated a first time 

    frequency2 -- variable over which the integral is calculated a second time 
    """

    integrand_with_denominator_from_chi_functions = integrand_with_denominator_from_xi_functions.doit()

    computational_complexity_estimation = get_info_how_to_close_contour(
        integrand_with_denominator_from_chi_functions, frequency1)

    integrand_with_denominator_from_f12_functions = integrand_with_denominator_from_chi_functions.doit()

    first_factor_2piI = 1
    second_factor_2piI = 1

    # calculate the residues for frequency1
    if computational_complexity_estimation[0][0] >= computational_complexity_estimation[0][1]:
        # computational complexity when closing the contour up >=
        # computational complexity when closing the contour down
        # i.e. it is advantageous to close the contour down
        residues_sum_without_2piI = calculate_residues_sum(
            integrand_with_denominator_from_f12_functions, frequency1, frequency2
        )[1]
        first_factor_2piI = -2*pi*I
    else:
        # it is advantageous to close the contour up
        residues_sum_without_2piI = calculate_residues_sum(
            integrand_with_denominator_from_f12_functions, frequency1, frequency2
        )[0]
        first_factor_2piI = 2*pi*I

    number_of_terms_in_sum = len(residues_sum_without_2piI.args)

    # calculate the residues for frequency2
    list_with_residues_sum_for_frequency2 = [0] * number_of_terms_in_sum
    if computational_complexity_estimation[1][0] >= computational_complexity_estimation[1][1]:
        # computational complexity when closing the contour up >=
        # computational complexity when closing the contour down
        # i.e. it is advantageous to close the contour down
        for i in range(number_of_terms_in_sum):
            term = residues_sum_without_2piI.args[i]
            list_with_residues_sum_for_frequency2[i] = calculate_residues_sum(
                term, frequency2, frequency1)[1]
        second_factor_2piI = -2*pi*I
    else:
        # it is advantageous to close the contour up
        for i in range(number_of_terms_in_sum):
            term = residues_sum_without_2piI.args[i]
            list_with_residues_sum_for_frequency2[i] = calculate_residues_sum(
                term, frequency2, frequency1)[0]
        second_factor_2piI = 2*pi*I

    total_sum_of_residues_for_both_frequencies = first_factor_2piI*second_factor_2piI*sum(
        list_with_residues_sum_for_frequency2)

    return total_sum_of_residues_for_both_frequencies


# ------------------------------------------------------------------------------------------------------------------#
#                             Calculation of the tensor part of the integrand for a diagram
# ------------------------------------------------------------------------------------------------------------------#

def dosad(
    zoznam, ind_hod, struktura, pozicia
):  # ?????????
    if ind_hod in zoznam:
        return zoznam
    elif ind_hod not in struktura:
        return list()
    elif zoznam[pozicia] != -1:
        return list()
    elif len(zoznam) - 1 == pozicia:
        return zoznam[:pozicia] + [ind_hod]
    else:
        return zoznam[:pozicia] + [ind_hod] + zoznam[pozicia + 1:]

# ------------------------------------------------------------------------------------------------------------------#
#                               Create a file with general notation and information
# ------------------------------------------------------------------------------------------------------------------#


def get_propagators_from_list_of_fields(
    fields_for_propagators
):
    """
    Glues separate fields indexes from propagators_with_helicity into the list of propagators

    Output data example: 

    list_of_propagators_with_helicity = ['vv', 'vb', 'bv', 'bb']
    """
    dimension = len(fields_for_propagators)
    list_of_propagators = [0] * dimension
    for i in range(dimension):
        list_of_propagators[i] = (
            fields_for_propagators[i][0] + fields_for_propagators[i][1])
    return list_of_propagators


def create_file_with_info_and_supplementary_matherials():
    """
    Creates a file with general information and supplementary matherials
    """
    with open('General_notation.txt', 'w+') as Notation_file:

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

        all_fields_glued_into_propagators = get_propagators_from_list_of_fields(
            all_nonzero_propagators)

        info_about_propagators = list()
        for m in range(len(all_nonzero_propagators)):
            info_about_propagators.append(define_propagator_product(
                ([]), ([]), 1, '', 1, all_nonzero_propagators[m], k, w_k, i, j))
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
function hyb(k, i) denotes i-th component of vector {k} (i = 1, ..., d), functions kd(i, j) and lcs(i, j, l) 
denotes the Kronecker delta and Levi-Civita symbols. 
{z} = cos(angle between k and q) = sc_prod(k, q)/ (abs(k) * abs(q)), vector {theta} is proportional to the 
magnetic induction, {nu} is a renormalized kinematic viscosity, {mu} is a renormalization mass, 
{u} is a renormalized reciprocal magnetic Prandtl number, {rho} is a gyrotropy parameter (abs(rho) < 1), 
g is a coupling constant, {eps} determines a degree of model deviation from logarithmicity (0 < eps =< 2). \n
"""
            f"\nD_v(k) = {D_v(k).doit()}\n"
            f"\nalpha(k, w) = {alpha(k, w).doit()}\n"
            f"\nalpha_star(k, w) = alpha*(k, w) = {alpha_star(k, w).doit()}\n"
            f"\nbeta(k, w) = {beta(k, w).doit()}\n"
            f"\nbeta_star(k, w) = beta*(k, w) = {beta_star(k, w).doit()}\n"
            f"\nf_1(k, w) = {f_1(k, A).doit().doit()}\n"
            f"\nf_2(k, w) = {f_2(k, A).doit().doit()}\n"
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


# ------------------------------------------------------------------------------------------------------------------#
#                                Computing the diagram (the major part of the program)
# ------------------------------------------------------------------------------------------------------------------#

def get_info_about_diagram(graf):

    # --------------------------------------------------------------------------------------------------------------#
    #                     Create a file and start write the information about diagram into it
    # --------------------------------------------------------------------------------------------------------------#

    output_file_name = get_information_from_Nickel_index(
        graf
    )[0]  # according to the given Nickel index of the diagram, create the name of the file with the results
    nickel_index = get_information_from_Nickel_index(
        graf
    )[1]  # get Nickel index from the line with the data
    symmetry_coefficient = get_information_from_Nickel_index(
        graf
    )[2]  # get symmetry factor from the line with the data

    Fey_graphs = open(
        f"Results/{output_file_name}", "a"
    )  # creating a file with all output data for the corresponding diagram

    print(
        f"\nNickel index of the Feynman diagram: {nickel_index}"
        ) # display the Nickel index of the diagram

    Fey_graphs.write(
        f"Nickel index of the Feynman diagram: {nickel_index} \n"
    )  # write the Nickel index to the file

    Fey_graphs.write(
        f"\nDiagram symmetry factor: {symmetry_coefficient} \n"
    )  # write the symmetry coefficient to the file

    # --------------------------------------------------------------------------------------------------------------#
    #                         Part 1: Geting the diagram description (lines, vertices, etc.)
    # --------------------------------------------------------------------------------------------------------------#

    Fey_graphs.write(
        f"\nSupporting information begin:\n"
    )  # start filling the supporting information (topology, momentum and frequency distribution) to file

    # --------------------------------------------------------------------------------------------------------------#
    #                Define a loop structure of the diagram (which lines form loops) and write it into file
    # --------------------------------------------------------------------------------------------------------------#

    internal_lines = get_list_with_propagators_from_nickel_index(
        graf)[0]  # list with diagram internal lines

    dict_with_internal_lines = get_list_as_dictionary(
        internal_lines
    )  # put the list of all internal lines in the diagram to a dictionary

    Fey_graphs.write(
        f"\nPropagators in the diagram: \n"
        f"{get_line_keywards_to_dictionary(dict_with_internal_lines)} \n"
    )  # write the dictionary with all internal lines to the file

    list_of_all_loops_in_diagram = check_if_the_given_lines_combination_is_a_loop_in_diagram(
        list_of_all_possible_lines_combinations(
            dict_with_internal_lines), dict_with_internal_lines
    )  # get list of all loops in the diagram (this function works for diagrams with any number of loops)

    momentums_in_helical_propagators = put_momentums_and_frequencies_to_propagators_with_helicity(
        dict_with_internal_lines, propagators_with_helicity,
        momentums_for_helicity_propagators, frequencies_for_helicity_propagators
    )[0]  # create a dictionary for momentums flowing in lines containing kernel D_v

    loop = get_usual_QFT_loops(
        list_of_all_loops_in_diagram, momentums_in_helical_propagators
    )  # select only those loops that contain only one helical propagator (usual QFT loops)

    Fey_graphs.write(
        f"\nLoops in the diagram for a given internal momentum "
        f"(digit corresponds to the line from previous dictionary): \n{loop} \n"
    )  # write the loop structure of the diagram to the file

    # --------------------------------------------------------------------------------------------------------------#
    #                      Get a distribution over momentums and frequencies flowing over lines
    # --------------------------------------------------------------------------------------------------------------#

    frequencies_in_helical_propagators = put_momentums_and_frequencies_to_propagators_with_helicity(
        dict_with_internal_lines, propagators_with_helicity,
        momentums_for_helicity_propagators, frequencies_for_helicity_propagators
    )[1]  # create a dictionary with frequency arguments for propagators defining loops

    # determine the start and end vertices in the diagram
    vertex_begin = 0
    vertex_end = number_int_vert - 1

    momentum_and_frequency_distribution = get_momentum_and_frequency_distribution(
        dict_with_internal_lines, momentums_in_helical_propagators, frequencies_in_helical_propagators,
        p, w, vertex_begin, vertex_end, number_int_vert
    )  # assign momentums and frequencies to the corresponding lines of the diagram

    momentum_distribution = momentum_and_frequency_distribution[0]
    # dictionary with momentums distributed along lines
    frequency_distribution = momentum_and_frequency_distribution[1]
    # dictionary with frequencies distributed along lines

    propagator_args_distribution_at_zero_p_and_w = get_momentum_and_frequency_distribution_at_zero_p_and_w(
        dict_with_internal_lines, momentum_distribution, frequency_distribution, p, w,
        momentums_for_helicity_propagators, frequencies_for_helicity_propagators
    )  # obtain the distribution of momentums and frequencies along the lines in the diagram
    # at zero external arguments

    momentum_distribution_at_zero_external_momentum = propagator_args_distribution_at_zero_p_and_w[
        0]
    # dictionary with momentums distributed along lines (at zero p)
    frequency_distribution_at_zero_external_frequency = propagator_args_distribution_at_zero_p_and_w[
        1]
    # dictionary with frequencies distributed along lines (at zero w)

    Fey_graphs.write(
        f"\nMomentum propagating along the lines: "
        f"\n{get_line_keywards_to_dictionary(momentum_distribution)}\n"
    )

    Fey_graphs.write(
        f"\nFrequency propagating along the lines: "
        f"\n{get_line_keywards_to_dictionary(frequency_distribution)}\n"
    )

    external_lines = get_list_with_propagators_from_nickel_index(
        graf)[1]  # list with diagram external lines

    distribution_of_diagram_parameters_over_vertices = momentum_and_frequency_distribution_at_vertexes(
        external_lines, dict_with_internal_lines, number_int_vert, p, w,
        momentum_distribution_at_zero_external_momentum, frequency_distribution_at_zero_external_frequency
    )  # all information about the diagram is collected and summarized
    # (which fields and with which arguments form pairs (lines in the diagram))

    indexB = distribution_of_diagram_parameters_over_vertices[0]

    indexb = distribution_of_diagram_parameters_over_vertices[1]

    frequency_and_momentum_distribution_at_vertexes = distribution_of_diagram_parameters_over_vertices[
        3]

    moznost = distribution_of_diagram_parameters_over_vertices[4]

    Fey_graphs.write(
        f"\nMomentum and frequency distribution at the vertices: "
        f"\n{frequency_and_momentum_distribution_at_vertexes} \n"
    )

    # --------------------------------------------------------------------------------------------------------------#
    #                   Obtaining the integrand for the diagram (rational function and tensor part)
    # --------------------------------------------------------------------------------------------------------------#

    # here we save the tensor structure
    Tenzor = 1 

    # here we save the product of propagators (without tensor structure)
    Product = 1

    # here we save all indices of the projctors in the form [[momentum, index1, index2]]
    P_structure = ([])

    # here we save all indices of the helical structures in the form [[momentum, index1, index2]]
    H_structure = ([])

    # here we save the propagator product argument structure (for Wolfram Mathematica file)
    propagator_product_for_WfMath = ''

    structure_of_propagator_product = get_propagator_product(
        moznost, dict_with_internal_lines, P_structure, H_structure, Tenzor,
        propagator_product_for_WfMath, Product, momentum_distribution_at_zero_external_momentum,
        frequency_distribution_at_zero_external_frequency)

    Tenzor = structure_of_propagator_product[0]

    P_structure = structure_of_propagator_product[1]

    H_structure = structure_of_propagator_product[2]

    propagator_product_for_WfMath = structure_of_propagator_product[3]

    Product = structure_of_propagator_product[4]

    Fey_graphs.write(
        f"\nArgument structure in the propagator product: \n{propagator_product_for_WfMath}\n"
    )

    print(
        f"\nProduct of propagators without tensor structure: \n{Product}"
        )

    Fey_graphs.write(
        f"\nProduct of propagators without tensor structure: \n{Product}\n"
    )

    # here we save all indices in Kronecker delta in the form [ [index 1, index 2]]
    kd_structure = ([])
    # here we save all momentums and their components in the form [ [ k, i] ]
    hyb_structure = ([])

    whole_tensor_structure_of_integrand_numerator = adding_vertex_factors_to_product_of_propagators(
        Tenzor, kd_structure, hyb_structure, number_int_vert, moznost)

    Tenzor = whole_tensor_structure_of_integrand_numerator[0]

    kd_structure = whole_tensor_structure_of_integrand_numerator[1]

    hyb_structure = whole_tensor_structure_of_integrand_numerator[2]

    print(
        f"\nDiagram tensor structure before computing tensor convolutions: \n{Tenzor}"
        )

    Fey_graphs.write(
        f"\nDiagram tensor structure before computing tensor convolutions: \n{Tenzor}\n"
    )

    Fey_graphs.write(
        f"\nSupporting information end.\n"
    )  # finish filling the supporting information to file

    # --------------------------------------------------------------------------------------------------------------#
    #                Part 2. Diagram calculation (integrals over frequencies, tensor convolutions, etc.)
    # --------------------------------------------------------------------------------------------------------------#

    Fey_graphs.write(
        f"\nDiagram calculation begin:\n"
    )  # starts filling the results of calculations (integrals over frequencies, tensor convolutions) to file

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing integrals over frequencies
    # --------------------------------------------------------------------------------------------------------------#

    total_sum_of_residues_for_both_frequencies = calculating_frequency_integrals_in_two_loop_diagrams(
        Product, w_k, w_q)

    Fey_graphs.write(
        f"\nThe integrand (after computing integrals over frequencies): "
        f"\n{total_sum_of_residues_for_both_frequencies} \n"
    )

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing diagram tensor structure
    # --------------------------------------------------------------------------------------------------------------#

    print(
        f"\nBeginning the tensor convolutions calculation: \n"
        )

    t = time.time()  # it is only used to calculate the calculation time -- can be omitted

    Tenzor = expand(Tenzor)  # The final tesor structure from the diagram.

    # What I need from the Tenzor structure
    Tenzor = rho * Tenzor.coeff(rho**stupen)
    Tenzor = expand(Tenzor.subs(I**5, I))  # calculate the imaginary unit
    # Tenzor = Tenzor.subs(A, 1)              # It depends on which part we want to calculate from the vertex Bbv
    # print(Tenzor)
    # We are interested in the leading (proportional to p) contribution to the diagram asymptotic, when p --> 0.
    print(f"step 0: {round(time.time() - t, 1)} sec")

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

    print(f"step 1: {round(time.time() - t, 1)} sec")

    i = 0
    # discard from the Tensor structure what is zero for the projection operator P_ij (k) * k_i = 0
    while i < len(P_structure):
        in1 = P_structure[i]
        if Tenzor.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(
                P(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(
                P(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(
                P(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(
                P(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[2]), 0)
        if Tenzor.coeff(P(in1[0], in1[1], in1[2])) == 0:
            P_structure.remove(in1)
        else:
            if (
                in1[0] == -k or in1[0] == -q
            ):  # Replace in the tensor structure in the projection operators:  P(-k,i,j) = P(k,i,j)
                Tenzor = Tenzor.subs(
                    P(in1[0], in1[1], in1[2]), P(-in1[0], in1[1], in1[2]))
                P_structure[i][0] = -in1[0]
            i += 1

    i = 0
    # discard from the Tensor structure what is zero for the helical operator H_ij (k) * k_i = 0
    while i < len(H_structure):
        in1 = H_structure[i]
        if Tenzor.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[2]), 0)
        if Tenzor.coeff(H(in1[0], in1[1], in1[2])) == 0:
            H_structure.remove(in1)
        else:
            i += 1

    print(f"step 2: {round(time.time() - t, 1)} sec")

    i = 0
    # sipmplify in the Tenzor part H_{ij} (k) P_{il} (k) =  H_{il} (k)
    while (len(H_structure) > i):
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

    print(f"step 3: {round(time.time() - t, 1)} sec")

    i = 0
    # sipmplify in the Tenzor part  P_{ij} (k) P_{il} (k) =  P_{il} (k)
    while (len(P_structure) > i):
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

    print(f"step 4: {round(time.time() - t, 1)} sec")

    for i in hyb_structure:  # replace: hyb(-k+q, i) = -hyb(k, i) + hyb(q, i)
        k_c = i[0].coeff(k)
        q_c = i[0].coeff(q)
        if k_c != 0 or q_c != 0:
            Tenzor = Tenzor.subs(
                hyb(i[0], i[1]), (k_c * hyb(k, i[1]) + q_c * hyb(q, i[1])))

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

    print(f"step 5: {round(time.time() - t, 1)} sec")

    Tenzor = expand(Tenzor)

    # discard from the Tensor structure what is zero for the helical operator H_{ij} (k) * k_i = 0
    for (in1) in (H_structure):
        clen = Tenzor.coeff(H(in1[0], in1[1], in1[2]))
        if clen.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        if clen.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        if in1[0] == k and clen.coeff(hyb(q, in1[1]) * hyb(q, in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(q, in1[1]) * hyb(q, in1[2]), 0
            )
        if in1[0] == q and clen.coeff(hyb(k, in1[1]) * hyb(k, in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(k, in1[1]) * hyb(k, in1[2]), 0
            )

    print(f"step 6: {round(time.time() - t, 1)} sec")

    inkd = 0
    while (inkd == 0):  # calculation part connected with the kronecker delta function: kd(i,j) *hyb(k,i) = hyb(k,j)
        for (
            in1
        ) in (
            kd_structure
        ):  # beware, I not treat the case if there remains a delta function with indexes of external fields !!
            clen = Tenzor.coeff(kd(in1[0], in1[1]))
            if clen.coeff(hyb(k, in1[0])) != 0:
                Tenzor = Tenzor.subs(
                    kd(in1[0], in1[1]) * hyb(k, in1[0]), hyb(k, in1[1]))
            if clen.coeff(hyb(k, in1[1])) != 0:
                Tenzor = Tenzor.subs(
                    kd(in1[0], in1[1]) * hyb(k, in1[1]), hyb(k, in1[0]))
            if clen.coeff(hyb(q, in1[0])) != 0:
                Tenzor = Tenzor.subs(
                    kd(in1[0], in1[1]) * hyb(q, in1[0]), hyb(q, in1[1]))
            if clen.coeff(hyb(q, in1[1])) != 0:
                Tenzor = Tenzor.subs(
                    kd(in1[0], in1[1]) * hyb(q, in1[1]), hyb(q, in1[0]))
            if clen.coeff(hyb(p, in1[0])) != 0:
                Tenzor = Tenzor.subs(
                    kd(in1[0], in1[1]) * hyb(p, in1[0]), hyb(p, in1[1]))
            if clen.coeff(hyb(p, in1[1])) != 0:
                Tenzor = Tenzor.subs(
                    kd(in1[0], in1[1]) * hyb(p, in1[1]), hyb(p, in1[0]))
            if Tenzor.coeff(kd(in1[0], in1[1])) == 0:
                kd_structure.remove(in1)
                inkd += 1
        if inkd != 0:
            inkd = 0
        else:
            inkd = 1

    print(f"step 7: {round(time.time() - t, 1)} sec")

    i = 0
    while len(H_structure) > i:  # calculation for helical term
        in1 = H_structure[i]
        clen = Tenzor.coeff(H(in1[0], in1[1], in1[2]))
        if (
            clen.coeff(hyb(in1[0], in1[1])) != 0
        ):  # I throw out the part:  H (k,i,j) hyb(k,i) = 0
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        if clen.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
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

    print(f"step 8: {round(time.time() - t, 1)} sec")

    p_structure = list()  # list of indeces for momentum p in Tenzor
    k_structure = list()  # list of indeces for momentum k in Tenzor
    q_structure = list()  # list of indeces for momentum q in Tenzor
    # It combines quantities with matching indices.
    for in1 in range(len(moznost)):
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

    print(f"step 9: {round(time.time() - t, 1)} sec")

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
        )  # I know which indexes are free. I know where the fields B or b are located.
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

    print(f"step 10: {round(time.time() - t, 1)} sec")

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

    print(f"step 11: {round(time.time() - t, 1)} sec")

    print(
        f"\nDiagram tensor structure after computing tensor convolutions: \n{Tenzor}"
        )

    Fey_graphs.write(
        f"\nDiagram tensor structure after computing tensor convolutions: \n{Tenzor} \n"
    )

    # result = str(Tenzor)
    # result = result.replace("**", "^")

    Fey_graphs.write(
        f"\nDiagram calculation end.\n"
    )  # finish  filling the results of calculation to file
    
    # print(pretty(Tenzor, use_unicode=False))

    # Fey_graphs.write("\n"+ pretty(Tenzor, use_unicode=False) + "\n")

    # Fey_graphs.write("\n"+ latex(Tenzor) + "\n")

    Fey_graphs.close()

# ------------------------------------------------------------------------------------------------------------------#
#                                         Computing all two-loop MHD diagrams
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

    number_of_counted_diagrams = 0 # counter to count already processed diagrams

    with open('Two-loop MHD diagrams.txt', 'r') as MHD_diagrams_file:

        for graf in MHD_diagrams_file.readlines():
        
            print(f"CALCULATION {number_of_counted_diagrams} BEGIN")

            get_info_about_diagram(graf)

            print(f"\nCALCULATION {number_of_counted_diagrams} END \n")

            number_of_counted_diagrams += 1

    print(f"Number of counted diagrams: {number_of_counted_diagrams}")

# ------------------------------------------------------------------------------------------------------------------#
#                                                    Entry point 
# ------------------------------------------------------------------------------------------------------------------#

if __name__ == '__main__':
    main()
