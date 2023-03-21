from sympy import *
from Functions.Global_variables import *

# -----------------------------------------------------------------------------------------------------------------#
#                                       Custom functions in subclass Function in SymPy
# -----------------------------------------------------------------------------------------------------------------#

# All functions here are given in momentum-frequency representation

# ATTENTION!!! The sign of the Fourier transform was chosen to be the same as in [1] (see main file)


class D_v(Function):
    """
    D_v(k) = go*nuo**3*k**(4 - d - 2*eps)

    ARGUMENTS:

    k -- absolute value of momentum

    PARAMETERS:

    go -- bare coupling constant, nuo -- bare kinematic viscosity, d -- space dimension,
    eps -- determines a degree of model deviation from logarithmicity

    PROPERTIES:

    D_v(k) = D_v(-k)
    """

    @classmethod
    def eval(cls, mom):

        global momentums_for_helicity_propagators
        global B, nuo, d, eps

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(-mom)
        if mom == B * mom1 / nuo:
            return (B / nuo) ** (4 - d - 2 * eps) * cls(mom1)
        if mom == B * mom2 / nuo:
            return (B / nuo) ** (4 - d - 2 * eps) * cls(mom2)

    def doit(self, deep=True, **hints):
        mom = self.args[0]

        if deep:
            mom = mom.doit(deep=deep, **hints)
        return go * nuo**3 * mom ** (4 - d - 2 * eps)


class alpha(Function):
    """
    alpha(nuo, k, w) = I*w + nuo*k**2

    ARGUMENTS:

    nuo -- bare kinematic viscosity,  k -- momentum, w - frequency

    PROPERTIES:

    alpha(nuo, k, w) = alpha(nuo, -k, w)
    """

    @classmethod
    def eval(cls, nuo, mom, freq):

        global momentums_for_helicity_propagators
        global B

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        mom = expand(mom)

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(nuo, -mom, freq)

        # define scaling properties

        mom_arg_1 = B * mom1 / nuo
        mom_arg_2 = B * mom2 / nuo

        freq_value_f1_mom1 = B**2 * f_1(1, 1, mom1) / nuo
        freq_value_f1_mom2 = B**2 * f_1(1, 1, mom2) / nuo
        freq_value_f1_mom12 = B**2 * f_1(1, 1, mom1 + mom2) / nuo
        freq_value_f2_mom1 = B**2 * f_2(1, 1, mom1) / nuo
        freq_value_f2_mom2 = B**2 * f_2(1, 1, mom2) / nuo
        freq_value_f2_mom12 = B**2 * f_2(1, 1, mom1 + mom2) / nuo

        if mom == mom_arg_1:
            if freq == freq_value_f1_mom1:
                return B**2 * cls(1, mom1, f_1(1, 1, mom1)) / nuo
            elif freq == -freq_value_f1_mom1:
                return B**2 * cls(1, mom1, -f_1(1, 1, mom1)) / nuo
            elif freq == freq_value_f2_mom1:
                return B**2 * cls(1, mom1, f_2(1, 1, mom1)) / nuo
            elif freq == -freq_value_f2_mom1:
                return B**2 * cls(1, mom1, -f_2(1, 1, mom1)) / nuo
        if mom == mom_arg_2:
            if freq == freq_value_f1_mom2:
                return B**2 * cls(1, mom2, f_1(1, 1, mom2)) / nuo
            elif freq == -freq_value_f1_mom2:
                return B**2 * cls(1, mom2, -f_1(1, 1, mom2)) / nuo
            elif freq == freq_value_f2_mom2:
                return B**2 * cls(1, mom2, f_2(1, 1, mom2)) / nuo
            elif freq == -freq_value_f2_mom2:
                return B**2 * cls(1, mom2, -f_2(1, 1, mom2)) / nuo
        if mom == mom_arg_1 + mom_arg_2:
            if freq == freq_value_f1_mom12:
                return B**2 * cls(1, mom1 + mom2, f_1(1, 1, mom1 + mom2)) / nuo
            elif freq == -freq_value_f1_mom12:
                return B**2 * cls(1, mom1 + mom2, -f_1(1, 1, mom1 + mom2)) / nuo
            elif freq == freq_value_f2_mom12:
                return B**2 * cls(1, mom1 + mom2, f_2(1, 1, mom1 + mom2)) / nuo
            elif freq == -freq_value_f2_mom12:
                return B**2 * cls(1, mom1 + mom2, f_2(1, 1, mom1 + mom2)) / nuo

    def doit(self, deep=True, **hints):
        nuo, mom, freq = self.args

        global momentums_for_helicity_propagators

        if deep:
            mom = mom.doit(deep=deep, **hints)
            freq = freq.doit(deep=deep, **hints)
            nuo = nuo.doit(deep=deep, **hints)

            if mom == sum(momentums_for_helicity_propagators):
                mom1 = momentums_for_helicity_propagators[0]
                mom2 = momentums_for_helicity_propagators[1]
                return -I * freq + nuo * (mom1**2 + 2 * mom1 * mom2 * z + mom2**2)
            else:
                return -I * freq + nuo * mom**2


class alpha_star(Function):
    """
    alpha_star(nuo, k, w) = conjugate(alpha(nuo, k, w)) = I*w + nuo*k**2

    ARGUMENTS:

    nuo -- bare kinematic viscosity, k -- momentum, w - frequency

    PROPERTIES:

    alpha_star(nuo, k, w) = alpha_star(nuo, -k, w),

    alpha_star(nuo, k, w) = alpha(nuo, k, -w)
    """

    @classmethod
    def eval(cls, nuo, mom, freq):

        global momentums_for_helicity_propagators
        global B

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(nuo, -mom, freq)
        if freq.could_extract_minus_sign():
            return alpha(nuo, mom, -freq)
        # define scaling properties
        if freq == B**2 * f_1(1, 1, mom1) / nuo:
            if mom == B * mom1 / nuo:
                return B**2 * cls(1, mom1, f_1(1, 1, mom1)) / nuo
        if freq == B**2 * f_1(1, 1, mom2) / nuo:
            if mom == B * mom2 / nuo:
                return B**2 * cls(1, mom2, f_1(1, 1, mom2)) / nuo
        if freq == B**2 * f_1(1, 1, mom1 + mom2) / nuo:
            if mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
                return B**2 * cls(1, mom1 + mom2, f_1(1, 1, mom1 + mom2)) / nuo
        if freq == B**2 * f_2(1, 1, mom1) / nuo:
            if mom == B * mom1 / nuo:
                return B**2 * cls(1, mom1, f_2(1, 1, mom1)) / nuo
        if freq == B**2 * f_2(1, 1, mom2) / nuo:
            if mom == B * mom2 / nuo:
                return B**2 * cls(1, mom2, f_2(1, 1, mom2)) / nuo
        if freq == B**2 * f_2(1, 1, mom1 + mom2) / nuo:
            if mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
                return B**2 * cls(1, mom1 + mom2, f_2(1, 1, mom1 + mom2)) / nuo

    def doit(self, deep=True, **hints):
        nuo, mom, freq = self.args

        global momentums_for_helicity_propagators

        if deep:
            mom = mom.doit(deep=deep, **hints)
            freq = freq.doit(deep=deep, **hints)
            nuo = nuo.doit(deep=deep, **hints)
            if mom == sum(momentums_for_helicity_propagators):
                mom1 = momentums_for_helicity_propagators[0]
                mom2 = momentums_for_helicity_propagators[1]
                return I * freq + nuo * (mom1**2 + 2 * mom1 * mom2 * z + mom2**2)
            else:
                return I * freq + nuo * mom**2


class beta(Function):
    """
    beta(nuo, k, w) = -I*w + uo*nuo*k**2

    ARGUMENTS:

    nuo -- bare kinematic viscosity, k -- momentum, w - frequency

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number,

    PROPERTIES:

    beta(nuo, k, w) = beta(nuo, -k, w)
    """

    @classmethod
    def eval(cls, nuo, mom, freq):

        global momentums_for_helicity_propagators
        global B

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(nuo, -mom, freq)

        # define scaling properties

        mom_arg_1 = B * mom1 / nuo
        mom_arg_2 = B * mom2 / nuo

        freq_value_f1_mom1 = B**2 * f_1(1, 1, mom1) / nuo
        freq_value_f1_mom2 = B**2 * f_1(1, 1, mom2) / nuo
        freq_value_f1_mom12 = B**2 * f_1(1, 1, mom1 + mom2) / nuo
        freq_value_f2_mom1 = B**2 * f_2(1, 1, mom1) / nuo
        freq_value_f2_mom2 = B**2 * f_2(1, 1, mom2) / nuo
        freq_value_f2_mom12 = B**2 * f_2(1, 1, mom1 + mom2) / nuo

        if mom == mom_arg_1:
            if freq == freq_value_f1_mom1:
                return B**2 * cls(1, mom1, f_1(1, 1, mom1)) / nuo
            elif freq == -freq_value_f1_mom1:
                return B**2 * cls(1, mom1, -f_1(1, 1, mom1)) / nuo
            elif freq == freq_value_f2_mom1:
                return B**2 * cls(1, mom1, f_2(1, 1, mom1)) / nuo
            elif freq == -freq_value_f2_mom1:
                return B**2 * cls(1, mom1, -f_2(1, 1, mom1)) / nuo
        if mom == mom_arg_2:
            if freq == freq_value_f1_mom2:
                return B**2 * cls(1, mom2, f_1(1, 1, mom2)) / nuo
            elif freq == -freq_value_f1_mom2:
                return B**2 * cls(1, mom2, -f_1(1, 1, mom2)) / nuo
            elif freq == freq_value_f2_mom2:
                return B**2 * cls(1, mom2, f_2(1, 1, mom2)) / nuo
            elif freq == -freq_value_f2_mom2:
                return B**2 * cls(1, mom2, -f_2(1, 1, mom2)) / nuo
        if mom == mom_arg_1 + mom_arg_2:
            if freq == freq_value_f1_mom12:
                return B**2 * cls(1, mom1 + mom2, f_1(1, 1, mom1 + mom2)) / nuo
            elif freq == -freq_value_f1_mom12:
                return B**2 * cls(1, mom1 + mom2, -f_1(1, 1, mom1 + mom2)) / nuo
            elif freq == freq_value_f2_mom12:
                return B**2 * cls(1, mom1 + mom2, f_2(1, 1, mom1 + mom2)) / nuo
            elif freq == -freq_value_f2_mom12:
                return B**2 * cls(1, mom1 + mom2, f_2(1, 1, mom1 + mom2)) / nuo

    def doit(self, deep=True, **hints):
        nuo, mom, freq = self.args

        global momentums_for_helicity_propagators

        if deep:
            mom = mom.doit(deep=deep, **hints)
            freq = freq.doit(deep=deep, **hints)
            nuo = nuo.doit(deep=deep, **hints)
            if mom == sum(momentums_for_helicity_propagators):
                mom1 = momentums_for_helicity_propagators[0]
                mom2 = momentums_for_helicity_propagators[1]
                return -I * freq + uo * nuo * (mom1**2 + 2 * mom1 * mom2 * z + mom2**2)
            else:
                return -I * freq + uo * nuo * mom**2


class beta_star(Function):
    """
    beta_star(nuo, k, w) = conjugate(beta(nuo, k, w)) = I*w + uo*nuo*k**2

    ARGUMENTS:

    nuo -- bare kinematic viscosity, k -- momentum, w - frequency

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number,

    PROPERTIES:

    beta_star(nuo, k, w) = beta_star(nuo, -k, w),

    beta_star(nuo, k, w) = beta(nuo, k, -w)
    """

    @classmethod
    def eval(cls, nuo, mom, freq):

        global momentums_for_helicity_propagators
        global B

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(nuo, -mom, freq)
        if freq.could_extract_minus_sign():
            return beta(nuo, mom, -freq)

        # define scaling properties
        if freq == B**2 * f_1(1, 1, mom1) / nuo:
            if mom == B * mom1 / nuo:
                return B**2 * cls(1, mom1, f_1(1, 1, mom1)) / nuo
        if freq == B**2 * f_1(1, 1, mom2) / nuo:
            if mom == B * mom2 / nuo:
                return B**2 * cls(1, mom2, f_1(1, 1, mom2)) / nuo
        if freq == B**2 * f_1(1, 1, mom1 + mom2) / nuo:
            if mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
                return B**2 * cls(1, mom1 + mom2, f_1(1, 1, mom1 + mom2)) / nuo
        if freq == B**2 * f_2(1, 1, mom1) / nuo:
            if mom == B * mom1 / nuo:
                return B**2 * cls(1, mom1, f_2(1, 1, mom1)) / nuo
        if freq == B**2 * f_2(1, 1, mom2) / nuo:
            if mom == B * mom2 / nuo:
                return B**2 * cls(1, mom2, f_2(1, 1, mom2)) / nuo
        if freq == B**2 * f_2(1, 1, mom1 + mom2) / nuo:
            if mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
                return B**2 * cls(1, mom1 + mom2, f_2(1, 1, mom1 + mom2)) / nuo

    def doit(self, deep=True, **hints):
        nuo, mom, freq = self.args

        global momentums_for_helicity_propagators

        if deep:
            mom = mom.doit(deep=deep, **hints)
            freq = freq.doit(deep=deep, **hints)
            nuo = nuo.doit(deep=deep, **hints)
            if mom == sum(momentums_for_helicity_propagators):
                mom1 = momentums_for_helicity_propagators[0]
                mom2 = momentums_for_helicity_propagators[1]
                return I * freq + uo * nuo * (mom1**2 + 2 * mom1 * mom2 * z + mom2**2)
            else:
                return I * freq + uo * nuo * mom**2


class sc_prod(Function):
    """
    This auxiliary function denotes the standard dot product of vectors in R**d

    ARGUMENTS:

    B -- external magnetic field, k -- momentum

    PROPERTIES:

    sc_prod(B, k + q) = sc_prod(B, k) + sc_prod(B, q)

    sc_prod(B, k) = -sc_prod(B, -k)

    sc_prod(0, k) = sc_prod(B, 0) = 0
    """

    @classmethod
    def eval(cls, field, mom):

        global momentums_for_helicity_propagators
        global B

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return -cls(field, -mom)
        if field == 0:
            return 0
        if mom == 0:
            return 0

        # define scaling properties
        if field == B and mom == B * mom1 / nuo:
            return B**2 * cls(1, mom1) / nuo
        if field == B and mom == B * mom2 / nuo:
            return B**2 * cls(1, mom2) / nuo
        if field == B and mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(1, mom1 + mom2) / nuo

        # function is linear with respect to the second argument by definition
        if mom.is_Add:
            x = list()
            for i in range(len(mom.args)):
                q = mom.args[i]
                x.append(cls(field, q))
            return sum(x)

    def doit(self, deep=True, **hints):
        field, mom = self.args

        global momentums_for_helicity_propagators

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        if deep:
            mom = mom.doit(deep=deep, **hints)
            field = field.doit(deep=deep, **hints)
            if mom == mom1:
                return field * mom * z_k
            elif mom == mom2:
                return field * mom * z_q
            else:
                return NameError


class D(Function):
    """
    This auxiliary function is introduced for the convenient calculation of
    integrals over frequencies (using the residue theorem). It defines the minus discriminant
    that occurs when solving the quadratic equation xi(k, w) = 0 with respect to w (see below)

    true_discriminant = 4*A*sc_prod(B, k)**2 - k**4*nuo**2*(uo - 1)**2

    We introduce: D(B, nuo, k) = k**4*nuo**2*(uo - 1)**2 - 4*A*sc_prod(B, k)**2   == >
    == >    sqrt(true_discriminant) = I*sqrt(D(B, nuo, k))

    ARGUMENTS:
    k -- momentum

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number, nuo -- bare kinematic viscosity
    sc_prod(B, k) -- dot product of external magnetic field B and momentum k,

    PROPERTIES:

    D(B, nuo, k) = D(B, nuo, -k)
    """

    @classmethod
    def eval(cls, field, visc, mom):

        global momentums_for_helicity_propagators
        global B, nuo

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        mom = expand(mom)
        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(field, visc, -mom)
        # define scaling properties
        if mom == B * mom1 / nuo:
            return B**4 * cls(1, 1, mom1) / nuo**2
        elif mom == B * mom2 / nuo:
            return B**4 * cls(1, 1, mom2) / nuo**2
        elif mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
            return B**4 * cls(1, 1, mom1 + mom2) / nuo**2

    is_integer = True
    is_negative = False

    def doit(self, deep=True, **hints):

        field, visc, mom = self.args

        global momentums_for_helicity_propagators

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        if deep:
            mom = mom.doit(deep=deep, **hints)
            field = field.doit(deep=deep, **hints)
            visc = visc.doit(deep=deep, **hints)

            if mom == mom1 + mom2:
                return (
                    -4 * A * sc_prod(field, mom) ** 2
                    + (mom1**2 + 2 * mom1 * mom2 * z + mom2**2) ** 2 * visc**2 * (uo - 1) ** 2
                )
            else:
                return -4 * A * sc_prod(field, mom) ** 2 + mom**4 * visc**2 * (uo - 1) ** 2


class f_1(Function):
    """
    This auxiliary function is introduced for the convenience of the calculation of
    integrals over frequencies (using the residue theorem). It returns the first root of the equation
    xi(k, w) = 0 with respect to w (see below).

    f_1(B, nuo, k) = -I*(sqrt(D(B, nuo, k)) + k**2*nuo*(uo + 1))/2

    ARGUMENTS:
    B -- external field, nuo -- bare kinematic viscosity, k -- momentum

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number,
    sc_prod(B, k) -- dot product of external magnetic field B and momentum k

    PROPERTIES:

    f_1(B, nuo, k) = f_1(B, nuo, -k)
    """

    @classmethod
    def eval(cls, field, visc, mom):

        global momentums_for_helicity_propagators
        global B, nuo

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        mom = expand(mom)

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(field, visc, -mom)
        # define scaling properties
        if mom == B * mom1 / nuo:
            return B**2 * cls(1, 1, mom1) / nuo
        elif mom == B * mom2 / nuo:
            return B**2 * cls(1, 1, mom2) / nuo
        elif mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(1, 1, mom1 + mom2) / nuo

    def doit(self, deep=True, **hints):

        field, visc, mom = self.args

        global momentums_for_helicity_propagators
        global B

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        if deep:
            mom = mom.doit(deep=deep, **hints)
            visc = visc.doit(deep=deep, **hints)
            field = field.doit(deep=deep, **hints)

            if mom == mom1 + mom2:
                return (
                    -I
                    * (sqrt(D(field, visc, mom)) + (mom1**2 + 2 * mom1 * mom2 * z + mom2**2) * visc * (uo + 1))
                    / 2
                )
            else:
                return -I * (sqrt(D(field, visc, mom)) + mom**2 * visc * (uo + 1)) / 2


class f_2(Function):
    """
    This auxiliary function is introduced for the convenience of the calculation of
    integrals over frequencies (using the residue theorem). It returns the second root of the equation
    xi(k, w) = 0 with respect to w (see below).

    f_2(B, nuo, k) = -I*( - sqrt(D(B, nuo, k)) + k**2*nuo*(uo + 1))/2

    Note:
    f_2(B, nuo, k) differs from f_1(B, nuo, k) by sign before the square root

    ARGUMENTS:
    B -- external field, nuo -- bare kinematic viscosity, k -- momentum

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number,
    sc_prod(B, k) -- dot product of external magnetic field B and momentum k

    PROPERTIES:

    f_2(B, nuo, k) = f_2(B, nuo, -k),

    conjugate(f_2(B, nuo, k)) = - f_1(B, nuo, k)  ==>  conjugate(f_1(B, nuo, k)) = - f_2(B, nuo, k)
    """

    @classmethod
    def eval(cls, field, visc, mom):

        global momentums_for_helicity_propagators
        global B, nuo

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        mom = expand(mom)

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(field, visc, -mom)
        # define scaling properties
        if mom == B * mom1 / nuo:
            return B**2 * cls(1, 1, mom1) / nuo
        elif mom == B * mom2 / nuo:
            return B**2 * cls(1, 1, mom2) / nuo
        elif mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(1, 1, mom1 + mom2) / nuo

    def doit(self, deep=True, **hints):
        field, visc, mom = self.args

        global momentums_for_helicity_propagators

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        if deep:
            mom = mom.doit(deep=deep, **hints)
            visc = visc.doit(deep=deep, **hints)
            field = field.doit(deep=deep, **hints)

            if mom == mom1 + mom2:
                return (
                    -I
                    * (-sqrt(D(field, visc, mom)) + (mom1**2 + 2 * mom1 * mom2 * z + mom2**2) * visc * (uo + 1))
                    / 2
                )
            else:
                return -I * (-sqrt(D(field, visc, mom)) + mom**2 * visc * (uo + 1)) / 2


class chi_1(Function):
    """
    This auxiliary function is introduced for the convenience of the subsequent calculation of
    integrals over frequencies (using the residue theorem). It defines the first monomial in the
    decomposition of the square polynomial xi(k, w) (with respect to w) into irreducible factors (see below).

    chi_1(k, w) = (w - f_1(B, nuo, k))

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
        return w - f_1(B, nuo, k)


class chi_2(Function):
    """
    This auxiliary function is introduced for the convenience of the subsequent calculation of
    integrals over frequencies (using the residue theorem). It defines the second monomial in the
    decomposition of the square polynomial xi(k, w) (with respect to w) into irreducible factors (see below).

    chi_2(k, w) = (w - f_2(B, nuo, k))

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    chi_2(k, w) = chi_2(-k, w),

    conjugate(chi_2(k, w)) = w - conjugate(f_2(B, nuo, k)) = w + f_1(B, nuo, k)
    = -(- w - f_1(B, nuo, k)) = -chi_1(k, -w),

    chi_2(k, -w) = (- w - f_2(B, nuo, k)) = -(w + f_2(B, nuo, k))
    = -w + conjugate(f_1(B, nuo, k)) = -conjugate(chi_1(k, w))
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
        return w - f_2(B, nuo, k)


class xi(Function):
    """
    The function xi(k, w) is defined by the equality

    xi(k, w) = A*sc_prod(B, k)**2 + alpha(k, w)*beta(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PARAMETERS:

    A - digit parameter of the model

    Representations for xi(k, w) (decomposition into irreducible ones):

    xi(k, w) = -chi_1(k, w)*chi_2(k, w) = -(w - f_1(B, nuo, k))*(w - f_2(B, nuo, k))
    (see definitions above)

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
        return -chi_1(k, w) * chi_2(k, w)


class xi_star(Function):
    """
    The function xi_star(k, w) is defined by the equality

    xi_star(k, w) = conjugate(xi(k, w)) = A*sc_prod(B, k)**2 + alpha_star(k, w)*beta_star(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PARAMETERS:

    A - digit parameter of the model

    Representations for xi_star(k, w) (decomposition into irreducible ones):

    xi_star(k, w) = -conjugate(chi_2(k, w))*conjugate(chi_1(k, w)) = -chi_2(k, -w)*chi_1(k, -w)

    PROPERTIES:

    xi(k, w) = xi(-k, w),

    xi_star(k, w) =  -(w + f_1(B, nuo, k))*(w + f_2(B, nuo, k))
    = -(- w - f_1(B, nuo, k))*(- w - f_2(B, nuo, k)) = xi(k, -w),

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
        return -chi_2(k, -w) * chi_1(k, -w)


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

    @classmethod
    def eval(cls, k, index_k):
        if k.is_Add:
            x = list()
            for i in range(len(k.args)):
                q = k.args[i]
                x.append(cls(q, index_k))
            return sum(x)
        if k.could_extract_minus_sign():
            return -cls(-k, index_k)


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

    @classmethod
    def eval(cls, k, index_1, index_2):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, index_1, index_2)


class H(Function):
    """
    Helical term
    """
