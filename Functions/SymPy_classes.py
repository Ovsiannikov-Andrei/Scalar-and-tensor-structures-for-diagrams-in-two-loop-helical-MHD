from sympy import *
from Functions.Global_variables import *

# -----------------------------------------------------------------------------------------------------------------#
#                                       Custom functions in subclass Function in SymPy
# -----------------------------------------------------------------------------------------------------------------#

# All functions here are given in momentum-frequency representation

# ATTENTION!!! The sign of the Fourier transform was chosen to be the same as in [2] (see main file)


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

    D_v(k*B/nuo) = (B/nuo)**(4 - d - 2*eps)*D_v(k)
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


class k_plus_q_square(Function):
    """
    Just the squared norm of the sum of two vectors.

    k_plus_q_square(k, q) = k**2 + 2*k*q*z + q**2

    ARGUMENTS:

    k, q -- absolute value of momentums

    PARAMETERS:

    z -- cosine of the angle between vectors k and q

    PROPERTIES:

    k_plus_q_square(k, q) -- non-negative function
    """

    # # the squared norm of the sum of two vectors is always non-negative
    # is_nonnegative = True

    def _eval_is_nonnegative(self):
        # the squared norm of the sum of two vectors is always non-negative
        mom1, mom2 = self.args
        if mom1.is_real is True and mom2.is_real is True:
            return True

    def doit(self, deep=True, **hints):
        mom1, mom2 = self.args

        global z

        if deep:
            mom1 = mom1.doit(deep=deep, **hints)
            mom2 = mom2.doit(deep=deep, **hints)
        return mom1**2 + 2 * mom1 * mom2 * z + mom2**2


class k_plus_q_square_square(Function):
    """
    Just the norm of the sum of vectors to the fourth power.

    k_plus_q_square_square(k, q) = k_plus_q_square(k, q)**2

    ARGUMENTS:

    k, q -- absolute value of momentums

    PROPERTIES:

    k_plus_q_square_square(k, q) -- non-negative function
    """

    def _eval_is_nonnegative(self):
        # the the norm of the sum of vectors to the fourth power is always non-negative
        mom1, mom2 = self.args
        if mom1.is_real is True and mom2.is_real is True:
            return True

    def doit(self, deep=True, **hints):
        mom1, mom2 = self.args

        if deep:
            mom1 = mom1.doit(deep=deep, **hints)
            mom2 = mom2.doit(deep=deep, **hints)
        return k_plus_q_square(mom1, mom2) ** 2


class k_minus_q_square(Function):
    """
    Just the squared norm of the difference between two vectors.

    k_minus_q_square(k, q) = k**2 - 2*k*q*z + q**2

    ARGUMENTS:

    k, q -- absolute value of momentums

    PARAMETERS:

    z -- cosine of the angle between vectors k and q

    PROPERTIES:

    k_minus_q_square(k, q) -- non-negative function
    """

    # the squared norm of the difference between two vectors is always non-negative
    is_nonnegative = True

    def doit(self, deep=True, **hints):
        mom1, mom2 = self.args

        global z

        if deep:
            mom1 = mom1.doit(deep=deep, **hints)
            mom2 = mom2.doit(deep=deep, **hints)
        return mom1**2 - 2 * mom1 * mom2 * z + mom2**2


class k_minus_q_square_square(Function):
    """
    Just the norm of the sum of vectors to the fourth power.

    k_minus_q_square_square(k, q) = k_minus_q_square(k, q)**2

    ARGUMENTS:

    k, q -- absolute value of momentums

    PROPERTIES:

    k_minus_q_square_square(k, q) -- non-negative function
    """

    def _eval_is_nonnegative(self):
        # the the norm of the difference between two vectors to the fourth power is always non-negative
        mom1, mom2 = self.args
        if mom1.is_real is True and mom2.is_real is True:
            return True

    def doit(self, deep=True, **hints):
        mom1, mom2 = self.args

        if deep:
            mom1 = mom1.doit(deep=deep, **hints)
            mom2 = mom2.doit(deep=deep, **hints)
        return k_minus_q_square(mom1, mom2) ** 2


class alpha(Function):
    """
    alpha(nuo, k, w) = I*w + nuo*k**2

    alpha(nuo, k + q, w) = I*w + nuo*(k**2 + 2*k*q*z + q**2)

    ARGUMENTS:

    nuo -- bare kinematic viscosity,  k -- momentum, w - frequency

    PROPERTIES:

    alpha(nuo, k, w) = alpha(nuo, -k, w)

    alpha(nuo, k*B/nuo, w*B**2/nuo) = B**2*alpha(1, k, w)/nuo
    """

    @classmethod
    def eval(cls, nuo, mom, freq):
        global momentums_for_helicity_propagators
        global B

        mom = expand(mom)

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(nuo, -mom, freq)

        # define scaling properties
        # if k = k_1 + k_2, then the replacement must be done consistently
        # (when k_1 --> k_1*B/nuo, k_2 --> k_2*B/nuo, respectively), i.e.
        # alpha(nuo, k*B/nuo + q*B/nuo, w*B**2/nuo) = B**2*alpha(1, k + q, w)/nuo

        if freq.has(B**2 / nuo) and mom.has(B / nuo):
            if mom.is_Add and freq.is_Mul:
                x = list()
                for i in range(len(mom.args)):
                    q = mom.args[i]
                    if q.has(B / nuo):
                        x.append(q)
                if (sum(x) - mom) == 0:
                    arg2 = sum(x).subs(nuo, 1).subs(B, 1)
                    arg3 = freq.subs(nuo, 1).subs(B, 1)

                    return B**2 * cls(1, arg2, arg3) / nuo
            elif mom.is_Mul and freq.is_Add:
                x = list()
                for i in range(len(freq.args)):
                    q = freq.args[i]
                    if q.has(B**2 / nuo):
                        x.append(q)
                if (sum(x) - freq) == 0:
                    arg2 = mom.subs(nuo, 1).subs(B, 1)
                    arg3 = sum(x).subs(nuo, 1).subs(B, 1)

                    return B**2 * cls(1, arg2, arg3) / nuo
            elif mom.is_Add and freq.is_Add:
                x_1 = list()
                x_2 = list()
                for i in range(len(mom.args)):
                    q = mom.args[i]
                    if q.has(B / nuo):
                        x_1.append(q)
                for i in range(len(freq.args)):
                    q = freq.args[i]
                    if q.has(B**2 / nuo):
                        x_2.append(q)
                if (len(x_2) - len(freq.args)) == 0 and (len(x_1) - len(mom.args)) == 0:
                    arg2 = sum(x_1).subs(nuo, 1).subs(B, 1)
                    arg3 = sum(x_2).subs(nuo, 1).subs(B, 1)
                    return B**2 * cls(1, arg2, arg3) / nuo

            elif mom.is_Mul and freq.is_Mul:
                arg2 = mom.subs(nuo, 1).subs(B, 1)
                arg3 = freq.subs(nuo, 1).subs(B, 1)
                return B**2 * cls(1, arg2, arg3) / nuo

    def doit(self, deep=True, **hints):
        nuo, mom, freq = self.args

        global momentums_for_helicity_propagators

        if deep:
            mom = mom.doit(deep=deep, **hints)
            freq = freq.doit(deep=deep, **hints)
            nuo = nuo.doit(deep=deep, **hints)

            mom1 = momentums_for_helicity_propagators[0]
            mom2 = momentums_for_helicity_propagators[1]

            if mom == mom1 + mom2:
                return I * freq + nuo * k_plus_q_square(mom1, mom2).doit()
            elif mom == mom1 - mom2 or mom == -mom1 + mom2:
                return I * freq + nuo * k_minus_q_square(mom1, mom2).doit()
            else:
                return I * freq + nuo * mom**2

    # define differentiation
    def fdiff(self, argindex):
        # argindex indexes the args, starting at 1
        nuo, mom, freq = self.args
        if argindex == 3:
            return I


class alpha_star(Function):
    """
    alpha_star(nuo, k, w) = conjugate(alpha(nuo, k, w)) = -I*w + nuo*k**2

    alpha_star(nuo, k + q, w) = -I*w + nuo*(k**2 + 2*k*q*z + q**2)

    ARGUMENTS:

    nuo -- bare kinematic viscosity, k -- momentum, w - frequency

    PROPERTIES:

    alpha_star(nuo, k, w) = alpha_star(nuo, -k, w),

    alpha_star(nuo, k, w) = alpha(nuo, k, -w)

    alpha_star(nuo, k*B/nuo, w*B**2/nuo) = B**2*alpha_star(1, k, w)/nuo
    """

    @classmethod
    def eval(cls, nuo, mom, freq):
        global momentums_for_helicity_propagators
        global B

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(nuo, -mom, freq)
        if freq.could_extract_minus_sign():
            return alpha(nuo, mom, -freq)

        # define scaling properties

        # if k = k_1 + k_2, then the replacement must be done consistently
        # (when k_1 --> k_1*B/nuo, k_2 --> k_2*B/nuo, respectively), i.e.
        # alpha_star(nuo, k*B/nuo + q*B/nuo, w*B**2/nuo) = B**2*alpha_star(1, k + q, w)/nuo

        if freq.has(B**2 / nuo) and mom.has(B / nuo):
            if mom.is_Add and freq.is_Mul:
                x = list()
                for i in range(len(mom.args)):
                    q = mom.args[i]
                    if q.has(B / nuo):
                        x.append(q)
                if (sum(x) - mom) == 0:
                    arg2 = sum(x).subs(nuo, 1).subs(B, 1)
                    arg3 = freq.subs(nuo, 1).subs(B, 1)

                    return B**2 * cls(1, arg2, arg3) / nuo
            elif mom.is_Mul and freq.is_Add:
                x = list()
                for i in range(len(freq.args)):
                    q = freq.args[i]
                    if q.has(B**2 / nuo):
                        x.append(q)
                if (sum(x) - freq) == 0:
                    arg2 = mom.subs(nuo, 1).subs(B, 1)
                    arg3 = sum(x).subs(nuo, 1).subs(B, 1)

                    return B**2 * cls(1, arg2, arg3) / nuo
            elif mom.is_Add and freq.is_Add:
                x_1 = list()
                x_2 = list()
                for i in range(len(mom.args)):
                    q = mom.args[i]
                    if q.has(B / nuo):
                        x_1.append(q)
                for i in range(len(freq.args)):
                    q = freq.args[i]
                    if q.has(B**2 / nuo):
                        x_2.append(q)
                if (len(x_2) - len(freq.args)) == 0 and (len(x_1) - len(mom.args)) == 0:
                    arg2 = sum(x_1).subs(nuo, 1).subs(B, 1)
                    arg3 = sum(x_2).subs(nuo, 1).subs(B, 1)
                    return B**2 * cls(1, arg2, arg3) / nuo

            elif mom.is_Mul and freq.is_Mul:
                arg2 = mom.subs(nuo, 1).subs(B, 1)
                arg3 = freq.subs(nuo, 1).subs(B, 1)
                return B**2 * cls(1, arg2, arg3) / nuo

    def doit(self, deep=True, **hints):
        nuo, mom, freq = self.args

        global momentums_for_helicity_propagators

        if deep:
            mom = mom.doit(deep=deep, **hints)
            freq = freq.doit(deep=deep, **hints)
            nuo = nuo.doit(deep=deep, **hints)

            mom1 = momentums_for_helicity_propagators[0]
            mom2 = momentums_for_helicity_propagators[1]

            if mom == mom1 + mom2:
                return -I * freq + nuo * k_plus_q_square(mom1, mom2).doit()
            elif mom == mom1 - mom2 or mom == -mom1 + mom2:
                return -I * freq + nuo * k_minus_q_square(mom1, mom2).doit()
            else:
                return -I * freq + nuo * mom**2

    # Define differentiation
    def fdiff(self, argindex):
        # argindex indexes the args, starting at 1
        nuo, mom, freq = self.args
        if argindex == 3:
            return -I


class beta(Function):
    """
    beta(nuo, k, w) = I*w + uo*nuo*k**2

    beta(nuo, k + q, w) = I*w + uo*nuo*(k**2 + 2*k*q*z + q**2)

    ARGUMENTS:

    nuo -- bare kinematic viscosity, k -- momentum, w - frequency

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number,

    PROPERTIES:

    beta(nuo, k, w) = beta(nuo, -k, w)

    beta(nuo, k, w) = beta(nuo, -k, w)

    beta(nuo, k*B/nuo, w*B**2/nuo) = B**2*beta(1, k, w)/nuo
    """

    @classmethod
    def eval(cls, nuo, mom, freq):
        global momentums_for_helicity_propagators
        global B

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(nuo, -mom, freq)

        # define scaling properties

        # if k = k_1 + k_2, then the replacement must be done consistently
        # (when k_1 --> k_1*B/nuo, k_2 --> k_2*B/nuo, respectively), i.e.
        # beta(nuo, k*B/nuo + q*B/nuo, w*B**2/nuo) = B**2*beta(1, k + q, w)/nuo

        if freq.has(B**2 / nuo) and mom.has(B / nuo):
            if mom.is_Add and freq.is_Mul:
                x = list()
                for i in range(len(mom.args)):
                    q = mom.args[i]
                    if q.has(B / nuo):
                        x.append(q)
                if (sum(x) - mom) == 0:
                    arg2 = sum(x).subs(nuo, 1).subs(B, 1)
                    arg3 = freq.subs(nuo, 1).subs(B, 1)

                    return B**2 * cls(1, arg2, arg3) / nuo
            elif mom.is_Mul and freq.is_Add:
                x = list()
                for i in range(len(freq.args)):
                    q = freq.args[i]
                    if q.has(B**2 / nuo):
                        x.append(q)
                if (sum(x) - freq) == 0:
                    arg2 = mom.subs(nuo, 1).subs(B, 1)
                    arg3 = sum(x).subs(nuo, 1).subs(B, 1)

                    return B**2 * cls(1, arg2, arg3) / nuo
            elif mom.is_Add and freq.is_Add:
                x_1 = list()
                x_2 = list()
                for i in range(len(mom.args)):
                    q = mom.args[i]
                    if q.has(B / nuo):
                        x_1.append(q)
                for i in range(len(freq.args)):
                    q = freq.args[i]
                    if q.has(B**2 / nuo):
                        x_2.append(q)
                if (len(x_2) - len(freq.args)) == 0 and (len(x_1) - len(mom.args)) == 0:
                    arg2 = sum(x_1).subs(nuo, 1).subs(B, 1)
                    arg3 = sum(x_2).subs(nuo, 1).subs(B, 1)
                    return B**2 * cls(1, arg2, arg3) / nuo

            elif mom.is_Mul and freq.is_Mul:
                arg2 = mom.subs(nuo, 1).subs(B, 1)
                arg3 = freq.subs(nuo, 1).subs(B, 1)
                return B**2 * cls(1, arg2, arg3) / nuo

    def doit(self, deep=True, **hints):
        nuo, mom, freq = self.args

        global momentums_for_helicity_propagators

        if deep:
            mom = mom.doit(deep=deep, **hints)
            freq = freq.doit(deep=deep, **hints)
            nuo = nuo.doit(deep=deep, **hints)

            mom1 = momentums_for_helicity_propagators[0]
            mom2 = momentums_for_helicity_propagators[1]

            if mom == mom1 + mom2:
                return I * freq + uo * nuo * k_plus_q_square(mom1, mom2).doit()
            elif mom == mom1 - mom2 or mom == -mom1 + mom2:
                return I * freq + uo * nuo * k_minus_q_square(mom1, mom2).doit()
            else:
                return I * freq + uo * nuo * mom**2

    # Define differentiation
    def fdiff(self, argindex):
        # argindex indexes the args, starting at 1
        nuo, mom, freq = self.args
        if argindex == 3:
            return I


class beta_star(Function):
    """
    beta_star(nuo, k, w) = conjugate(beta(nuo, k, w)) = -I*w + uo*nuo*k**2

    beta_star(nuo, k, w) = -I*w + uo*nuo*(k**2 + 2*k*q*z + q**2)

    ARGUMENTS:

    nuo -- bare kinematic viscosity, k -- momentum, w - frequency

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number,

    PROPERTIES:

    beta_star(nuo, k, w) = beta_star(nuo, -k, w),

    beta_star(nuo, k, w) = beta(nuo, k, -w)

    beta_star(nuo, k*B/nuo, w*B**2/nuo) = B**2*beta_star(nuo, k, w)/nuo
    """

    @classmethod
    def eval(cls, nuo, mom, freq):
        global momentums_for_helicity_propagators
        global B

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(nuo, -mom, freq)
        if freq.could_extract_minus_sign():
            return beta(nuo, mom, -freq)

        # define scaling properties

        # if k = k_1 + k_2, then the replacement must be done consistently
        # (when k_1 --> k_1*B/nuo, k_2 --> k_2*B/nuo, respectively), i.e.
        # beta_star(nuo, k*B/nuo + q*B/nuo, w*B**2/nuo) = B**2*beta_star(1, k + q, w)/nuo

        if freq.has(B**2 / nuo) and mom.has(B / nuo):
            if mom.is_Add and freq.is_Mul:
                x = list()
                for i in range(len(mom.args)):
                    q = mom.args[i]
                    if q.has(B / nuo):
                        x.append(q)
                if (sum(x) - mom) == 0:
                    arg2 = sum(x).subs(nuo, 1).subs(B, 1)
                    arg3 = freq.subs(nuo, 1).subs(B, 1)

                    return B**2 * cls(1, arg2, arg3) / nuo
            elif mom.is_Mul and freq.is_Add:
                x = list()
                for i in range(len(freq.args)):
                    q = freq.args[i]
                    if q.has(B**2 / nuo):
                        x.append(q)
                if (sum(x) - freq) == 0:
                    arg2 = mom.subs(nuo, 1).subs(B, 1)
                    arg3 = sum(x).subs(nuo, 1).subs(B, 1)

                    return B**2 * cls(1, arg2, arg3) / nuo
            elif mom.is_Add and freq.is_Add:
                x_1 = list()
                x_2 = list()
                for i in range(len(mom.args)):
                    q = mom.args[i]
                    if q.has(B / nuo):
                        x_1.append(q)
                for i in range(len(freq.args)):
                    q = freq.args[i]
                    if q.has(B**2 / nuo):
                        x_2.append(q)
                if (len(x_2) - len(freq.args)) == 0 and (len(x_1) - len(mom.args)) == 0:
                    arg2 = sum(x_1).subs(nuo, 1).subs(B, 1)
                    arg3 = sum(x_2).subs(nuo, 1).subs(B, 1)
                    return B**2 * cls(1, arg2, arg3) / nuo

            elif mom.is_Mul and freq.is_Mul:
                arg2 = mom.subs(nuo, 1).subs(B, 1)
                arg3 = freq.subs(nuo, 1).subs(B, 1)
                return B**2 * cls(1, arg2, arg3) / nuo

    def doit(self, deep=True, **hints):
        nuo, mom, freq = self.args

        global momentums_for_helicity_propagators

        if deep:
            mom = mom.doit(deep=deep, **hints)
            freq = freq.doit(deep=deep, **hints)
            nuo = nuo.doit(deep=deep, **hints)

            mom1 = momentums_for_helicity_propagators[0]
            mom2 = momentums_for_helicity_propagators[1]

            if mom == mom1 + mom2:
                return -I * freq + uo * nuo * k_plus_q_square(mom1, mom2).doit()
            elif mom == mom1 - mom2 or mom == -mom1 + mom2:
                return -I * freq + uo * nuo * k_minus_q_square(mom1, mom2).doit()
            else:
                return -I * freq + uo * nuo * mom**2

    # Define differentiation
    def fdiff(self, argindex):
        # argindex indexes the args, starting at 1
        nuo, mom, freq = self.args
        if argindex == 3:
            return -I


class sc_prod(Function):
    """
    This auxiliary function denotes the standard dot product of vectors in R**d.

    sc_prod(B, k) = B*b*k*z_k

    ARGUMENTS:

    B -- external magnetic field, k -- momentum, z_k = cos(angle between B and k),
    b is an auxiliary parameter equal to 0 or 1

    PROPERTIES:

    sc_prod(B, k + q) = sc_prod(B, k) + sc_prod(B, q)

    sc_prod(B, k) = -sc_prod(B, -k)

    sc_prod(0, k) = sc_prod(B, 0) = 0

    sc_prod(B, k*B/nuo) = B**2*sc_prod(b, k)/nuo
    """

    @classmethod
    def eval(cls, field, mom):
        global momentums_for_helicity_propagators
        global B, b

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
            return B**2 * cls(b, mom1) / nuo
        if field == B and mom == B * mom2 / nuo:
            return B**2 * cls(b, mom2) / nuo
        if field == B and mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(b, mom1 + mom2) / nuo

        # function is linear with respect to the second argument by definition
        if mom.is_Add:
            x = list()
            for i in range(len(mom.args)):
                q = mom.args[i]
                x.append(cls(field, q))
            return sum(x)

    def doit(self, deep=True, **hints):
        field, mom = self.args

        global momentums_for_helicity_propagators, b

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        if deep:
            mom = mom.doit(deep=deep, **hints)
            field = field.doit(deep=deep, **hints)
            if mom == mom1:
                return b * field * mom * z_k
            elif mom == mom2:
                return b * field * mom * z_q
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

    D(B, nuo, k + q) = (k**2 + 2*k*q*z + q**2)**2*nuo**2*(uo - 1)**2 - 4*A*sc_prod(B, k + q)**2

    ARGUMENTS:
    k -- momentum

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number, nuo -- bare kinematic viscosity
    sc_prod(B, k) -- dot product of external magnetic field B and momentum k,
    b is an auxiliary parameter equal to 0 or 1

    PROPERTIES:

    D(B, nuo, k) = D(B, nuo, -k)

    D(B, nuo, k*B/nuo) = B**4*D(b, 1, k)/nuo**2
    """

    @classmethod
    def eval(cls, field, visc, mom):
        global momentums_for_helicity_propagators
        global B, nuo, b

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        mom = expand(mom)
        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(field, visc, -mom)

        # define scaling properties

        if mom == B * mom1 / nuo:
            return B**2 * cls(b, 1, mom1) / nuo
        elif mom == B * mom2 / nuo:
            return B**2 * cls(b, 1, mom2) / nuo
        elif mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(b, 1, mom1 + mom2) / nuo
        elif mom == B * (mom1 - mom2) / nuo or mom == B * mom1 / nuo - B * mom2 / nuo:
            return B**2 * cls(b, 1, mom1 - mom2) / nuo
        elif mom == B * (-mom1 + mom2) / nuo or mom == -B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(b, 1, -mom1 + mom2) / nuo

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
                    -4 * A * sc_prod(field, mom) ** 2 + k_plus_q_square_square(mom1, mom2) * visc**2 * (uo - 1) ** 2
                )
            elif mom == mom1 - mom2 or mom == -mom1 + mom2:
                return (
                    -4 * A * sc_prod(field, mom) ** 2 + k_minus_q_square_square(mom1, mom2) * visc**2 * (uo - 1) ** 2
                )
            else:
                return -4 * A * sc_prod(field, mom) ** 2 + mom**4 * visc**2 * (uo - 1) ** 2


class f_1(Function):
    """
    This auxiliary function is introduced for the convenience of the calculation of
    integrals over frequencies (using the residue theorem). It returns the first root of the equation
    xi(k, w) = 0 with respect to w (see below).

    f_1(B, nuo, k) = I*(sqrt(D(B, nuo, k)) + k**2*nuo*(uo + 1))/2

    f_1(B, nuo, k + q) = I*(sqrt(D(B, nuo, k + q)) + (k**2 + 2*k*q*z + q**2)*nuo*(uo + 1))/2

    ARGUMENTS:
    B -- external field, nuo -- bare kinematic viscosity, k -- momentum

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number,
    sc_prod(B, k) -- dot product of external magnetic field B and momentum k,
    b is an auxiliary parameter equal to 0 or 1

    PROPERTIES:

    f_1(B, nuo, k) = f_1(B, nuo, -k)

    f_1(B, nuo, k*B/nuo) = B**2*f_1(b, 1, k)/nuo
    """

    @classmethod
    def eval(cls, field, visc, mom):
        global momentums_for_helicity_propagators
        global B, nuo, b

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        mom = expand(mom)

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(field, visc, -mom)

        # define scaling properties

        if mom == B * mom1 / nuo:
            return B**2 * cls(b, 1, mom1) / nuo
        elif mom == B * mom2 / nuo:
            return B**2 * cls(b, 1, mom2) / nuo
        elif mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(b, 1, mom1 + mom2) / nuo
        elif mom == B * (mom1 - mom2) / nuo or mom == B * mom1 / nuo - B * mom2 / nuo:
            return B**2 * cls(b, 1, mom1 - mom2) / nuo
        elif mom == B * (-mom1 + mom2) / nuo or mom == -B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(b, 1, -mom1 + mom2) / nuo

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
                if field == 0:
                    return I * visc * k_plus_q_square(mom1, mom2).doit() * (uo + 1 + Abs(uo - 1)) / 2
                else:
                    return (
                        I
                        * (
                            powdenest(sqrt(D(field, visc, mom)), force=True)
                            + k_plus_q_square(mom1, mom2).doit() * visc * (uo + 1)
                        )
                        / 2
                    )
            elif mom == mom1 - mom2 or mom == -mom1 + mom2:
                if field == 0:
                    return I * visc * k_minus_q_square(mom1, mom2).doit() * (uo + 1 + Abs(uo - 1)) / 2
                else:
                    return (
                        I
                        * (
                            powdenest(sqrt(D(field, visc, mom)), force=True)
                            + k_minus_q_square(mom1, mom2).doit() * visc * (uo + 1)
                        )
                        / 2
                    )
            else:
                if field == 0:
                    return I * visc * mom**2 * (uo + 1 + Abs(uo - 1)) / 2
                else:
                    return I * (powdenest(sqrt(D(field, visc, mom)), force=True) + mom**2 * visc * (uo + 1)) / 2


class f_2(Function):
    """
    This auxiliary function is introduced for the convenience of the calculation of
    integrals over frequencies (using the residue theorem). It returns the second root of the equation
    xi(k, w) = 0 with respect to w (see below).

    f_2(B, nuo, k) = I*( - sqrt(D(B, nuo, k)) + k**2*nuo*(uo + 1))/2

    f_2(B, nuo, k + q) = I*( - sqrt(D(B, nuo, k + q)) + (k**2) + 2*k*q*z + q**2)*nuo*(uo + 1))/2

    Note:
    f_2(B, nuo, k) differs from f_1(B, nuo, k) by sign before the square root

    ARGUMENTS:
    B -- external field, nuo -- bare kinematic viscosity, k -- momentum

    PARAMETERS:

    uo -- bare reciprocal magnetic Prandtl number,
    sc_prod(B, k) -- dot product of external magnetic field B and momentum k,
    b is an auxiliary parameter equal to 0 or 1

    PROPERTIES:

    f_2(B, nuo, k) = f_2(B, nuo, -k),

    conjugate(f_2(B, nuo, k)) = - f_1(B, nuo, k)  ==>  conjugate(f_1(B, nuo, k)) = - f_2(B, nuo, k),

    f_2(B, nuo, k*B/nuo) = B**2*f_2(b, 1, k)/nuo
    """

    @classmethod
    def eval(cls, field, visc, mom):
        global momentums_for_helicity_propagators
        global B, nuo, b

        mom1 = momentums_for_helicity_propagators[0]
        mom2 = momentums_for_helicity_propagators[1]

        mom = expand(mom)

        # function is even with respect to k by definition
        if mom.could_extract_minus_sign():
            return cls(field, visc, -mom)

        # define scaling properties

        if mom == B * mom1 / nuo:
            return B**2 * cls(b, 1, mom1) / nuo
        elif mom == B * mom2 / nuo:
            return B**2 * cls(b, 1, mom2) / nuo
        elif mom == B * (mom1 + mom2) / nuo or mom == B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(b, 1, mom1 + mom2) / nuo
        elif mom == B * (mom1 - mom2) / nuo or mom == B * mom1 / nuo - B * mom2 / nuo:
            return B**2 * cls(b, 1, mom1 - mom2) / nuo
        elif mom == B * (-mom1 + mom2) / nuo or mom == -B * mom1 / nuo + B * mom2 / nuo:
            return B**2 * cls(b, 1, -mom1 + mom2) / nuo

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
                if field == 0:
                    return I * visc * k_plus_q_square(mom1, mom2).doit() * (uo + 1 - Abs(uo - 1)) / 2
                else:
                    return (
                        I
                        * (
                            -powdenest(sqrt(D(field, visc, mom)), force=True)
                            + k_plus_q_square(mom1, mom2).doit() * visc * (uo + 1)
                        )
                        / 2
                    )
            elif mom == mom1 - mom2 or mom == -mom1 + mom2:
                if field == 0:
                    return I * visc * k_minus_q_square(mom1, mom2).doit() * (uo + 1 - Abs(uo - 1)) / 2
                else:
                    return (
                        I
                        * (
                            -powdenest(sqrt(D(field, visc, mom)), force=True)
                            + k_minus_q_square(mom1, mom2).doit() * visc * (uo + 1)
                        )
                        / 2
                    )
            else:
                if field == 0:
                    return I * visc * mom**2 * (uo + 1 - Abs(uo - 1)) / 2
                else:
                    return I * (-powdenest(sqrt(D(field, visc, mom)), force=True) + mom**2 * visc * (uo + 1)) / 2


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

    xi(k, w) = A*sc_prod(B, k)**2 + alpha(nuo, k, w)*beta(nuo, k, w)

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

    xi_star(k, w) = conjugate(xi(k, w)) = A*sc_prod(B, k)**2 +
                    alpha_star(nuo, k, w)*beta_star(nuo, k, w)

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


class mom(Function):
    """
    mom(k, index) returns the momentum index-th component

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

    vertex_factor_Bbv(k, index_B, index_b, index_v) = I * (mom(k, index_v) * kd(index_B, index_b) -
    A * mom(k, index_b) * kd(index_B, index_v))

    ARGUMENTS:
    k -- momentum, index_B, index_b, index_v -- positive integers
    """

    @classmethod
    def eval(cls, k, index_B, index_b, index_v):
        if isinstance(k, Number) and all(isinstance(m, Integer) for m in [index_B, index_b, index_v]):
            return cls(k, index_B, index_b, index_v)

    def doit(self, deep=True, **hints):
        k, index_B, index_b, index_v = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            index_B = index_B.doit(deep=deep, **hints)
            index_b = index_b.doit(deep=deep, **hints)
            index_v = index_v.doit(deep=deep, **hints)

        return I * (mom(k, index_v) * kd(index_B, index_b) - A * mom(k, index_b) * kd(index_B, index_v))


class vertex_factor_Vvv(Function):
    """
    The vertex_factor_Vvv(k, index_V, index1_v, index2_v) function determines the corresponding
    vertex multiplier (Vvv) of the diagram.

    vertex_factor_Vvv(k, index_V, index1_v, index2_v) = -I * (mom(k, index1_v) * kd(index_V, index2_v) +
    mom(k, index2_v) * kd(index_V, index1_v))

    ARGUMENTS:
    k -- momentum, index_V, index1_v, index2_v -- positive integers
    """

    @classmethod
    def eval(cls, k, index_V, index1_v, index2_v):
        if isinstance(k, Number) and all(isinstance(m, Integer) for m in [index_V, index1_v, index2_v]):
            return cls(k, index_V, index1_v, index2_v)

    def doit(self, deep=True, **hints):
        k, index_V, index1_v, index2_v = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            index_V = index_V.doit(deep=deep, **hints)
            index1_v = index1_v.doit(deep=deep, **hints)
            index2_v = index2_v.doit(deep=deep, **hints)

        return -I * (mom(k, index1_v) * kd(index_V, index2_v) + mom(k, index2_v) * kd(index_V, index1_v))


class vertex_factor_Vbb(Function):
    """
    The vertex_factor_Vbb(k, index_V, index1_b, index2_b) function determines the corresponding
    vertex multiplier (Vbb) of the diagram.

    vertex_factor_Vbb(k, index_V, index1_b, index2_b) = I * (mom(k, index1_b) * kd(index_V, index2_b) +
    mom(k, index2_b) * kd(index_V, index1_b))

    ARGUMENTS:
    k -- momentum, index_V, index1_b, index2_b -- positive integers

    PROPERTIES:

    vertex_factor_Vbb = -vertex_factor_Vvv
    """

    @classmethod
    def eval(cls, k, index_V, index1_b, index2_b):
        if isinstance(k, Number) and all(isinstance(m, Integer) for m in [index_V, index1_b, index2_b]):
            return cls(k, index_V, index1_b, index2_b)

    def doit(self, deep=True, **hints):
        k, index_V, index1_b, index2_b = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            index_V = index_V.doit(deep=deep, **hints)
            index1_b = index1_b.doit(deep=deep, **hints)
            index2_b = index2_b.doit(deep=deep, **hints)

        return I * (mom(k, index1_b) * kd(index_V, index2_b) + mom(k, index2_b) * kd(index_V, index1_b))


class P(Function):
    """
    Transverse projection operator.

    P(k, index1, index2) = kd(index1, index2) - mom(k, index1)*mom(k, index2)/k**2
    """

    @classmethod
    def eval(cls, k, index_1, index_2):
        if k.could_extract_minus_sign():
            # function is even with respect to k by definition
            return cls(-k, index_1, index_2)


class H(Function):
    """
    Helical term.

    H(k, index1, index2) = lcs(index1, index2, index3)*mom(k, index3)
    """


class R(Function):
    """
    Tensor part of the velocity field correlator.
    The function R(k, index1, index2) is defined by the equality

    R(k, index1, index2) = P(k, index1, index2) + I*rho*H(k, index1, index2)

    ARGUMENTS:
    k -- momentum, index_1, index_2 -- positive integers
    """

    def doit(self, deep=True, **hints):
        k, index1, index2 = self.args

        global rho

        if deep:
            k = k.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) + I * rho * H(k, index1, index2)


class Pvv_scalar_part(Function):
    """
    Scalar part of the propagator <vv>.
    The function Pvv_scalar_part(k, w) is defined by the equality

    Pvv_scalar_part(k, w) = beta(nuo, k, w)*beta_star(nuo, k, w)*D_v(k)/(xi(k, w)*xi_star(k, w))

    ARGUMENTS:
    k -- momentum, w -- frequency
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global nuo

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return beta(nuo, k, w) * beta_star(nuo, k, w) * D_v(k) / (xi(k, w) * xi_star(k, w))


class Pvv(Function):
    """
    Propagator <vv>.
    The function Pvv(k, w, index_1, index_2) is defined by the equality

    Pvv(k, w, index_1, index_2) = R(k, index1, index2)*Pvv_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return R(k, index1, index2) * Pvv_scalar_part(k, w).doit()


class PvV_scalar_part(Function):
    """
    Scalar part of the propagator <vV>.
    The function PvV_scalar_part(k, w) is defined by the equality

    PvV_scalar_part(k, w) = beta_star(nuo, k, w) / xi_star(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global nuo

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return beta_star(nuo, k, w) / xi_star(k, w)


class PvV(Function):
    """
    Propagator <vV>.
    The function  Pvv(k, w, index_1, index_2) is defined by the equality

    Pvv(k, w, index_1, index_2) = P(k, index1, index2)*PvV_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) * PvV_scalar_part(k, w).doit()


class PVv_scalar_part(Function):
    """
    Scalar part of the propagator <vv>.
    The function  PVv_scalar_part(k, w) is defined by the equality

    PVv_scalar_part(k, w) = beta(nuo, k, w) / xi(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    PVv_scalar_part(k, w) = complex_conjugate(PvV_scalar_part(k, w))
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global nuo

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return beta(nuo, k, w) / xi(k, w)


class PVv(Function):
    """
    Propagator <Vv>.
    The function  PVv(k, w, index_1, index_2) is defined by the equality

    PVv(k, w, index_1, index_2) = P(k, index1, index2)*PVv_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers

    PROPERTIES:

    PVv(k, w, index_1, index_2) = complex_conjugate(PVv(k, w, index_1, index_2))
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) * PVv_scalar_part(k, w).doit()


class PbB_scalar_part(Function):
    """
    Scalar part of the propagator <bB>.
    The function  PbB_scalar_part(k, w) is defined by the equality

    PbB_scalar_part(k, w) = alpha_star(nuo, k, w) / xi_star(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global nuo

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return alpha_star(nuo, k, w) / xi_star(k, w)


class PbB(Function):
    """
    Propagator <bB>.
    The function  PbB(k, w, index_1, index_2) is defined by the equality

    PbB(k, w, index_1, index_2) = P(k, index1, index2)*PbB_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) * PbB_scalar_part(k, w).doit()


class PBb_scalar_part(Function):
    """
    Scalar part of the propagator <Bb>.
    The function  PBb_scalar_part(k, w) is defined by the equality

    PBb_scalar_part(k, w) = alpha(nuo, k, w) / xi(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    PBb_scalar_part(k, w) = complex_conjugate(PbB_scalar_part(k, w))
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global nuo

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return alpha(nuo, k, w) / xi(k, w)


class PBb(Function):
    """
    Propagator <Bb>.
    The function  PBb(k, w, index_1, index_2) is defined by the equality

    PBb(k, w, index_1, index_2) = P(k, index1, index2)*PBb_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers

    PROPERTIES:

    PBb(k, w, index_1, index_2) = complex_conjugate(PbB(k, w, index_1, index_2))
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) * PBb_scalar_part(k, w).doit()


class Pvb_scalar_part(Function):
    """
    Scalar part of the propagator <vb>.
    The function  Pvb_scalar_part(k, w) is defined by the equality

    Pvb_scalar_part(k, w) = -I*A*beta_star(nuo, k, w)*sc_prod(B, k)*D_v(k)/(xi(k, w)*xi_star(k, w))

    ARGUMENTS:
    k -- momentum, w -- frequency
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global nuo, B, A

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return -I * A * beta_star(nuo, k, w) * sc_prod(B, k) * D_v(k) / (xi(k, w) * xi_star(k, w))


class Pvb(Function):
    """
    Propagator <vb>.
    The function  Pvb(k, w, index_1, index_2) is defined by the equality

    Pvb(k, w, index_1, index_2) = R(k, index1, index2)*Pvb_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return R(k, index1, index2) * Pvb_scalar_part(k, w).doit()


class Pbv_scalar_part(Function):
    """
    Scalar part of the propagator <bv>.
    The function  Pbv_scalar_part(k, w) is defined by the equality

    Pbv_scalar_part(k, w) = I*A*beta(nuo, k, w)*sc_prod(B, k)*D_v(k)/(xi(k, w)*xi_star(k, w))

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    Pbv_scalar_part(k, w) = complex_conjugate(Pvb_scalar_part(k, w))
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global nuo, B, A

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return I * A * beta(nuo, k, w) * sc_prod(B, k) * D_v(k) / (xi(k, w) * xi_star(k, w))


class Pbv(Function):
    """
    Propagator <bv>.
    The function  Pbv(k, w, index_1, index_2) is defined by the equality

    Pbv(k, w, index_1, index_2) = R(k, index1, index2)*Pbv_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers

    PROPERTIES:

    Pbv(k, w, index_1, index_2) = complex_conjugate(Pvb(k, w, index_1, index_2))
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return R(k, index1, index2) * Pbv_scalar_part(k, w).doit()


class Pbb_scalar_part(Function):
    """
    Scalar part of the propagator <bb>.
    The function  Pbb_scalar_part(k, w) is defined by the equality

    Pbb_scalar_part(k, w) = A**2*sc_prod(B, k)**2*D_v(k)/(xi(k, w)*xi_star(k, w))

    ARGUMENTS:
    k -- momentum, w -- frequency
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global A, B

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return A**2 * sc_prod(B, k) ** 2 * D_v(k) / (xi(k, w) * xi_star(k, w))


class Pbb(Function):
    """
    Propagator <bb>.
    The function  Pbb(k, w, index_1, index_2) is defined by the equality

    Pbb(k, w, index_1, index_2) = R(k, index1, index2)*Pbb_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return R(k, index1, index2) * Pbb_scalar_part(k, w).doit()


class PVb_scalar_part(Function):
    """
    Scalar part of the propagator <Vb>.
    The function  PVb_scalar_part(k, w) is defined by the equality

    PVb_scalar_part(k, w) = -I * A * sc_prod(B, k) / xi(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global A, B

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return -I * A * sc_prod(B, k) / xi(k, w)


class PVb(Function):
    """
    Propagator <Vb>.
    The function  PVb(k, w, index_1, index_2) is defined by the equality

    PVb(k, w, index_1, index_2) = P(k, index1, index2)*PVb_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) * PVb_scalar_part(k, w).doit()


class PbV_scalar_part(Function):
    """
    Scalar part of the propagator <bV>.
    The function  PbV_scalar_part(k, w) is defined by the equality

    PbV_scalar_part(k, w) = I * A * sc_prod(B, k) / xi_star(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    PbV_scalar_part(k, w) = complex_conjugate(PVb_scalar_part(k, w))
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global A, B

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return I * A * sc_prod(B, k) / xi_star(k, w)


class PbV(Function):
    """
    Propagator <bV>.
    The function  PbV(k, w, index_1, index_2) is defined by the equality

    PbV(k, w, index_1, index_2) = P(k, index1, index2)*PbV_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers

    PROPERTIES:

    PbV(k, w, index_1, index_2) = complex_conjugate(PVb(k, w, index_1, index_2))
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) * PbV_scalar_part(k, w).doit()


class PBv_scalar_part(Function):
    """
    Scalar part of the propagator <Bv>.
    The function  PBv_scalar_part(k, w) is defined by the equality

    PBv_scalar_part(k, w) = -I * A * sc_prod(B, k) / xi(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    PBv_scalar_part(k, w) = PVb_scalar_part(k, w)
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global A, B

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return -I * A * sc_prod(B, k) / xi(k, w)


class PBv(Function):
    """
    Propagator <Bv>.
    The function  PBv(k, w, index_1, index_2) is defined by the equality

    PBv(k, w, index_1, index_2) = P(k, index1, index2)*PBv_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers

    PROPERTIES:

    PBv(k, w, index_1, index_2) = PVb(k, w, index_1, index_2)
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) * PBv_scalar_part(k, w).doit()


class PvB_scalar_part(Function):
    """
    Scalar part of the propagator <vB>.
    The function  PvB_scalar_part(k, w) is defined by the equality

    PvB_scalar_part(k, w) = I * A * sc_prod(B, k) / xi_star(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency

    PROPERTIES:

    PvB_scalar_part(k, w) = complex_conjugate(PBv_scalar_part(k, w))

    PvB_scalar_part(k, w) = PbV_scalar_part(k, w)
    """

    def doit(self, deep=True, **hints):
        k, w = self.args

        global A, B

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)

        return I * A * sc_prod(B, k) / xi_star(k, w)


class PvB(Function):
    """
    Propagator <vB>.
    The function  PvB(k, w, index_1, index_2) is defined by the equality

    PvB(k, w, index_1, index_2) = P(k, index1, index2)*PvB_scalar_part(k, w)

    ARGUMENTS:
    k -- momentum, w -- frequency, index_1, index_2 -- positive integers

    PROPERTIES:

    PvB(k, w, index_1, index_2) = complex_conjugate(PBv(k, w, index_1, index_2))

    PvB(k, w, index_1, index_2) = PbV(k, w, index_1, index_2)
    """

    def doit(self, deep=True, **hints):
        k, w, index1, index2 = self.args

        if deep:
            k = k.doit(deep=deep, **hints)
            w = w.doit(deep=deep, **hints)
            index1 = index1.doit(deep=deep, **hints)
            index2 = index2.doit(deep=deep, **hints)

        return P(k, index1, index2) * PvB_scalar_part(k, w).doit()
