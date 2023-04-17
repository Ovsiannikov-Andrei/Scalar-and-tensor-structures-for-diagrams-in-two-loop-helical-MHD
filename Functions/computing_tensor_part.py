import time

from sympy import *

from Functions.SymPy_classes import *


# ------------------------------------------------------------------------------------------------------------------#
#                             Calculation of the tensor part of the integrand for a diagram
# ------------------------------------------------------------------------------------------------------------------#


def tensor_structure_term_classification(expr, list_for_expr_args):
    """
    This function compares expr (some kind of one multiplier) with a template
    and adds its arguments to the list.
    """
    match expr:
        case hyb():
            return list_for_expr_args.append(["hyb", list(expr.args)])
        case P():
            return list_for_expr_args.append(["P", list(expr.args)])
        case H():
            return list_for_expr_args.append(["H", list(expr.args)])
        case kd():
            return list_for_expr_args.append(["kd", list(expr.args)])
        case _:  # symbols or numbers
            return list_for_expr_args.append([expr])


def get_tensor_structure_arguments_structure(expr):
    """
    This function returns the expr expression structure as [expr_term_1, expr_term_2, ...],
    expr_term_1 = [term_multiplier_1, term_multiplier_2, ...], where, for example,
    term_multiplier_1 = ['P', [k, 2, 4]].

    ARGUMENTS:

    expr -- expression containing functions hyb(), P(), H(), kd(), symbols rho, A, I and numbers

    OUTPUT DATA EXAMPLE:

    expr = hyb(-q, 9)*kd(11, 10)*P(k, 2, 6) + A*hyb(-q, 10)*kd(11, 9) + kd(12, 9)

    get_tensor_structure_arguments_structure(expr) =
    [
    [-1, A, ['hyb', [q, 10]], ['kd', [11, 9]]], [-1, ['P', [k, 2, 6]], ['hyb', [q, 9]], ['kd', [11, 10]]],
    [['kd', [12, 9]]]
    ]

    """
    expr_new = expand(expr)  # a priori, open all brackets

    list_for_term_multipliers = list()

    number_of_terms_in_expr = len(expr_new.args)

    for i in range(number_of_terms_in_expr):
        term = expr_new.args[i]

        list_for_multiplier_args = list()
        if term.is_Mul:  # adds its arguments to the list
            number_of_multipliers_in_term = len(term.args)

            for j in range(number_of_multipliers_in_term):
                multiplier = term.args[j]
                tensor_structure_term_classification(multiplier, list_for_multiplier_args)

        else:  # term in expr_new has one element
            tensor_structure_term_classification(term, list_for_multiplier_args)

        list_for_term_multipliers.append(list_for_multiplier_args)

    return list_for_term_multipliers


def dosad(zoznam, ind_hod, struktura, pozicia):  # ?????????
    if ind_hod in zoznam:
        return zoznam
    elif ind_hod not in struktura:
        return list()
    elif zoznam[pozicia] != -1:
        return list()
    elif len(zoznam) - 1 == pozicia:
        return zoznam[:pozicia] + [ind_hod]
    else:
        return zoznam[:pozicia] + [ind_hod] + zoznam[pozicia + 1 :]

    
def kronecker_transver_operator(tensor, transver, kronecker):
    """
    The function replace the Kronecker's delta function and transverse projector by the transverse projector. 
    For example: P(k, i, j) * kd(i, l) = P(k, l, j)
    
    ARGUMENTS:
    
    tensor    - Tenzor - projector, kronecker symbol and momenta
    transver  - P_structure - all possible transverse structure in Tensor
    kronecker - kd_structure - all possible kronecker structure in Tensor
    """
    
    for j in kronecker:
        structure = list()
        for i in transver:
            if i[1] == j[0]:
                tensor = tensor.subs(
                    P(i[0], i[1], i[2]) * kd(j[0], j[1]),
                    P(i[0], j[1], i[2]),
                )
                structure.append([i[0], j[1], i[2]])
            elif i[1] == j[1]:
                tensor = tensor.subs(
                    P(i[0], i[1], i[2]) * kd(j[0], j[1]),
                    P(i[0], j[0], i[2]),
                )
                structure.append([i[0], j[0], i[2]])
            elif i[2] == j[0]:
                tensor = tensor.subs(
                    P(i[0], i[1], i[2]) * kd(j[0], j[1]),
                    P(i[0], i[1], j[1]),
                )
                structure.append([i[0], i[1], j[1]])
            elif i[2] == j[1]:
                tensor = tensor.subs(
                    P(i[0], i[1], i[2]) * kd(j[0], j[1]),
                    P(i[0], i[1], j[0]),
                )
                structure.append([i[0], i[1], j[0]])
            if tensor.coeff(kd(j[0], j[1])) == 0:
                break
        
        for i in structure:
            if i is not transver:
                transver.append(i)
        
    return [tensor, transver]


def kronecker_helical_operator(tensor, helical, kronecker):
    """
    The function replace the Kronecker's delta function and helical projector by the helical projector 
    For example: H(k, i, j) * kd(i, l) = H(k, l, j)
    
    ARGUMENTS:
    
    tensor    - Tenzor - projector, kronecker symbol and momenta
    helical   - H_structure - all possible transverse structure in Tensor
    kronecker - kd_structure - all possible kronecker structure in Tensor
    """
    
    for j in kronecker:
        structure = list()
        for i in helical:
            if i[1] == j[0]:
                tensor = tensor.subs(
                    H(i[0], i[1], i[2]) * kd(j[0], j[1]),
                    H(i[0], j[1], i[2]),
                )
                structure.append([i[0], j[1], i[2]])
            elif i[1] == j[1]:
                tensor = tensor.subs(
                    H(i[0], i[1], i[2]) * kd(j[0], j[1]),
                    H(i[0], j[0], i[2]),
                )
                structure.append([i[0], j[0], i[2]])
            elif i[2] == j[0]:
                tensor = tensor.subs(
                    H(i[0], i[1], i[2]) * kd(j[0], j[1]),
                    H(i[0], i[1], j[1]),
                )
                structure.append([i[0], i[1], j[1]])
            elif i[2] == j[1]:
                tensor = tensor.subs(
                    H(i[0], i[1], i[2]) * kd(j[0], j[1]),
                    H(i[0], i[1], j[0]),
                )
                structure.append([i[0], i[1], j[0]])
            if tensor.coeff(kd(j[0], j[1])) == 0:
                break
        
        for i in structure:
            if i is not helical:
                helical.append(i)
        
    return [tensor, helical]
    
    
def momenta_transver_operator(tensor, transver):
    """
    The function replace the momentum and transver projector with the same index by 0. 
    For example: P(k, i, j) * hyb(k, i) = 0
    
    ARBUMENTS:
    
    tensor    - Tenzor - projector, kronecker symbol and momenta
    transver  - P_structure - all possible transverse structure in Tensor
    """

    i = 0
    while i < len(transver):
        j = transver[i]
        if tensor.coeff(hyb( j[0], j[1])) != 0:
            tensor = tensor.subs(P( j[0], j[1], j[2]) * hyb( j[0], j[1]), 0)
        elif tensor.coeff(hyb( - j[0], j[1])) != 0:
            tensor = tensor.subs(P( j[0], j[1], j[2]) * hyb( - j[0], j[1]), 0)
        if tensor.coeff(hyb( j[0], j[2])) != 0:
            tensor = tensor.subs(P( j[0], j[1], j[2]) * hyb( j[0], j[2]), 0)
        elif tensor.coeff(hyb( - j[0], j[2])) != 0:
            tensor = tensor.subs(P( j[0], j[1], j[2]) * hyb( - j[0], j[2]), 0)
        if tensor.coeff(P(j[0], j[1], j[2])) == 0:
            transver.remove( j)
        else:
            if j[0] == -k or j[0] == -q:
                # Replace in the tensor structure the projection operators:  P(-k,i,j) = P(k,i,j)
                tensor = tensor.subs(P( j[0], j[1], j[2]), P( -j[0], j[1], j[2]))
                transver[i][0] = - j[0]
            i += 1
        
    return [tensor, transver]


def momenta_helical_operator(tensor, helical):
    """
    The function replace the momentum and helical operator with the same index by 0. 
    For example: H(k, i, j) * hyb(k, i) = 0
    
    ARBUMENTS:
    
    tensor    - Tenzor - projector, kronecker symbol and momenta
    helical   - H_structure - all possible helical structure in Tensor
    """

    i = 0
    while i < len(helical):
        j = helical[i]
        if tensor.coeff( hyb( j[0], j[1])) != 0:
            tensor = tensor.subs(H( j[0], j[1], j[2]) * hyb( j[0], j[1]), 0)
        elif tensor.coeff( hyb( - j[0], j[1])) != 0:
            tensor = tensor.subs(H( j[0], j[1], j[2]) * hyb( - j[0], j[1]), 0)
        if tensor.coeff( hyb( j[0], j[2])) != 0:
            tensor = tensor.subs(H( j[0], j[1], j[2]) * hyb( j[0], j[2]), 0)
        elif tensor.coeff( hyb( - j[0], j[2])) != 0:
            tensor = tensor.subs(H( j[0], j[1], j[2]) * hyb( - j[0], j[2]), 0)
        if tensor.coeff( H( j[0], j[1], j[2])) == 0:
            helical.remove( j)
        else:
            i += 1
        
    return [tensor, helical]

def transfer_helical_operator(tensor, transver, helical):
    """
    The function replace the product between transver and helical operator with a same momenta and one a same index by helical operator.
    For example: P(k, i, j) * H(k, i, l) = H (k, l, j)
    
    ARGUMENTS:
    
    tensor    - Tenzor - projector, kronecker symbol and momenta
    helical   - H_structure - all possible helical structure in Tensor
    transver  - P_structure - all possible transver structure in Tensor 
    """
    
    i = 0
    while len(helical) > i:
        j = helical[i]
        for in2 in transver:
            if j[0] == in2[0] and tensor.coeff( H( j[0], j[1], j[2]) * P(in2[0], in2[1], in2[2])) != 0:
                if j[1] == in2[1]:
                    tensor = tensor.subs(
                        H( j[0], j[1], j[2]) * P( in2[0], in2[1], in2[2]),
                        H( j[0], in2[2], j[2]),
                    )
                    helical += [[ j[0], in2[2], j[2]]]
                elif j[1] == in2[2]:
                    tensor = tensor.subs(
                        H( j[0], j[1], j[2]) * P(in2[0], in2[1], in2[2]),
                        H( j[0], in2[1], j[2]),
                    )
                    helical += [[ j[0], in2[1], j[2]]]
                elif j[2] == in2[1]:
                    tensor = tensor.subs(
                        H( j[0], j[1], j[2]) * P(in2[0], in2[1], in2[2]),
                        H( j[0], j[1], in2[2]),
                    )
                    helical += [[ j[0], j[1], in2[2]]]
                elif j[2] == in2[2]:
                    tensor = tensor.subs(
                        H( j[0], j[1], j[2]) * P(in2[0], in2[1], in2[2]),
                        H( j[0], j[1], in2[1]),
                    )
                    helical += [[ j[0], j[1], in2[1]]]
        if tensor.coeff(H( j[0], j[1], j[2])) == 0:
            helical.remove( j)
        else:
            i += 1
            
    return [tensor, helical]

def transver_transver_operator(tensor, transver):
    """
    The function replace the product between transver and transver operator with a same momenta and one a same index by transver operator.
    For example: P(k, i, j) * P(k, i, l) = P (k, l, j)
    
    ARGUMENTS:
    
    tensor    - Tenzor - projector, kronecker symbol and momenta
    transver  - P_structure - all possible transver structure in Tensor 
    """    
    
    i = 0
    while len(transver) > i:
        in1 = transver[i]
        structure = list()
        for j in range(i + 1, len(transver)):
            in2 = transver[j]
            if in1[0] == in2[0] and tensor.coeff(P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2])) != 0:
                if in1[1] == in2[1]:
                    tensor = tensor.subs(
                        P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        P(in1[0], in2[2], in1[2]),
                    )
                    structure.append([in1[0], in2[2], in1[2]])
                elif in1[1] == in2[2]:
                    tensor = tensor.subs(
                        P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        P(in1[0], in2[1], in1[2]),
                    )
                    structure.append([in1[0], in2[1], in1[2]])
                elif in1[2] == in2[1]:
                    tensor = tensor.subs(
                        P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        P(in1[0], in1[1], in2[2]),
                    )
                    structure.append([in1[0], in1[1], in2[2]])
                elif in1[2] == in2[2]:
                    tensor = tensor.subs(
                        P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2]),
                        P(in1[0], in1[1], in2[1]),
                    )
                    structure.append([in1[0], in1[1], in2[1]])
        if tensor.coeff(P(in1[0], in1[1], in1[2])) == 0:
            transver.remove(in1)
        else:
            i += 1
        transver = transver + structure
        
    return [tensor, transver]

def kronecker_momenta(tensor, structure):
    """
    The function replace the kronecker symbol and momentum with a same index by momentum.
    For example: hyb( k, i) * kd(i, j) = hyb(k, j)
        
    ARGUMENTS:
    
    tensor     - Tenzor - projector, kronecker symbol and momenta
    kronecker  - kd_structure - all possible transver structure in Tensor 
    """
    
    i = 0
    while i == 0:
        for j in structure:
            part = tensor.coeff( kd( j[0], j[1]))
            if part !=0:
                if part.coeff( hyb( k, j[0])) != 0:
                    tensor = tensor.subs( kd( j[0], j[1]) * hyb( k, j[0]), hyb( k, j[1]))
                    i += 1
                if part.coeff( hyb( k, j[1])) != 0:
                    tensor = tensor.subs( kd( j[0], j[1]) * hyb( k, j[1]), hyb( k, j[0]))
                    i += 1
                if part.coeff( hyb( q, j[0])) != 0:
                    tensor = tensor.subs( kd( j[0], j[1]) * hyb( q, j[0]), hyb( q, j[1]))
                    i += 1
                if part.coeff( hyb( q, j[1])) != 0:
                    tensor = tensor.subs( kd( j[0], j[1]) * hyb( q, j[1]), hyb( q, j[0]))
                    i += 1
                if part.coeff( hyb( p, j[0])) != 0:
                    tensor = tensor.subs( kd( j[0], j[1]) * hyb( p, j[0]), hyb( p, j[1]))
                    i += 1
                if part.coeff( hyb( p, j[1])) != 0:
                    tensor = tensor.subs( kd( j[0], j[1]) * hyb( p, j[1]), hyb( p, j[0]))
                    i += 1
            else:
                structure.remove(j)
                i = 1
                break
        if i != 0:
            i = 0
        else:
            break
    
    return [tensor, structure]

def momenta_momenta_helical_operator( tensor, helical):
    """
    The function replace the product among helical term and two same momenta with by the 0.
    For example: H( k, i, j) * hyb( q, i) * hyb( q, j) = 0
    
    ARGUMENTS:
    
    tensor    - Tenzor - projector, kronecker symbol and momenta
    helical   - H_structure - all possible transverse structure in Tensor
    """
    
    i = 0
    while len(helical) > i:  # calculation for helical term
        j = helical[i]
        part = tensor.coeff(H( j[0], j[1], j[2]))
        if j[0] == k and part.coeff( hyb( q, j[1]) * hyb( q, j[2])) != 0:      
            tensor = tensor.subs( H( k, j[1], j[2]) * hyb( q, j[1]) * hyb( q, j[2]), 0)
        if j[0] == q and part.coeff( hyb( k, j[1]) * hyb( k, j[2])) != 0:
            tensor = tensor.subs( H( q, j[1], j[2]) * hyb( k, j[1]) * hyb( k, j[2]), 0)            
           
        i += 1
        
    return [tensor, helical]


    
def computing_tensor_structures(moznost, indexb, indexB, P_structure, H_structure, kd_structure, hyb_structure, Tenzor):
    """
    This function calculates the integrand of the corresponding diagram in terms of tensor and scalar parts.

    ARGUMENTS:

    moznost
    indexb, indexB
    P_structure
    H_structure
    kd_structure
    hyb_structure
    Tenzor

    OUTPUT DATA EXAMPLE:
    """

    t = time.time()
    # print(f"{Tenzor}")
    Tenzor = expand(Tenzor)  # The final tesor structure from the diagram

    # print(f"{kd_structure}")
    # print(f"{P_structure}")
    # print(f"{H_structure}")
    # exit()

    # What I need from the Tenzor structure
    Tenzor = rho * Tenzor.coeff(rho**stupen)
    Tenzor = expand(Tenzor.subs(I**5, I))  # calculate the imaginary unit
    # Tenzor = Tenzor.subs(A, 1)              # It depends on which part we want to calculate from the vertex Bbv
    # print(Tenzor)
    # We are interested in the leading (proportional to p) contribution to the diagram asymptotic, when p --> 0.
    print(f"step 0: {round(time.time() - t, 1)} sec")

    [Tenzor, P_structure] = kronecker_transver_operator(Tenzor, P_structure, kd_structure)
    [Tenzor, H_structure] = kronecker_helical_operator(Tenzor, H_structure, kd_structure)

    print(f"step 1: {round(time.time() - t, 1)} sec")

    [Tenzor, P_structure] = momenta_transver_operator(Tenzor, P_structure)
    [Tenzor, H_structure] = momenta_helical_operator(Tenzor, H_structure)

    print(f"step 2: {round(time.time() - t, 1)} sec")

    [Tenzor, H_structure] = transfer_helical_operator(Tenzor, P_structure, H_structure)

    print(f"step 3: {round(time.time() - t, 1)} sec")

    [Tenzor, P_structure] = transver_transver_operator(Tenzor, P_structure)

    print(f"step 4: {round(time.time() - t, 1)} sec")

    for i in hyb_structure:  # replace: hyb(-k+q, i) = -hyb(k, i) + hyb(q, i)
        k_c = i[0].coeff(k)
        q_c = i[0].coeff(q)
        if k_c != 0 or q_c != 0:
            Tenzor = Tenzor.subs(hyb(i[0], i[1]), (k_c * hyb(k, i[1]) + q_c * hyb(q, i[1])))
            
    Tenzor = expand(Tenzor)
    [Tenzor, P_structure] = momenta_transver_operator(Tenzor, P_structure)
    [Tenzor, H_structure] = momenta_helical_operator(Tenzor, H_structure)    

    kd_structure = list()
    for i in P_structure:  # Define transverse projection operator P(k,i,j) = kd(i,j) - hyb(k,i)*hyb(k,j)/k^2
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
    
    [Tenzor, H_structure] = momenta_helical_operator( Tenzor, H_structure)
    [Tenzor, H_structure] = momenta_momenta_helical_operator( Tenzor, H_structure)

    print(f"step 6: {round(time.time() - t, 1)} sec")

    [Tenzor, kd_structure] = kronecker_momenta( Tenzor, kd_structure)

    print(f"step 7: {round(time.time() - t, 1)} sec")
    
    [Tenzor, H_structure] = momenta_helical_operator( Tenzor, H_structure)
    [Tenzor, H_structure] = momenta_momenta_helical_operator( Tenzor, H_structure)
    [Tenzor, H_structure] = kronecker_helical_operator( Tenzor, H_structure, kd_structure)

    print(f"step 8: {round(time.time() - t, 1)} sec")

    p_structure = list()  # list of indeces for momentum p in Tenzor
    k_structure = list()  # list of indeces for momentum k in Tenzor
    q_structure = list()  # list of indeces for momentum q in Tenzor
    # It combines quantities with matching indices.
    for in1 in range(len(moznost)):
        Tenzor = Tenzor.subs(hyb(k, in1) ** 2, k**2)
        Tenzor = Tenzor.subs(hyb(q, in1) ** 2, q**2)
        Tenzor = Tenzor.subs(hyb(q, in1) * hyb(k, in1), k * q * z)
        # k.q = k q z, where z = cos(angle) = k . q/ |k| /|q|
        if Tenzor.coeff(hyb(p, in1)) != 0:
            # H( , j, s) hyb( ,j) hyb( ,s) hyb( , indexb) hyb(p, i) hyb(q, i) hyb(q, indexB) = 0
            # or H( , j, indexb) hyb( ,j) hyb(p, i) hyb(q, i) hyb(q, indexB) = 0
            Tenzor = Tenzor.subs(hyb(p, in1) * hyb(q, in1) * hyb(q, indexB), 0)
            Tenzor = Tenzor.subs(hyb(p, in1) * hyb(k, in1) * hyb(k, indexB), 0)
            Tenzor = Tenzor.subs(hyb(p, in1) * hyb(q, in1) * hyb(q, indexb), 0)
            Tenzor = Tenzor.subs(hyb(p, in1) * hyb(k, in1) * hyb(k, indexb), 0)
            p_structure += [in1]
        if Tenzor.coeff(hyb(q, in1)) != 0:
            q_structure += [in1]
        if Tenzor.coeff(hyb(k, in1)) != 0:
            k_structure += [in1]

    Tenzor = Tenzor.subs(hyb(q, indexb) * hyb(q, indexB), 0)
    # delete zero values in the Tenzor: H( ,i,j) hyb(p, i) hyb( ,j) hyb(k, indexB) hyb(k, indexb) = 0
    Tenzor = Tenzor.subs(hyb(k, indexb) * hyb(k, indexB), 0)

    print(f"step 9: {round(time.time() - t, 1)} sec")

    # calculation of H structure - For this particular case, one of the external indices (p_s, b_i or B_j) is paired with a helicity term.
    # we will therefore use the information that, in addition to the helicity term H( , i,j), they can be multiplied by a maximum of three internal momenta.
    # For examle: H(k, indexb, j) hyb(q, j) hyb(k, indexB) hyb(q, i) hyb(p, i) and thus in this step I will calculate all possible combinations for this structure.
    # In this case, helical term H(k, i, j) = epsilon(i,j,s) k_s /k

    i = 0
    while i < len(H_structure):  # I go through all of them helical term H( , , )
        in1 = H_structure[i]
        while Tenzor.coeff(H(in1[0], in1[1], in1[2])) == 0:
            # if the H( , , ) structure is no longer in the Tenzor, I throw it away
            H_structure.remove(in1)
            if len(H_structure) < i:
                break
            else:
                in1 = H_structure[i]
        if in1[0] == k:
            # it create a list where momenta are stored in the positions and indexes pf momenta. - I have for internal momenta k or q
            kombinacia = in1 + [q, -1, p, -1, k, -1, q, -1]
            # [ k, indexH, indexH, q, -1, p, -1,  k, -1, q, -1 ]
        else:
            kombinacia = in1 + [k, -1, p, -1, k, -1, q, -1]
        if indexB == in1[1]:
            # it looks for whether the H helicity term contains an idex corresponding to the externa field b or B
            kombinacia[4] = in1[2]
        elif indexB == in1[2]:
            kombinacia[4] = in1[1]
        elif indexb == in1[1]:
            kombinacia[4] = in1[2]
        elif indexb == in1[2]:
            kombinacia[4] = in1[1]
        kombinacia_old = [kombinacia]
        # search whether the index B or b is in momenta not associated with the helicity term
        kombinacia_new = list()
        kombinacia_new.append(dosad(kombinacia_old[0], indexB, k_structure, 8))
        # it create and put the field index B in to the list on the position 8: hyb(k,indexB)
        kombinacia = dosad(kombinacia_old[0], indexB, q_structure, 10)
        # it create and put the field index B in to the list on the position 10: hyb(q,indexB)
        if kombinacia not in kombinacia_new:
            kombinacia_new.append(kombinacia)
        kombinacia_old = kombinacia_new
        kombinacia_new = list()
        for in2 in kombinacia_old:
            # # it create and put the field index b in to the list with index
            kombinacia_new.append(dosad(in2, indexb, k_structure, 8))
            kombinacia = dosad(in2, indexb, q_structure, 10)
            if kombinacia not in kombinacia_new:
                kombinacia_new.append(kombinacia)
            if list() in kombinacia_new:
                kombinacia_new.remove(list())
        kombinacia_old = kombinacia_new
        kombinacia_new = list()
        # I know which indexes are free. I know where the fields B or b are located.
        for in2 in kombinacia_old:
            # I have free two indecies and I start summing in the tensor structure
            if in2[4] == -1 and in2[0] == k:
                # it calculate if there is H(k,...,...) and the indecies of the external fields are outside
                if in2[1] in p_structure and in2[2] in q_structure:
                    # H(k, i, j) hyb(q, j) hyb(p, i) hyb(k, indexb) hyb(q, indexB) = ... or  H(k, i, j) hyb(q, j) hyb(p, i) hyb(k, indexB) hyb(q, indexb)
                    Tenzor = Tenzor.subs(
                        H(k, in2[1], in2[2]) * hyb(q, in2[2]) * hyb(p, in2[1]) * hyb(k, in2[8]) * hyb(q, in2[10]),
                        hyb(p, s) * lcs(s, in2[10], in2[8]) * q**2 * k * (1 - z**2) / d / (d + 2),
                    )
                if in2[2] in p_structure and in2[1] in q_structure:
                    Tenzor = Tenzor.subs(
                        H(k, in2[1], in2[2]) * hyb(q, in2[1]) * hyb(p, in2[2]) * hyb(k, in2[8]) * hyb(q, in2[10]),
                        -hyb(p, s) * lcs(s, in2[10], in2[8]) * q**2 * k * (1 - z**2) / d / (d + 2),
                    )
            if in2[4] == -1 and in2[0] == q:  #
                if in2[1] in p_structure and in2[2] in k_structure:
                    Tenzor = Tenzor.subs(
                        H(q, in2[1], in2[2]) * hyb(k, in2[2]) * hyb(p, in2[1]) * hyb(k, in2[8]) * hyb(q, in2[10]),
                        -hyb(p, s) * lcs(s, in2[10], in2[8]) * q * k**2 * (1 - z**2) / d / (d + 2),
                    )
                if in2[2] in p_structure and in2[1] in k_structure:
                    Tenzor = Tenzor.subs(
                        H(q, in2[1], in2[2]) * hyb(k, in2[1]) * hyb(p, in2[2]) * hyb(k, in2[8]) * hyb(q, in2[10]),
                        hyb(p, s) * lcs(s, in2[10], in2[8]) * q * k**2 * (1 - z**2) / d / (d + 2),
                    )
            if in2[8] == -1 and in2[0] == k:
                # H(k, indexb, j) hyb(q, j) hyb(p, i) hyb(k, i) hyb(q, indexB) = ... or  H(k, indexB, j) hyb(q, j) hyb(p, i) hyb(k, i) hyb(q, indexb)
                for in3 in p_structure:
                    if in2[1] == indexb or in2[1] == indexB:
                        Tenzor = Tenzor.subs(
                            H(k, in2[1], in2[2]) * hyb(q, in2[2]) * hyb(p, in3) * hyb(k, in3) * hyb(q, in2[10]),
                            -hyb(p, s) * lcs(s, in2[10], in2[1]) * q**2 * k * (1 - z**2) / d / (d + 2),
                        )
                    if in2[2] == indexb or in2[2] == indexB:
                        Tenzor = Tenzor.subs(
                            H(k, in2[1], in2[2]) * hyb(q, in2[1]) * hyb(p, in3) * hyb(k, in3) * hyb(q, in2[10]),
                            hyb(p, s) * lcs(s, in2[10], in2[2]) * q**2 * k * (1 - z**2) / d / (d + 2),
                        )
            if in2[8] == -1 and in2[0] == q:
                for in3 in p_structure:
                    if in2[1] == indexb or in2[1] == indexB:
                        Tenzor = Tenzor.subs(
                            H(q, in2[1], in2[2]) * hyb(k, in2[2]) * hyb(p, in3) * hyb(k, in3) * hyb(q, in2[10]),
                            hyb(p, s) * lcs(s, in2[10], in2[1]) * q * k**2 * (1 - z**2) / d / (d + 2),
                        )
                    if in2[2] == indexb or in2[2] == indexB:
                        Tenzor = Tenzor.subs(
                            H(q, in2[1], in2[2]) * hyb(k, in2[1]) * hyb(p, in3) * hyb(k, in3) * hyb(q, in2[10]),
                            -hyb(p, s) * lcs(s, in2[10], in2[2]) * q * k**2 * (1 - z**2) / d / (d + 2),
                        )
            if in2[10] == -1 and in2[0] == k:
                for in3 in p_structure:
                    if in2[1] == indexb or in2[1] == indexB:
                        Tenzor = Tenzor.subs(
                            H(k, in2[1], in2[2]) * hyb(q, in2[2]) * hyb(p, in3) * hyb(k, in2[8]) * hyb(q, in3),
                            -hyb(p, s) * lcs(s, in2[1], in2[8]) * q**2 * k * (1 - z**2) / d / (d + 2),
                        )
                    if in2[2] == indexb or in2[2] == indexB:
                        Tenzor = Tenzor.subs(
                            H(k, in2[1], in2[2]) * hyb(q, in2[1]) * hyb(p, in3) * hyb(k, in2[8]) * hyb(q, in3),
                            hyb(p, s) * lcs(s, in2[2], in2[8]) * q**2 * k * (1 - z**2) / d / (d + 2),
                        )
            if in2[10] == -1 and in2[0] == q:
                for in3 in p_structure:
                    if in2[1] == indexb or in2[1] == indexB:
                        Tenzor = Tenzor.subs(
                            H(q, in2[1], in2[2]) * hyb(k, in2[2]) * hyb(k, in2[8]) * hyb(p, in3) * hyb(q, in3),
                            hyb(p, s) * lcs(s, in2[1], in2[8]) * q * k**2 * (1 - z**2) / d / (d + 2),
                        )
                    if in2[2] == indexb or in2[2] == indexB:
                        Tenzor = Tenzor.subs(
                            H(q, in2[1], in2[2]) * hyb(k, in2[1]) * hyb(k, in2[8]) * hyb(p, in3) * hyb(q, in3),
                            -hyb(p, s) * lcs(s, in2[2], in2[8]) * q * k**2 * (1 - z**2) / d / (d + 2),
                        )
        i += 1

    print(f"step 10: {round(time.time() - t, 1)} sec")

    for in1 in H_structure:
        # calculate the structure where there are two external momentums: H(momentum, i, indexB)* p(i) hyb( , indexb) and other combinations except H(momentum, indexB, indexb) hyb(p, i) hyb(k, i)
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
    Tenzor = Tenzor.subs(lcs(s, indexb, indexB), -lcs(s, indexB, indexb))

    Tenzor = simplify(Tenzor)

    print(f"step 11: {round(time.time() - t, 1)} sec")

    print(f"\nDiagram tensor structure after computing tensor convolutions: \n{Tenzor}")

    return Tenzor
