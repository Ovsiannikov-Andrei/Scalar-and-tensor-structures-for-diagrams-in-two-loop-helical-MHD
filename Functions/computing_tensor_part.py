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


def external_index(tensor_structure, ext_index, index_list, position):
    """
    The function put the index of external field on the position in list, if the index of external field is in the index_list. On the other hand, the result is sn empty list.   
    
    ARGUMENTS: 
    
    tensor_structure - combination of index and momenta: [i, j, "q", l, "k", j, "p", -1, "k", -1, "q", -1]
    ext_index  - indexb or indexB
    index_list - list of index for momenta k or q
    position   - position on which the index is change to the ex_index
    """
    if ext_index in tensor_structure:
        return tensor_structure
    elif ext_index not in index_list:
        return list()
    elif tensor_structure[position] != -1:
        return list()
    elif len(tensor_structure) - 1 == position:
        return tensor_structure[:position] + [ext_index]
    else:
        return tensor_structure[:position] + [ext_index] + tensor_structure[ position+1:]    

    
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

def four_indices_external_fields( helical, indexb, indexB, k_indices, q_indices):
    """
    The function looking for the index structure in the helical part with four internal momenta. The result is the list with momenta and indices
    For Example: H(k, i_b, j) hyb(q, j) hyb(k, indexB) -> [i_b, j, "k", l, "q", j, "p", i_p, "k", i_B, "q", i_]
    H(q, i_b, j) hyb(k, j) hyb(k, indexB) ... -> [i_b, j, l, "q", l, "k", j, "p", -2, "k", i_B, "q", -1]
      
    ARGUMENTS:
    helical   - is helical term H(k, i, j) = epsilon(i,j,l) k_l /k
    indexb    - index correspond to the external field b
    indexB    - index correspond to the external field b'
    k_indices - k_structure - list of possible indices for momentum k
    q_indices - q_structure - list of possible indices for momentum q
    """
       
    if helical[0] == k:
        structure = [helical[1], helical[2], s, k, s, q, -1, p, -2, k, -1, q, -1]
    else:
        structure = [helical[1], helical[2], s, q, s, k, -1, p, -2, k, -1, q, -1]

    if indexB == structure[0]:
        structure[6] = structure[1]
    elif indexB == structure[1]:
        structure[6] = structure[0]
    elif indexb == structure[0]:
        structure[6] = structure[1]
    elif indexb == structure[1]:
        structure[6] = structure[0]
    
    if structure[3] == q:
        structure = structure[0:3] + structure[5:7] + structure[3:5] + structure[7:13]
        
    structure_old = [external_index( structure, indexB, k_indices, 10)]
    structure_new = [external_index( structure, indexB, q_indices, 12)]
    if structure_new not in structure_old:
        structure_old += structure_new
    
    structure_new = list()
    for i in structure_old:
        structure_new.append(external_index( i, indexb, k_indices, 10))
        structure = external_index( i, indexb, q_indices, 12)
        if structure not in structure_new:
            structure_new.append(structure)
        if list() in structure_new:
            structure_new.remove(list())
                
    return structure_new

def four_indices_external_momentum( structure, p_indices, k_indices, q_indices):
    """
    The function give the specified indecies structure for index of external momentum among four momenta and helical term in a tensor structure.
    For Example: H(k, i_b, j) hyb(q, j) hyb(k, indexB)  hyb(p, i) hyb(q, i) -> [i_b, j, "k", l, "q", j, "p", i_p, "k", i_B, "q", i_p]
    
    ARGUMENTS:
    structure   - the structure what it is needed to be replace with
    p_indices - list all indices for external momentum 
    k_indices - list all indices for k momentum 
    q_indices - list all indices for q momentum 
    """
    
    i = structure.index(-1)
    result = list()
    if i == 4:
        if (structure[0] in p_indices) and (structure[1] in k_indices):
            result += [structure[0:4] + [structure[1]] + structure[5:8] + [structure[0]] + structure[9:13]]
        if (structure[1] in p_indices) and (structure[0] in k_indices):
            result += [structure[0:4] + [structure[0]] + structure[5:8] + [structure[1]] + structure[9:13]]        
    if i == 6:
        if p_indices.count(structure[0]) == 1 and q_indices.count(structure[1]) == 1:
            result += [structure[0:6] + [structure[1]] + structure[7:8] + [structure[0]] + structure[9:13]]
        if p_indices.count(structure[1]) == 1 and q_indices.count(structure[0]) == 1:
            result += [structure[0:6] + [structure[0]] + structure[7:8] + [structure[1]] + structure[9:13]]        
    if i == 10:
        indices = list(set(p_indices).intersection(k_indices))
        for j in indices:
            result += [structure[0:8] + [j] + structure[9:10] + [j] + structure[11:13]]        
    if i == 12:
        indices = list(set(p_indices).intersection(q_indices))
        for j in indices:
            result += [structure[0:8] + [j] + structure[9:12] + [j]]
    
    return result


def scalar_result( momenta, part):
    """
    The function replace the momenta the equivalent form proportional Lambda or B magnetic field.
    
    ARGUMENTS:
    
    momenta  - the structure of momenta
    part     - lambda or B field 
    """
    
    if momenta == k**2 * q**2:
        if part == "lambda":
            return q**2 * k**2 * (1 - z**2) / d / (d + 2)
        if part == "Bfield":
            return  ((2*(B*k*z_k)*(B*q*z_q)*(k*q*z) - (B*q*z_q)**2 * k**2  - (B*k*z_k)**2 * q**2 + B**2*(k**2 * q**2 - (k*q*z)**2))/(B**2*(d-2)*(d-1)))
    if momenta == k**2:
        if part == "lambda":
            return k**2 / d 
        if part == "Bfield":
            return ( k**2 - (B*k*z_k)**2 / B**2 )/(d - 1)
    if momenta == q**2:
        if part == "lambda":
            return q**2 / d 
        if part == "Bfield":
            return ( q**2 - (B*q*z_q)**2 / B**2 )/(d - 1)
    if momenta == k*q:
        if part == "lambda":
            return k * q * z / d 
        if part == "Bfield":
            return ( k * q * z - (B*k*z_k) * (B*q*z_q) / B**2 )/(d - 1)

    
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
    
    Tenzor = Tenzor.subs(hyb(q, indexb) * hyb(q, indexB), 0)
    # delete zero values in the Tenzor: H( ,i,j) hyb(p, i) hyb( ,j) hyb(k, indexB) hyb(k, indexb) = 0
    Tenzor = Tenzor.subs(hyb(k, indexb) * hyb(k, indexB), 0)        
    
    [Tenzor, H_structure] = momenta_helical_operator( Tenzor, H_structure)
    [Tenzor, H_structure] = momenta_momenta_helical_operator( Tenzor, H_structure)
    
    x = 0
    while len(kd_structure) > 0:
        [Tenzor, H_structure] = kronecker_helical_operator( Tenzor, H_structure, kd_structure)
        for y in kd_structure:
            if Tenzor.coeff( kd( y[0], y[1])) == 0:
                kd_structure.remove(y)
        if x == int(len(moznost)/3 - 1):  # Solve a problem: for example kd(indexb, indexB) -> len( kd_structure) > 1, I do not have a cycle
            break
        else: 
            x += 1
    
    [Tenzor, H_structure] = momenta_helical_operator( Tenzor, H_structure)
    [Tenzor, H_structure] = momenta_momenta_helical_operator( Tenzor, H_structure)

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
            p_structure += [in1]  # list correspond to the index of momentum 
        if Tenzor.coeff(hyb(q, in1)) != 0:
            q_structure += [in1]
        if Tenzor.coeff(hyb(k, in1)) != 0:
            k_structure += [in1]


    print(f"step 9: {round(time.time() - t, 1)} sec")

    y = 0
    while y < len(H_structure):     
        combination = four_indices_external_fields(H_structure[y], indexb, indexB, k_structure, q_structure)
        combination_new = list()
        for x in combination:
             combination_new += four_indices_external_momentum(x, p_structure, k_structure, q_structure)
        Lambda_term = q**2 * k**2 * (1 - z**2) / d / (d + 2)
        B_term = k *q
        for x in combination_new:
            if x[8] == x[12]: 
                if x[1] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * hyb(q, x[6]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        - hyb(p, s) * lcs(s, x[0], x[10]) * scalar_result( k**2 * q**2, "Bfield") / k,
                    )
                if x[1] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * hyb(k, x[4]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        hyb(p, s) * lcs(s, x[0], x[10]) * scalar_result( k**2 * q**2, "Bfield") / q,
                    )
                if x[0] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * hyb(q, x[6]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        hyb(p, s) * lcs(s, x[1], x[10])  * scalar_result( k**2 * q**2, "Bfield") / k,
                    )
                if x[0] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * hyb(k, x[4]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        - hyb(p, s) * lcs(s, x[1], x[10]) * scalar_result( k**2 * q**2, "Bfield") / q,
                    )
            if x[8] == x[10]:
                if x[1] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * hyb(q, x[6]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        hyb(p, s) * lcs(s, x[0], x[12]) * scalar_result( k**2 * q**2, "Bfield") / k,
                    )
                if x[1] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * hyb(k, x[4]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        - hyb(p, s) * lcs(s, x[0], x[12]) * scalar_result( k**2 * q**2, "Bfield") / q,
                    )
                if x[0] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * hyb(q, x[6]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        - hyb(p, s) * lcs(s, x[1], x[12])  * scalar_result( k**2 * q**2, "Bfield") / k,
                    )
                if x[0] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * hyb(k, x[4]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        hyb(p, s) * lcs(s, x[1], x[12]) * scalar_result( k**2 * q**2, "Bfield") / q,
                    )
            if x[8] == x[0]:
                if x[1] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * hyb(q, x[6]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        - hyb(p, s) * lcs(s, x[10], x[12]) * scalar_result( k**2 * q**2, "Bfield") / k,
                    )
                if x[1] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * hyb(k, x[4]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        hyb(p, s) * lcs(s, x[10], x[12])  * scalar_result( k**2 * q**2, "Bfield") / q,
                    )
            if x[8] == x[1]:
                if x[0] == x[6]:
                    Tenzor = Tenzor.subs(
                        H(k, x[0], x[1]) * hyb(q, x[6]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        hyb(p, s) * lcs(s, x[10], x[12]) * scalar_result( k**2 * q**2, "Bfield") / k,
                    )
                if x[0] == x[4]:
                    Tenzor = Tenzor.subs(
                        H(q, x[0], x[1]) * hyb(k, x[4]) * hyb(p, x[8]) * hyb(k, x[10]) * hyb(q, x[12]),
                        - hyb(p, s) * lcs(s, x[10], x[12]) * scalar_result( k**2 * q**2, "Bfield") / q,
                    )
        y += 1

    print(f"step 10: {round(time.time() - t, 1)} sec")

    for x in H_structure:
        if Tenzor.coeff(H( x[0], x[1], x[2])) != 0:
            if x[1] in p_structure and x[2] == indexb:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, x[1], indexb) * hyb(p, x[1]) * hyb(k, indexB),
                        hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k**2, "Bfield") / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, x[1], indexb) * hyb(p, x[1]) * hyb(q, indexB),
                        hyb(p, s) * lcs(s, indexb, indexB)  * scalar_result( k*q, "Bfield") / k,
                    )
                if x[0] == q:
                    Tenzor = Tenzor.subs(
                        H(q, x[1], indexb) * hyb(p, x[1]) * hyb(q, indexB),
                        hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( q**2, "Bfield") / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, x[1], indexb) * hyb(p, x[1]) * hyb(k, indexB),
                        hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k*q, "Bfield") / q,
                    )
            if x[2] in p_structure and x[1] == indexb:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, indexb, x[2]) * hyb(p, x[2]) * hyb(k, indexB),
                        - hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k**2, "Bfield") / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, indexb, x[2]) * hyb(p, x[2]) * hyb(q, indexB),
                        - hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k*q, "Bfield") / k,
                    )
                if x[0] == q:
                    Tenzor = Tenzor.subs(
                        H(q, indexb, x[2]) * hyb(p, x[2]) * hyb(q, indexB),
                        - hyb(p, s) * lcs(s, indexb, indexB)  * scalar_result( q**2, "Bfield") / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, indexb, x[2]) * hyb(p, x[2]) * hyb(k, indexB),
                        - hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k*q, "Bfield") / q,
                    )
            if x[1] in p_structure and x[2] == indexB:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, x[1], indexB) * hyb(p, x[1]) * hyb(k, indexb),
                        - hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k**2, "Bfield") / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, x[1], indexB) * hyb(p, x[1]) * hyb(q, indexb),
                        - hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k*q, "Bfield") / k,
                    )
                if x[0] == q: 
                    Tenzor = Tenzor.subs(
                        H(q, x[1], indexB) * hyb(p, x[1]) * hyb(q, indexb),
                        - hyb(p, s) * lcs(s, indexb, indexB)  * scalar_result( q**2, "Bfield") / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, x[1], indexB) * hyb(p, x[1]) * hyb(k, indexb),
                        - hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k*q, "Bfield") / q,
                    )
            if x[2] in p_structure and x[1] == indexB:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, indexB, x[2]) * hyb(p, x[2]) * hyb(k, indexb),
                        hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k**2, "Bfield") / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, indexB, x[2]) * hyb(p, x[2]) * hyb(q, indexb),
                        hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k*q, "Bfield") / k,
                    )
                else:
                    Tenzor = Tenzor.subs(
                        H(q, indexB, x[2]) * hyb(p, x[2]) * hyb(q, indexb),
                        hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( q**2, "Bfield") / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, indexB, x[2]) * hyb(p, x[2]) * hyb(k, indexb),
                        hyb(p, s) * lcs(s, indexb, indexB) * scalar_result( k*q, "Bfield") / q,
                    )
        if Tenzor.coeff(H( x[0], x[1], x[2])) != 0:
            for y in p_structure:
                if x[0] == k:
                    Tenzor = Tenzor.subs(
                        H(k, x[1], x[2]) * hyb(p, y) * hyb(k, y),
                        hyb(p, s) * lcs(s, x[1], x[2]) * scalar_result( k**2, "Bfield") / k,
                    )
                    Tenzor = Tenzor.subs(
                        H(k, x[1], x[2]) * hyb(p, y) * hyb(q, y),
                        hyb(p, s) * lcs(s, x[1], x[2]) * scalar_result( k*q, "Bfield") / k,
                    )
                if x[0] == q:
                    Tenzor = Tenzor.subs(
                        H(q, x[1], x[2]) * hyb(p, y) * hyb(q, y),
                        hyb(p, s) * lcs(s, x[1], x[2]) * scalar_result( q**2, "Bfield") / q,
                    )
                    Tenzor = Tenzor.subs(
                        H(q, x[1], x[2]) * hyb(p, y) * hyb(k, y),
                        hyb(p, s) * lcs(s, x[1], x[2]) * scalar_result( k*q, "Bfield") / q,
                    )

    # lcs( i, j, l) - Levi-Civita symbol
    Tenzor = Tenzor.subs(lcs(s, indexb, indexB), -lcs(s, indexB, indexb))

    Tenzor = simplify(Tenzor)

    print(f"step 11: {round(time.time() - t, 1)} sec")

    print(f"\nDiagram tensor structure after computing tensor convolutions: \n{Tenzor}")

    return Tenzor
