import time

from Functions.computing_integrals_over_frequencies import *
from Functions.computing_tensor_part import *
from Functions.create_file_with_general_notation import *

# ------------------------------------------------------------------------------------------------------------------#
#                   Part 2. Diagram calculation (integrals over frequencies, tensor convolutions, etc.)
# ------------------------------------------------------------------------------------------------------------------#


def diagram_integrand_calculation(
    output_file_name,
    moznost,
    indexb,
    indexB,
    P_structure,
    H_structure,
    kd_structure,
    hyb_structure,
    Tenzor,
    Product,
    is_diagram_convergent,
):
    """
    This function calculates the integrand of the corresponding diagram in terms of tensor and scalar parts.

    ARGUMENTS:

    output_file_name
    moznost
    indexb, indexB
    P_structure
    H_structure
    kd_structure
    hyb_structure
    Tenzor
    Product
    is_diagram_convergent

    OUTPUT DATA EXAMPLE:
    """

    Feynman_graph = open(f"Results/{output_file_name}", "a+")
    # start filling the results of calculation to file

    Feynman_graph.write(f"\nDiagram integrand calculation begin:\n")
    # starts filling the results of calculations (integrals over frequencies, tensor convolutions) to file

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing integrals over frequencies
    # --------------------------------------------------------------------------------------------------------------#

    total_sum_of_residues_for_both_frequencies = calculating_frequency_integrals_in_two_loop_diagrams(Product, w_k, w_q)
    # calculate integrals over frequencies using the residue theorem

    Feynman_graph.write(
        f"\nThe scalar part of the given diagram integrand (after computing integrals over frequencies): "
        f"\n{total_sum_of_residues_for_both_frequencies} \n"
    )

    diagram_expression = reduction_to_common_denominator(
        total_sum_of_residues_for_both_frequencies, is_diagram_convergent
    )

    diagram_expression_without_prefactor_and_substitution = diagram_expression[0][0]
    diagram_prefactor_without_substitution = diagram_expression[0][1]
    diagram_expression_without_prefactor_and_after_substitution = diagram_expression[1][0]
    diagram_prefactor_after_substitution = diagram_expression[1][1]

    Feynman_graph.write(
        f"\nPrefactor (part of the integrand scalar part numerator, initially frequency-independent): "
        f"\n{diagram_prefactor_without_substitution} \n"
    )

    diagram_sc_part_without_mom_repl = (
        diagram_prefactor_without_substitution * diagram_expression_without_prefactor_and_substitution
    )

    Feynman_graph.write(
        f"\nThe scalar part of the given diagram integrand (with prefactor) after reduction to common denominator: "
        f"\n{diagram_sc_part_without_mom_repl} \n"
    )

    print(f"\nScalar part of the integrand reduced to a common denominator: " f"\n{diagram_sc_part_without_mom_repl}")

    if is_diagram_convergent == True:

        partial_simplification_of_diagram_scalar_part = partial_simplification_of_diagram_expression(
            diagram_expression_without_prefactor_and_after_substitution
        )

        additional_prefactor_simplification = prefactor_simplification(diagram_prefactor_after_substitution)

        part_to_integrand = additional_prefactor_simplification[0]
        field_and_nuo_depend_factor = additional_prefactor_simplification[1]

        Feynman_graph.write(
            f"\nPartially simplified expression for a integrand scalar part without prefactor "
            f"after momentums replacement k, q --> B*k/nuo, B*q/nuo:"
            f"\n{part_to_integrand*partial_simplification_of_diagram_scalar_part} \n"
        )

        Feynman_graph.write(
            f"\nNumerical factor depending on the absolute value of the field and viscosity "
            f"(after replacing of momentums, all entire dependence on |B| and nuo of the diagram is concentrated here)"
            f"\n{field_and_nuo_depend_factor} \n"
        )

        integrand_scalar_part_depending_only_on_uo = (
            (part_to_integrand * partial_simplification_of_diagram_scalar_part).doit().doit()
        )

        if integrand_scalar_part_depending_only_on_uo.has(go):
            sys.exit("Error when getting integrand for numerical integration")
        elif integrand_scalar_part_depending_only_on_uo.has(B):
            sys.exit("Error when getting integrand for numerical integration")
        elif integrand_scalar_part_depending_only_on_uo.has(nuo):
            sys.exit("Error when getting integrand for numerical integration")

        Feynman_graph.write(
            f"\nCompletely simplified expression for a integrand scalar part without prefactor "
            f"after momentums replacement k, q --> B*k/nuo, B*q/nuo:"
            f"\n{integrand_scalar_part_depending_only_on_uo} \n"
        )

    else:
        integrand_scalar_part_depending_only_on_uo = None
        field_and_nuo_depend_factor = None

    # --------------------------------------------------------------------------------------------------------------#
    #                                        Сomputing diagram tensor structure
    # --------------------------------------------------------------------------------------------------------------#

    print(f"\nBeginning the tensor convolutions calculation: \n")

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

    structurep = list()
    structureh = list()

    for in2 in kd_structure:
        structurep = list()
        for in1 in P_structure:
            # calculation via Kronecker's delta function: P(k, i, j) kd(i, l) = P(k, l, j)
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
                # del kd_structure[0]
                # it deletes the kronecker delta from the list if it is no longer in the tensor structure
                break
        # it adds all newly created structures to the list
        P_structure = P_structure + structurep

        for in1 in H_structure:
            # calculation via Kronecker's delta function: H(k, i, j) kd(i, l) = H(k, l, j)
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
            Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[2]), 0)
        if Tenzor.coeff(P(in1[0], in1[1], in1[2])) == 0:
            P_structure.remove(in1)
        else:
            if in1[0] == -k or in1[0] == -q:
                # Replace in the tensor structure in the projection operators:  P(-k,i,j) = P(k,i,j)
                Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]), P(-in1[0], in1[1], in1[2]))
                P_structure[i][0] = -in1[0]
            i += 1

    i = 0
    # discard from the Tensor structure what is zero for the helical operator H_ij (k) * k_i = 0
    while i < len(H_structure):
        in1 = H_structure[i]
        if Tenzor.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[1]), 0)
        elif Tenzor.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        elif Tenzor.coeff(hyb(-in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(-in1[0], in1[2]), 0)
        if Tenzor.coeff(H(in1[0], in1[1], in1[2])) == 0:
            H_structure.remove(in1)
        else:
            i += 1

    print(f"step 2: {round(time.time() - t, 1)} sec")

    i = 0
    # sipmplify in the Tenzor part H_{ij} (k) P_{il} (k) =  H_{il} (k)
    while len(H_structure) > i:
        in1 = H_structure[i]
        for in2 in P_structure:
            if in1[0] == in2[0] and Tenzor.coeff(H(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2])) != 0:
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
    while len(P_structure) > i:
        in1 = P_structure[i]
        structurep = list()
        for j in range(i + 1, len(P_structure)):
            in2 = P_structure[j]
            if in1[0] == in2[0] and Tenzor.coeff(P(in1[0], in1[1], in1[2]) * P(in2[0], in2[1], in2[2])) != 0:
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
        P_structure = P_structure + structurep
        # it add all newly created structures to the list

    print(f"step 4: {round(time.time() - t, 1)} sec")

    for i in hyb_structure:  # replace: hyb(-k+q, i) = -hyb(k, i) + hyb(q, i)
        k_c = i[0].coeff(k)
        q_c = i[0].coeff(q)
        if k_c != 0 or q_c != 0:
            Tenzor = Tenzor.subs(hyb(i[0], i[1]), (k_c * hyb(k, i[1]) + q_c * hyb(q, i[1])))

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

    # discard from the Tensor structure what is zero for the helical operator H_{ij} (k) * k_i = 0
    for in1 in H_structure:
        clen = Tenzor.coeff(H(in1[0], in1[1], in1[2]))
        if clen.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        if clen.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        if in1[0] == k and clen.coeff(hyb(q, in1[1]) * hyb(q, in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(q, in1[1]) * hyb(q, in1[2]), 0)
        if in1[0] == q and clen.coeff(hyb(k, in1[1]) * hyb(k, in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(k, in1[1]) * hyb(k, in1[2]), 0)

    print(f"step 6: {round(time.time() - t, 1)} sec")

    inkd = 0
    while inkd == 0:
        # calculation part connected with the kronecker delta function: kd(i,j)*hyb(k,i) = hyb(k,j)
        for in1 in kd_structure:
            # beware, I not treat the case if there remains a delta function with indexes of external fields !!
            clen = Tenzor.coeff(kd(in1[0], in1[1]))
            if clen.coeff(hyb(k, in1[0])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(k, in1[0]), hyb(k, in1[1]))
            if clen.coeff(hyb(k, in1[1])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(k, in1[1]), hyb(k, in1[0]))
            if clen.coeff(hyb(q, in1[0])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(q, in1[0]), hyb(q, in1[1]))
            if clen.coeff(hyb(q, in1[1])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(q, in1[1]), hyb(q, in1[0]))
            if clen.coeff(hyb(p, in1[0])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(p, in1[0]), hyb(p, in1[1]))
            if clen.coeff(hyb(p, in1[1])) != 0:
                Tenzor = Tenzor.subs(kd(in1[0], in1[1]) * hyb(p, in1[1]), hyb(p, in1[0]))
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
        if clen.coeff(hyb(in1[0], in1[1])) != 0:
            # I throw out the part:  H (k,i,j) hyb(k,i) = 0
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        if clen.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        if in1[0] == k and clen.coeff(hyb(q, in1[1]) * hyb(q, in1[2])) != 0:
            # I throw out the part:  H (k,i,j) hyb(q,i) hyb(q, j) = 0
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(q, in1[1]) * hyb(q, in1[2]), 0)
        if in1[0] == q and clen.coeff(hyb(k, in1[1]) * hyb(k, in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(k, in1[1]) * hyb(k, in1[2]), 0)
        for in2 in kd_structure:
            # it puts together the Kronecker delta and the helical term: H(k,i,j)*kd(i,l) = H(k,l,j)
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

    Feynman_graph.write(f"\nDiagram tensor structure after computing tensor convolutions: \n{Tenzor} \n")

    # result = str(Tenzor)
    # result = result.replace("**", "^")

    # finish  filling the results of calculation to file
    Feynman_graph.write(f"\nDiagram integrand calculation end.\n")

    # print(pretty(Tenzor, use_unicode=False))

    # Fey_graphs.write("\n"+ pretty(Tenzor, use_unicode=False) + "\n")

    # Fey_graphs.write("\n"+ latex(Tenzor) + "\n")

    Feynman_graph.close()

    return [
        diagram_sc_part_without_mom_repl,
        integrand_scalar_part_depending_only_on_uo,
        field_and_nuo_depend_factor,
        Tenzor,
    ]


def transform_to_Wolfram_Mathematica_format(input_string_expression):
    """ """

    integrand_str = str(input_string_expression).replace("**", "^")

    while integrand_str.find("sqrt(") != -1:

        index_sqrt = integrand_str.find("sqrt(")

        integrand_before_sqrt = integrand_str[:index_sqrt]
        sqrt_shift = len("sqrt(")
        integrand_after_sqrt = integrand_str[index_sqrt + sqrt_shift :]

        index_bracket = integrand_after_sqrt.find(")")

        symbol_after_bracket = integrand_after_sqrt[index_bracket + 1]

        sqrt_possible_pattern1 = "q^4*(uo - 1)^2 - 4*q^2*z_q^2"
        len_sqrt_possible_pattern1 = len(sqrt_possible_pattern1)
        sqrt_possible_pattern2 = "k^4*(uo - 1)^2 - 4*k^2*z_k^2"
        len_sqrt_possible_pattern2 = len(sqrt_possible_pattern2)
        sqrt_possible_pattern3 = "(uo - 1)^2*(k^2 + 2*k*q*z + q^2)^2 - 4*(k*z_k + q*z_q)^2"
        len_sqrt_possible_pattern3 = len(sqrt_possible_pattern3)
        if integrand_after_sqrt[:len_sqrt_possible_pattern1] == sqrt_possible_pattern1:
            under_sqrt_expression = integrand_after_sqrt[:len_sqrt_possible_pattern1]
            integrand_after_sqrt_after_under_root_expr = integrand_after_sqrt[len_sqrt_possible_pattern1 + 1 :]
        elif integrand_after_sqrt[:len_sqrt_possible_pattern2] == sqrt_possible_pattern2:
            under_sqrt_expression = integrand_after_sqrt[:len_sqrt_possible_pattern2]
            integrand_after_sqrt_after_under_root_expr = integrand_after_sqrt[len_sqrt_possible_pattern2 + 1 :]
        elif integrand_after_sqrt[:len_sqrt_possible_pattern3] == sqrt_possible_pattern3:
            under_sqrt_expression = integrand_after_sqrt[:len_sqrt_possible_pattern3]
            integrand_after_sqrt_after_under_root_expr = integrand_after_sqrt[len_sqrt_possible_pattern3 + 1 :]
        else:
            if symbol_after_bracket == "^":
                right_index = integrand_after_sqrt.find(")^2)")
                length = len(")^2")
                under_sqrt_expression = integrand_after_sqrt[: right_index + length]
                integrand_after_sqrt_after_under_root_expr = integrand_after_sqrt[right_index + length + 1 :]
            else:
                under_sqrt_expression = integrand_after_sqrt[:index_bracket]
                integrand_after_sqrt_after_under_root_expr = integrand_after_sqrt[index_bracket + 1 :]

        integrand_after_replacement = (
            integrand_before_sqrt
            + str("Sqrt[")
            + under_sqrt_expression
            + str("]")
            + integrand_after_sqrt_after_under_root_expr
        )

        integrand_str = integrand_after_replacement

        cos_k_str = "Cos[ToExpression[SubscriptBox[\[Theta], 1]]]"
        cos_q_str = "Cos[ToExpression[SubscriptBox[\[Theta], 2]]]"
        cos_2u_str = "Cos[2*u]"
        sin_k_str = "Sin[ToExpression[SubscriptBox[\[Theta], 1]]]"
        sin_q_str = "Sin[ToExpression[SubscriptBox[\[Theta], 2]]]"

        integrand_str_final = (
            integrand_str.replace("uo", "ToExpression[SubscriptBox[u, 0]]")
            .replace("z_k", cos_k_str)
            .replace("z_q", cos_q_str)
            .replace("z", sin_q_str + "*" + sin_k_str + "*" + cos_2u_str + " + " + cos_k_str + "*" + cos_q_str)
        )

    return integrand_str_final


def preparing_diagram_for_numerical_integration(
    output_file_name,
    Tensor,
    eps_default,
    d_default,
    A_MHD,
    list_with_uo_values,
    field_and_nuo_depend_factor,
    integrand_scalar_part_depending_only_on_uo,
):
    """
    The function finally prepares the integrand for numerical integration. Works only with converging diagrams.

    ARGUMENTS:

    output_file_name is given by get_info_about_diagram(),
    Tensor, field_and_nuo_depend_factor, and  integrand_scalar_part_depending_only_on_uo are given by
    diagram_integrand_calculation()

    eps_default -- value of the eps regularization parameter (default is assumed to be 0.5),
    d_default -- coordinate space dimension (assumed to be 3 by default)
    A_MHD -- value of the model parameter A (MHD corresponds to A = 1)
    list_with_uo_values -- the desired values of the unrenormalized magnetic Prandtl number uo
    introduced before the calculations

    OUTPUT DATA EXAMPLE:

    field_and_nuo_depend_factor = 4*I*pi**2*go**2*nuo**11*rho*lcs(s, 0, 9)*hyb(p, s)/B**10

    list_with_integrands = too long
    """

    tensor_part_numerator = fraction(Tensor)[0]
    tensor_part_denominator = fraction(Tensor)[1]
    scalar_part_numerator = fraction(integrand_scalar_part_depending_only_on_uo)[0]
    scalar_part_denominator = fraction(integrand_scalar_part_depending_only_on_uo)[1]

    tensor_part_numerator_args = tensor_part_numerator.args

    for i in range(len(tensor_part_numerator_args)):
        numerator_term = tensor_part_numerator_args[i]
        if numerator_term.has(I) or numerator_term.has(rho):
            field_and_nuo_depend_factor *= numerator_term
        elif numerator_term.has(hyb) or numerator_term.has(lcs):
            field_and_nuo_depend_factor *= numerator_term
        else:
            scalar_part_numerator *= numerator_term

    integrand_denominator = scalar_part_denominator * tensor_part_denominator
    integrand_numerator = scalar_part_numerator

    integrand = integrand_numerator / integrand_denominator

    intgrand_for_numerical_calc = integrand.subs(d, d_default).subs(eps, eps_default).subs(A, A_MHD)

    field_and_nuo_depend_factor = field_and_nuo_depend_factor.subs(d, d_default).subs(eps, eps_default).subs(A, A_MHD)

    WF_intgrand_for_numerical_calc = transform_to_Wolfram_Mathematica_format(intgrand_for_numerical_calc)

    Feynman_graph = open(f"Results/{output_file_name}", "a+")

    Feynman_graph.write(f"\nDefault parameter values: d = {d_default}, eps = {eps_default}, and A = {A_MHD}.\n")

    Feynman_graph.write(f"\nDiagram integrand with default parameters: \n{intgrand_for_numerical_calc}\n")

    Feynman_graph.write(f"\nDiagram integrand in Wolfram Mathematica format: \n{WF_intgrand_for_numerical_calc}\n")

    list_with_integrands = list()

    if list_with_uo_values[0] != "uo":

        Feynman_graph.write(f"\nThe entered values of the bare magnetic Prandtl number uo: {list_with_uo_values}")

        Feynman_graph.write(f"\nBelow are the integrands corresponding to the entered values of uo.\n")

        for i in range(len(list_with_uo_values)):
            uo_value = list_with_uo_values[i]
            list_with_integrands.append(intgrand_for_numerical_calc.subs(uo, uo_value))
            integrand = list_with_integrands[i]
            integrand_in_Wolfram_format = transform_to_Wolfram_Mathematica_format(integrand)
            # Feynman_graph.write(f"\nDiagram integrand for uo = {uo_value}: \n{integrand} \n")
            Feynman_graph.write(
                f"\nDiagram integrand for uo = {uo_value} in Wolfram Mathematica format: "
                f"\n{integrand_in_Wolfram_format} \n"
            )

    Feynman_graph.close()

    return field_and_nuo_depend_factor, list_with_integrands
