#!/usr/bin/python3

import sys
import itertools
from sympy import *
from functools import reduce
import time


def get_output_file_name(graf):
    # topological part of the Nickel index
    Nickel_topology = " ".join(graf.split(sep=":")[0].split(sep="|"))

    # line structure in the diagram corresponding to Nickel_topology
    Nickel_lines = " ".join(graf.split(sep=":")[1].split(sep="|"))

    return f"Diagram_{Nickel_topology.strip()}{Nickel_lines.strip()}.txt"


# ---------------------------------------------------------------------------------------------------------------------------
#                   Auxiliary functions
# ---------------------------------------------------------------------------------------------------------------------------
"""
ну вот челик знает про функции. но культуры того, как выглядят большие проекты нет. 
надо чаще смотреть чужой код на гитхабе было
"""
def propagator(
    nickel,
):  # arranges the propagators into a list of inner and outer lines with fields
    s1 = 0
    s2 = nickel.find(":")
    propi = []
    prope = []
    for i in nickel[: nickel.find(":")]:
        if i == "e":
            prope += [[(-1, s1), ["0", nickel[s2 + 2]]]]
            s2 += 3
        elif i != "|":
            propi += [[(s1, int(i)), [nickel[s2 + 1], nickel[s2 + 2]]]]
            s2 += 3
        else:
            s1 += 1
    return [propi, prope]

def compare_fun(
    tup1, tup2
):  # it gives the one that is bigger if the smaller one is found in it, or 0 if it is not found
    g = len(set(tup1).intersection(tup2))
    if g == len(tup1):
        return tup2
    elif g == len(tup2):
        return tup1
    else:
        return 0

def dosad(zoznam, ind_hod, struktura, pozicia):
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



def main():
    # ---------------------------------------------------------------------------------------------------------------------------
    #                   Creating a file with the results for the corresponding diagram
    # ---------------------------------------------------------------------------------------------------------------------------

    # All diagrams is completely defined by the Nickel index

    # Examples:
    # graf1 = str("e12|e3|33||:0B_bB_vv|0b_vV|Vv_vv||")
    # graf2 = str("e12|23|3|e|:0B_bb_vB|bb_Vb|vV|0b|")
    # graf3 = str("e12|e3|33||:0B_vB_bb|0b_vV|Bb_vb||")

    """
    Плохая практика. Вообще реально удобно через инпут делать?
    Обычно создают конфигурационные файлы, типа
    parametrs.init - это по сути обычный txt, где будут записаны все важные штуки
    там прописать никелевый индекс, и каждый раз оттуда считывать.

    либо уж как аргумент в консоли. 
    """
    # graf = input("Enter Nickel index:")
    graf = str("e12|e3|33||:0B_bB_vv|0b_vV|Vv_vv||")

    output_file_name = get_output_file_name(graf)

    Fey_graphs = open(output_file_name, "w")

    # Windows OS does not understand the symbols ":" and "|" in the file name
    # File name example: "Diagram e12 e3 33  0B_bB_vv 0b_vV Vv_vv.txt" (all "|" are replaced by a space, ":" is removed)

    Fey_graphs.write(f"Nickel index of feynman diagram:  {graf} \n")


    """
    схема должна быть такая. ты вообще всё должен оформить в функции. все блоки кода связанные
    какой-то общей логикой. какие-то операции которые повторяются и т.д.

    а в итоге у тебя программа должны выглядеть как последовательность вызовов функций.

    вот тут например функция создаёт файл и записывает туда название графа, который ты передал
    ей на входе. 

    ну вот я и оформляют этот кусок кода как функцию, потом вызываю. И всё. Если функция 
    правильная и я это проверил. То мне потом уже можно разгрузить свою оперативку в мозгу 
    и не думать про этот кусок. А то когда это выглядит как перечисление строчек кода,
    мозг каждый раз пытается понять что это за хуйня. А так по названию функции можно врубиться
    что происходит. 

    И на первом этапе оживления этого говна нужно вот так пройтись со всем. 

    Функции могут вызывать другие функции. Так что можно создавать маленькие логические блоки
    и потом их объединять в большие.
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    #                   Global variables
    # ---------------------------------------------------------------------------------------------------------------------------
    """
    с этим пока что хуй знает как поступить. но первая мысля. может пускай реально остаётся,
    если это глобальные константы, которые потом везде юзаются. Хорошо, что это выделено в 
    отдельный блок, который можно пропускать и не переживать о нём. 
    """
    # Throughout the text p always denotes an external momentum

    stupen = 1  # proportionality of the tensor structure to the external momentum p

    number_int_vert = 4  # the number of internal (three-point) vertecies


    hyb = [p, k, q] = symbols("p k q")  # symbols for momentums (p denotes an external momentum)
    P = Function("P")  # Transverse projection operator
    H = Function("H")  # Helical term
    kd = Function("kd")  # Kronecker delta function
    hyb = Function("hyb")  # "momentums" defines momentum as follows: momentums(k, 1) is $k_1$
    lcs = Function("lcs")  # Levi-Civita symbol


    [I, A, z, nu, vo, uo, rho] = symbols("I A z nu vo uo rho")

    # I is an imaginary unit
    # A is a model parameter (the model action includes a term ~ A Bbv ).
    # z ????
    # nu ???
    # v_0 ???
    # u_0^(-1) is a "magnetic Prandtl number"
    # rho is a gyrotropy parameter, |rho| < 1

    [go, d, eps] = symbols("go d eps")  # coupling constant, dimenzion

    # g_0 is a bare coupling constant
    # d is a space dimension
    # eps determines a degree of model deviation from logarithmicity

    [s, b, B] = symbols("s b B")

    # the index of field connected to field and the external momentum p_s ???



    # ------------------------------------------------------------------------------------
    #      The part to solve the momentum in diagrams
    # -------------------------------------------------------------------------------------
    linie = propagator(graf)
    vnutorne = linie[0]  # notation of internal line in diagram
    vonkajsie = linie[1]  # notation of external line in diagram

    '''
    Если в поиск вбить эти интёрнал_лайнс, то они живут до примено 356 строчки
    стало быть это блок кода изолированный. пускай и длиной в 200 строчек. 
    возможно его выделить в какую-то функцию
    '''


    internal_lines = dict()
    for x in range(len(vnutorne)):
        internal_lines.update(
            {x: vnutorne[x]}
        )  # dict - the keys ara the digits of the lines

  
    Fey_graphs.write(f"\n Marking the lines in the diagram: {internal_lines} \n")

    hybnost = dict()
    propagator_hel = [["v", "v"], ["v", "b"], ["b", "v"], ["b", "b"]]
    for (i) in (internal_lines):  # assigning momentum (k or q) to the propagator that contains kernel D_v - it defines the loop of the diagram later
        linia = internal_lines[i]
        if linia[1] in propagator_hel:
            if len(hybnost) == 0:
                hybnost.update({i: k})
            elif len(hybnost) == 1:
                hybnost.update({i: q})
            else:
                hybnost.update({i: "None"})

    loop = list()
    for i in range(len(internal_lines) - 1):
        pomoc = list(itertools.combinations(internal_lines.keys(), r=i + 2))
        [
            loop.append(x) for x in pomoc
        ]  # All possible combinations of propagators for any loops

    i = 0
    while i < len(loop):  # Check if the given lines (propagators) combination is a loop
        slucka = [
            vnutorne[in1][0] for in1 in loop[i]
        ]  # prida propagator zodpovedajuci tuple
        hodnoty = list(
            itertools.chain.from_iterable(slucka)
        )  # prepise dvojice do list vertexov
        sucet = list(
            map(lambda x: hodnoty.count(x), range(max(hodnoty) + 1))
        )  # srata kolko krat sa dany vertex nachadza v slucke
        if sucet.count(2) + sucet.count(0) < len(sucet):
            del loop[i]
        else:
            i += 1  # loop - I have all loops - one time

    loop_pomoc = list(itertools.combinations(loop, r=2))  # made for the two-loop case
    i = 0
    loop_pomoc = sorted(loop_pomoc, key=len)
    while i < len(loop_pomoc):  # the result is a combination of loops that covers the entire graph -- the list of selected
        b = list(itertools.chain.from_iterable(loop_pomoc[i]))
        pocet = compare_fun(b, internal_lines.keys())
        if pocet == 0:
            del loop_pomoc[i]
        else:
            i += 1

    loop = loop_pomoc
    i = 0
    while i < len(loop):  # check momentums in loops - first loop = k and second loop = q
        slucka1 = loop[i][0]
        sucet = list(map(lambda x: slucka1.count(x), hybnost))
        if sucet.count(1) != 1:
            del loop[i]
        else:
            slucka2 = loop[i][1]
            sucet = list(map(lambda x: slucka2.count(x), hybnost))
            if sucet.count(1) != 1:
                del loop[i]
            else:
                i += 1

    try:  # vyhodi ak nemam moznost priradit hybnosti
        loop = loop[0]
    except IndexError as err:
        print("Neexistuje moznost ako usporiadat hybnosti na rozne slucky", err)

    Fey_graphs.write(
        "\n"
        + "Loops in the diagram for a given internal momentum (digit coresponds the line): "
        + str(loop)
        + "\n"
    )

    # The beginning of the momentum distribution. In this case, momentum flows into the diagram via field B and flows out through field b.
    # If the momentum flows into the vertex is with (+) if the outflow is with (-).
    # In propagator the line follows nickel index. For example line (1,2) is with (+) momentum and line (2,1) is with (-) momentum - starting point is propagators vv or propagators with kernel D_v

    hybnost_pomoc = [0] * len(internal_lines)
    for i in loop:
        prop = list(map(lambda x: internal_lines[x][0], range(len(hybnost_pomoc))))
        prop = reduce(lambda x, y: list(x) + list(y), prop)
        for j in range(len(internal_lines)):
            if j not in i:
                prop[2 * j] = -1
                prop[2 * j + 1] = -1
        for j in hybnost:
            if j in i:
                zaciatok = j  # finds the beginning of the loop where there is kernel D_v with k or q
                hybnost_pomoc[j] += hybnost[j]  # put momentum k or q
                hodnota = prop[2 * zaciatok + 1]
                prop[2 * zaciatok + 1] = -1
                prop[2 * zaciatok] = -1
                while prop.count(-1) < len(
                    prop
                ):  # I go through the loop, I follow the propagator with D_v from vertex with smaller number to bigger according to the indexes, if it fits, it goes (+), the opposite (-)
                    poradie = prop.index(
                        hodnota
                    )  # I write the momentum into the propagator in the tuple
                    if poradie % 2 == 0:
                        zaciatok = int(poradie / 2)
                        hybnost_pomoc[zaciatok] = hybnost_pomoc[zaciatok] + hybnost[j]
                        hodnota = prop[poradie + 1]
                    else:
                        zaciatok = int((poradie - 1) / 2)
                        hybnost_pomoc[zaciatok] = hybnost_pomoc[zaciatok] - hybnost[j]
                        hodnota = prop[poradie - 1]
                    prop[2 * zaciatok + 1] = -1
                    prop[2 * zaciatok] = -1

    for i in range(len(hybnost_pomoc)):  # write the momentum as they stand on the lines
        if i not in hybnost:
            hybnost.update({i: hybnost_pomoc[i]})

    Fey_graphs.write(
        "\n" + "Momentum distributed among propagators (lines): " + str(hybnost) + "\n"
    )

    moznost = [0] * (number_int_vert * 3)
    for i in vonkajsie:  # deploy external momentum
        vrchol = i[0][1]
        if i[1][1] == "B":
            moznost[3 * vrchol] = [-1, "B", p]
            indexB = 3 * vrchol  # save the index of the external field b'
        else:
            moznost[3 * vrchol] = [-1, i[1][1], -p]
            indexb = 3 * vrchol  # save the index of the outer field b

    for (i) in (internal_lines):  # distributes momentum on the fields for a given vertex - (+) for the momentum that flows into the vertex and (-) for the momentum that  flows from the vertex
        linia = internal_lines[i]
        if moznost[linia[0][0] * 3] == 0:
            moznost[linia[0][0] * 3] = [i, linia[1][0], -hybnost[i]]
        elif moznost[linia[0][0] * 3 + 1] == 0:
            moznost[linia[0][0] * 3 + 1] = [i, linia[1][0], -hybnost[i]]
        else:
            moznost[linia[0][0] * 3 + 2] = [i, linia[1][0], -hybnost[i]]
        if moznost[linia[0][1] * 3] == 0:
            moznost[linia[0][1] * 3] = [i, linia[1][1], hybnost[i]]
        elif moznost[linia[0][1] * 3 + 1] == 0:
            moznost[linia[0][1] * 3 + 1] = [i, linia[1][1], hybnost[i]]
        else:
            moznost[linia[0][1] * 3 + 2] = [i, linia[1][1], hybnost[i]]

    Fey_graphs.write(
        "\n" + "Momentum distribution at the vertices: " + str(moznost) + "\n"
    )  # [[index of propagator, field, momentum]]
    # --------------------------------------------------------------------------------------
    # The previous part is the prepartion for the writing the structure from the diagram.

    Tenzor = 1  # writing the tensor structure - this is the quantity where the tensor structure is constructed

    indexy = list(map(lambda x: x[0], moznost))
    P_structure = (
        []
    )  # I save the structures so that I don't have to guess through all possible combinations (faster running of the program) [ [momentum, index 1, index 2]]
    H_structure = (
        []
    )  # I save the helical structures so that I don't have to guess through all possible combinations (faster running of the program) [ [momentum, index 1, index 2]]

    integral = ""
    for (i) in (internal_lines):  # I write into the Tenzor, the projection part of the propagators (vv, Vv = v'v, Bb = b'b )
        linia = internal_lines[i]
        in1 = indexy.index(i)
        indexy[in1] = len(internal_lines)
        in2 = indexy.index(i)
        '''
        если быть совсем модным и молодёжным, то это надо переписать на
        pattern matching, но это не самая большая проблема этой программы,
        это в самую последнюю очередь
        '''
        if linia[1] == ["v", "v"]:
            Tenzor = Tenzor * (
                P(hybnost[i], in1, in2) + I * rho * H(hybnost[i], in1, in2)
            )  # H_{ij} (k) = \epsilon_{i, j, l} k_l/ k = H(k, i, j) - it is part with levi-civita symbol and momentum
            H_structure.append([hybnost[i], in1, in2])
            integral = integral + "Pvv[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["v", "V"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2))  # P_{i, j} (k) = P(k, i, j)
            integral = integral + "PvV[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["V", "v"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2))
            integral = integral + "PVv[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["b", "B"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2))
            integral = integral + "PbB[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["B", "b"]:
            Tenzor = Tenzor * (
                P(hybnost[i], in1, in2)
            )  # Here I can define new propagator for tenzor structure: vb, bv, ...
            integral = integral + "PBb[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["v", "b"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2) + I * rho * H(hybnost[i], in1, in2))
            H_structure.append([hybnost[i], in1, in2])
            integral = integral + "Pvb[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["b", "v"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2) + I * rho * H(hybnost[i], in1, in2))
            H_structure.append([hybnost[i], in1, in2])
            integral = integral + "Pbv[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["b", "b"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2) + I * rho * H(hybnost[i], in1, in2))
            H_structure.append([hybnost[i], in1, in2])
            integral = integral + "Pbb[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["V", "b"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2))
            integral = integral + "PVb[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["b", "V"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2))
            integral = integral + "PbV[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["B", "v"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2))
            integral = integral + "PBv[" + str(hybnost[i]) + "]*"
        elif linia[1] == ["v", "B"]:
            Tenzor = Tenzor * (P(hybnost[i], in1, in2))
            integral = integral + "PvB[" + str(hybnost[i]) + "]*"
        P_structure.append([hybnost[i], in1, in2])


    '''
    все вот эти выводы сделаны в старом, уродливом и медленно работающем стиле
    надо переписывать на f-строки, ну вот тут это должно выглядеть так:

    Fey_graphs.write(
        f"\n Momentum in propagators for the Wolfram Mathematica file: {integral[:-1]} \n"
    )
    '''
    Fey_graphs.write(
        "\n"
        + "Momentum in propagators for the Wolfram Mathematica file: "
        + str(integral[:-1])
        + "\n"
    )


    kd_structure = (
        []
    )  # I save the kronecker delta so that I don't have to guess through all possible combinations (faster running of the program) [ [index 1, index 2]] ... kd(index 1, index 2)
    hyb_structure = (
        []
    )  # I save the momemntum and their index (faster running of the program) [ [ k, i] ] ... k_i = hyb(k, i)
    polia = list(
        map(lambda x: x[1], moznost)
    )  # polia - the ordered list: [ V, v, v, b, V, v,... ] - the first three fields corespond the 0 vertex
    for i in range(number_int_vert):  # part to add vertices - all vertieces have
        vrchol = polia[3 * i : 3 * (i + 1)]  # triple for vertex
        vrchol_usp = sorted(vrchol)
        if vrchol_usp == [
            "B",
            "b",
            "v",
        ]:  # it finds the indexes of fields and add  to the Tenzor
            in1 = 3 * i + vrchol.index("B")
            in2 = 3 * i + vrchol.index("b")
            in3 = 3 * i + vrchol.index("v")
            Bbv = I * (
                hyb(moznost[in1][2], in3) * kd(in1, in2)
                - A * hyb(moznost[in1][2], in2) * kd(in1, in3)
            )
            Tenzor = Tenzor * Bbv
            kd_structure.append([in1, in2])  # it remembers structures for kronecker delta
            kd_structure.append([in1, in3])
            hyb_structure.append(
                [moznost[in1][2], in3]
            )  # it remembers structures for momentum
            hyb_structure.append([moznost[in1][2], in2])
        elif vrchol_usp == ["V", "v", "v"]:
            in1 = 3 * i + vrchol.index("V")
            in2 = [3 * i, 3 * i + 1, 3 * i + 2]
            in2.remove(in1)
            Vvv = I * (
                hyb(moznost[in1][2], in2[0]) * kd(in1, in2[1])
                + hyb(moznost[in1][2], in2[1]) * kd(in1, in2[0])
            )
            Tenzor = Tenzor * Vvv
            kd_structure.append([in1, in2[1]])
            kd_structure.append([in1, in2[0]])
            hyb_structure.append([moznost[in1][2], in2[0]])
            hyb_structure.append([moznost[in1][2], in2[1]])
        elif vrchol_usp == ["V", "b", "b"]:
            in1 = 3 * i + vrchol.index("V")
            in2 = [3 * i, 3 * i + 1, 3 * i + 2]
            in2.remove(in1)
            Vvv = I * (
                hyb(moznost[in1][2], in2[0]) * kd(in1, in2[1])
                + hyb(moznost[in1][2], in2[1]) * kd(in1, in2[0])
            )
            Tenzor = Tenzor * Vvv
            kd_structure.append([in1, in2[1]])
            kd_structure.append([in1, in2[0]])
            hyb_structure.append([moznost[in1][2], in2[0]])
            hyb_structure.append([moznost[in1][2], in2[1]])

    # ----------------------------------------------------------------------------------------------
    # The program start here. The previous part is only for the reason that I don't have to write the whole structure from the diagram.
    t = time.time()  # it is only used to calculate the calculation time -- can be omitted

    Fey_graphs.write(
        "\n"
        + "The tensor structure of the diagram after calculation: "
        + str(Tenzor)
        + "\n"
    )

    print(Tenzor, "\n")

    Tenzor = expand(Tenzor)  # The final tesor structure from the diagram.

    Tenzor = rho * Tenzor.coeff(rho**stupen)  # What I need from the Tenzor structure
    Tenzor = expand(Tenzor.subs(I**5, I))  # calculate the imaginary unit
    # Tenzor = Tenzor.subs(A, 1)              # It depends on which part we want to calculate from the vertex Bbv
    # print(Tenzor)

    print("step 0:", time.time() - t)


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


    print("step 1:", time.time() - t)

    i = 0
    while i < len(P_structure):  # discard from the Tensor structure what is zero for the projection operator P_ij (k) * k_i = 0
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
            if (
                in1[0] == -k or in1[0] == -q
            ):  # Replace in the tensor structure in the projection operators:  P(-k,i,j) = P(k,i,j)
                Tenzor = Tenzor.subs(P(in1[0], in1[1], in1[2]), P(-in1[0], in1[1], in1[2]))
                P_structure[i][0] = -in1[0]
            i += 1

    i = 0
    while i < len(H_structure):  # discard from the Tensor structure what is zero for the helical operator H_ij (k) * k_i = 0
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

    print("step 2:", time.time() - t)

    i = 0
    while (len(H_structure) > i):  # sipmplify in the Tenzor part H_{ij} (k) P_{il} (k) =  H_{il} (k)
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

    print("step 3:", time.time() - t)

    i = 0
    while (len(P_structure) > i):  # sipmplify in the Tenzor part  P_{ij} (k) P_{il} (k) =  P_{il} (k)
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

    print("step 4:", time.time() - t)

    for i in hyb_structure:  # replace: hyb(-k+q, i) = -hyb(k, i) + hyb(q, i)
        k_c = i[0].coeff(k)
        q_c = i[0].coeff(q)
        if k_c != 0 or q_c != 0:
            Tenzor = Tenzor.subs(hyb(i[0], i[1]), (k_c * hyb(k, i[1]) + q_c * hyb(q, i[1])))


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

    print("step 5:", time.time() - t)

    Tenzor = expand(Tenzor)

    for (in1) in (H_structure):  # discard from the Tensor structure what is zero for the helical operator H_{ij} (k) * k_i = 0
        clen = Tenzor.coeff(H(in1[0], in1[1], in1[2]))
        if clen.coeff(hyb(in1[0], in1[1])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        if clen.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
        if in1[0] == k and clen.coeff(hyb(q, in1[1]) * hyb(q, in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(q, in1[1]) * hyb(q, in1[2]), 0
            )
        if in1[0] == q and clen.coeff(hyb(k, in1[1]) * hyb(k, in1[2])) != 0:
            Tenzor = Tenzor.subs(
                H(in1[0], in1[1], in1[2]) * hyb(k, in1[1]) * hyb(k, in1[2]), 0
            )

    print("step 6:", time.time() - t)

    inkd = 0
    while (inkd == 0):  # calculation part connected with the kronecker delta function: kd(i,j) *hyb(k,i) = hyb(k,j)
        for (
            in1
        ) in (
            kd_structure
        ):  # beware, I not treat the case if there remains a delta function with indexes of external fields !!
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

    print("step 7:", time.time() - t)

    i = 0
    while len(H_structure) > i:  # calculation for helical term
        in1 = H_structure[i]
        clen = Tenzor.coeff(H(in1[0], in1[1], in1[2]))
        if (
            clen.coeff(hyb(in1[0], in1[1])) != 0
        ):  # I throw out the part:  H (k,i,j) hyb(k,i) = 0
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[1]), 0)
        if clen.coeff(hyb(in1[0], in1[2])) != 0:
            Tenzor = Tenzor.subs(H(in1[0], in1[1], in1[2]) * hyb(in1[0], in1[2]), 0)
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

    print("step 8:", time.time() - t)

    p_structure = list()  # list of indeces for momentum p in Tenzor
    k_structure = list()  # list of indeces for momentum k in Tenzor
    q_structure = list()  # list of indeces for momentum q in Tenzor
    for in1 in range(len(moznost)):  # It combines quantities with matching indices.
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


    print("step 9:", time.time() - t)

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
        )  #  I know which indexes are free. I know where the fields B or b are located.
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


    print("step 10:", time.time() - t)

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

    print("step 11:", time.time() - t)

    result = str(Tenzor)
    result = result.replace("**", "^")
    Fey_graphs.write(
        "\n" + "The tensor structure of the diagram after calculation: " + "\n"
    )
    Fey_graphs.write("\n" + result + "\n")

    print("The tensor structure of the diagram after calculation:", Tenzor)

    # print(pretty(Tenzor, use_unicode=False))

    # Fey_graphs.write("\n"+ pretty(Tenzor, use_unicode=False) + "\n")

    # Fey_graphs.write("\n"+ latex(Tenzor) + "\n")

    Fey_graphs.close()

'''
эта штука называется точкой входа. по сути то место, с которого
начнётся выполнение программы. Грубо говоря как функция main в Си

Простой блок, который вызывает только одну главную функцию. 
А эта функция уже содержит в себе всё!

По сути теперь программа выглядит как:

функция1
функция2
функция3
...

функция main()
    внутри комбинация всех фукнций выше


вызов функции main

---

Значит в коде нет НИ ОДНОЙ строчки кода, которая не была бы
завёрнута в функцию. Это уже правильный стиль написания программ. 

Теперь твоя программа перестала быть просто простынёй перечисляющей всё что нужно сделать
и стала правильно оформленной программой с очень большой функцией main, которая содержит
эту простыню. 

Для примера я выделил тебе один блок. В самом начале челик из входных данных
из вида графа пытается сгенерировать название выходного файла. Я выделил эту операцию
в функцию get_output_file_name

вот тебе этим и нужно заниматься. читаешь мэйн, видишь какой-то логически связанный
и законченный блок и выделяешь его в отдельную функцию, выносишь из main 

порежь большую программу на несколько кусков, которые всё ещё большие, но уже поменьше.

'''
if __name__ == '__main__':
    main()