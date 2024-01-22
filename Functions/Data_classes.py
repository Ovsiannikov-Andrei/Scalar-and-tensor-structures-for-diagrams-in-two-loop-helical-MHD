from dataclasses import *
from typing import Any


@dataclass
class DiagramData:
    """
    The class stores all the information that defines the diagram in the helical MHD.

    ARGUMENTS:

    nickel_index -- here we save a Nickel index from the line with the data,
    output_file_name -- the name of the file with results,
    momentums_at_vertices -- distribution of momentums at the vertices,
    indexb -- index of the inflowing field,
    indexB -- index of the outflowing field,
    P_structure -- here we save the Projector operator arguments,
    H_structure -- here we save the Kronecker delta arguments,
    kd_structure -- here we save the Kronecker delta arguments,
    momentum_structure -- here we save all momentums and their components,
    integrand_tensor_part -- here we save the tensor structure (product of the tensor operators),
    integrand_scalar_part -- here we save the scalar function,
    expression_UV_convergence_criterion -- corresponding integral is convergent (True/False)
    """

    nickel_index: str
    output_file_name: str
    momentums_at_vertices: list
    indexb: int
    indexB: int
    P_structure: list
    H_structure: list
    kd_structure: list
    momentum_structure: list
    integrand_tensor_part: Any
    integrand_scalar_part: Any
    expression_UV_convergence_criterion: bool


@dataclass
class IntegrandData:
    """
    The class stores all the information about corresponding diagram integrand.

    ARGUMENTS:

    scalar_part_without_repl -- here we save scalar part of the integrand reduced
    to a common denominator and partially simplified,
    scalar_part_depending_only_on_uo -- here we save scalar part of the integrand
    depending only on uo (replacing  k, q -- > B*k/nuo, B*q/nuo),
    scalar_part_field_and_nuo_factor_lambda -- here we save factor by which the scalar part of the
    integrand is multiplied (all dependence on Cutoff and nuo),
    scalar_part_field_and_nuo_factor_B -- here we save factor by which the scalar part of the
    integrand is multiplied (all dependence on |B| and nuo),
    tensor_convolution_lambda_momentum_depend_part -- here we save computed tensor structure
    corresponding to the Cutoff part of the (divergent) integrand
    tensor_convolution_lambda_field_and_nuo_factor -- here we save a common multiplier
    (all dependence on Cutoff and nuo) which corresponds to the Cutoff integrand's part
    tensor_convolution_B_momentum_depend_part -- here we save computed tensor structure
    corresponding to the |B| part of the (convergent) integrand
    tensor_convolution_B_field_and_nuo_factor -- here we save a common multiplier
    (all dependence on |B| and nuo) which corresponds to the |B| integrand's part
    """

    scalar_part_without_replacement: Any
    convergent_scalar_part_depending_only_on_uo: Any
    divergent_scalar_part_depending_only_on_uo: Any
    scalar_part_field_and_nuo_factor_lambda: Any
    scalar_part_field_and_nuo_factor_B: Any
    tensor_convolution_lambda_momentum_depend_part: Any
    tensor_convolution_lambda_field_and_nuo_factor: Any
    tensor_convolution_B_momentum_depend_part: Any
    tensor_convolution_B_field_and_nuo_factor: Any


@dataclass
class NickelIndexInfo:
    """
    This class stores all the information from the n-th line from the input txt file.

    ARGUMENTS:

    result_file_name -- here we save a name of the file with results,
    nickel_index -- here we save a Nickel index from the line with the data,
    symmetry_factor -- here we save a symmetry factor from the line with the data
    """

    result_file_name: str
    nickel_index: str
    symmetry_factor: int


@dataclass
class InternalAndExternalLines:
    """
    This class stores all information about the lines in the diagram,
    dividing them into external and internal.

    ARGUMENTS:

    internal_propagators -- here we save all information about internal lines,
    external_propagators -- here we save all information about external lines,
    dict_internal_propagators -- here we save all information about internal lines
    in dictionary with key digits,
    dict_external_propagators -- here we save all information about external lines
    in dictionary with key digits,
    """

    internal_propagators: list
    external_propagators: list
    dict_internal_propagators: dict
    dict_external_propagators: dict


@dataclass
class IndependentMomentumsInHelicalPropagators:
    """
    This class stores information about which lines in the diagram are assigned independent
    momentums and frequencies (k, w_k, q, w_q).

    ARGUMENTS:

    momentums_for_helical_lines -- here we save momentums flowing in lines containing kernel D_v,
    frequencies_for_helical_lines -- here we save frequencies flowing in lines containing kernel D_v
    """

    momentums_for_helical_lines: dict
    frequencies_for_helical_lines: dict


@dataclass
class ArgumentsDistributionAlongLines:
    """
    This class stores information about all arguments (momentums and frequencies) flowing along the lines.

    ARGUMENTS:

    frequency_distribution -- here we save momentums flowing in diagram lines,
    frequency_distribution -- here we save frequencies flowing in diagram lines
    """

    momentum_distribution: dict
    frequency_distribution: dict


@dataclass
class ArgumentsDistributionAlongLinesAtZeroExternalArguments:
    """
    This class stores information about all arguments (momentums and frequencies) flowing along the lines,
    provided that the corresponding amputated diagram is assumed at zero external impulses and frequencies.

    ARGUMENTS:

    frequency_distribution -- here we save momentums flowing in diagram lines (p = 0),
    frequency_distribution -- here we save frequencies flowing in diagram lines (w = 0)
    """

    momentum_distribution: dict
    frequency_distribution: dict


@dataclass
class MomentumFrequencyDistributionAtVertices:
    """
    This class stores information about the arguments, as well as the inflow and outflow
    lines to each vertex (they differ in the arguments signs: "+" for inflows, "-" for outflows).

    ARGUMENTS:

    indexB -- here we save the index of inner tail B,
    indexb -- here we save the index of inner tail b,
    all_data_of_vertexes_distribution -- here we save the type of propagators
    (and the structure of their arguments) stuck at each vertex,
    frequency_and_momentum_distribution_at_vertices -- here we save as a dictionary
    with vertex numbers as keys the type of propagators
    (and the structure of their arguments) stuck at each vertex,
    momentums_at_vertices -- here from frequency_and_momentum_distribution_at_vertices
    the frequencies are removed and the data type is changed to list
    """

    indexB: int
    indexb: int
    all_data_of_vertices_distribution: list
    frequency_and_momentum_distribution_at_vertices: dict
    momentums_at_vertices: list


@dataclass
class IntegrandPropagatorProduct:
    """
    This class stores all the necessary information regarding
    obtaining a product of propagators.

    ARGUMENTS:

    propagator -- here we save the full product of propagators
    scalar_part -- here we save the product of propagators (without tensor structure),
    tensor_part -- here we save the tensor structures (P_ij, H_ij, vertex factors),
    P_data -- here we save all indices of the projctors in the form [[momentum, index1, index2]],
    H_data -- here we save all indices of the helical structures in the form [[momentum, index1, index2]],
    WfMath_propagators_prod -- here we save the propagator product argument structure
    (for Wolfram Mathematica file),
    """

    propagator_prod: Any = 1
    scalar_part: Any = 1
    tensor_part: Any = 1
    P_data: list = field(default_factory=list)
    H_data: list = field(default_factory=list)
    WfMath_propagators_prod: str = ""


@dataclass
class IntegrandScalarAndTensorParts(IntegrandPropagatorProduct):
    """
    This class stores all the necessary information regarding
    obtaining the appropriate integrand for the amputated diagram
    (product of propagators and vertices factors).

    ARGUMENTS:

    mom_data -- here we save all momentums and their components in the form [[k, i]],
    kd_data -- here we save all indices in Kronecker delta in the form [[index 1, index 2]],
    """

    mom_data: list = field(default_factory=list)
    kd_data: list = field(default_factory=list)


@dataclass
class IntegrandIsRationalFunction:
    """
    This class stores all expressions obtained after reduction to a common
    denominator of the expression obtained after frequency integration.
    In the numerator of the resulting expression, a common factor can be
    distinguished. It is convenient to divide the output by this factor and
    the remainder. replacing of variables k, q --> B*k/nuo, B*q/nuo, the
    common factor is divided into a part depending only on the momenta and
    cosines of the corresponding angles, and into a part depending on the
    dimension factors. The remaining rational function is also preceded by
    some additional dimensional factor. This two dimensional factors are combined
    into a single dimensional factor, the terms depending on the integration
    variables are combined into an integrand that depends only on the
    magnetic Prandtl number. By construction, all dimensional factors for all
    UV finite expressions must be the same.

    ARGUMENTS:

    residues_sum_without_common_factor -- result without common factor,
    common_factor -- common factor,
    residues_sum_without_dim_factor_after_subs -- expression depending only on
    the magnetic Prandtl number and suitable for numerical integration,
    new_dim_factor_after_subs -- dimensional factor by which the corresponding
    integral is multiplied.
    """

    residues_sum_without_common_factor: Any
    common_factor: Any
    residues_sum_without_dim_factor_after_subs: Any
    new_dim_factor_after_subs: Any


@dataclass
class IntegrandPrefactorAfterSubstitution:
    """
    After replacing of variables k, q --> B*k/nuo, B*q/nuo,
    the common factor in front of the rational function (obtained
    by reducing to a common denominator the result obtained after
    integrating over frequencies) can be divided into a part
    depending on k, q and a part depending on the dimensional
    parameters. The first part is multiplied by a rational function
    and the resulting expression can already be suitable for numerical
    integration.

    ARGUMENTS:

    momentum_depend_factor -- term depending on k and q
    momentum_independ_factor -- term depending on dimensional factors (B, nuo, ...)
    """

    momentum_depend_factor: Any
    momentum_independ_factor: Any


@dataclass
class IntegrandTensorStructure:
    """
    After calculating the tensor structure, the Parisi-Waltman scalarization
    procedure is performed separately for the divergent part (proportional to Lambda)
    and the convergent part (proportional to B).

    ARGUMENTS:

    lambda_proportional_term -- result for divergent part
    B_proportional_term -- result for convergent part
    """

    lambda_proportional_term: Any
    B_proportional_term: Any
