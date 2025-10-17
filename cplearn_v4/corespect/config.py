from dataclasses import dataclass
from typing import Literal, Union

@dataclass
class FlowRankConfig:
    """
        Configuration for FlowRank algorithm.
        q- int: Number of nearest neighbors to consider for constructing the k-NN graph.
        r- int: Number of neighbors to consider during the ascending random walk process.

    """


    q: int = 20
    r: int = 10


@dataclass
class StableCoreConfig:

    """
        densification:
        - "k-nn":  local k-nearest-neighbor densification
        - "rw":    random-walk-based densification
        - False:   skip refinement

    """

    auto_select_core_frac: bool = False
    core_frac: float = 0.15
    resolution: float = 1.0
    ng_num: int = 15
    densification: Union[Literal["k-nn", "rw"], bool] = False


@dataclass
class FineGrainedConfig:

    stable_core_densification: Union[Literal["k-nn", "rw"], bool] = False
    auto_select_core_frac: bool = False
    core_frac: float=0.2
    fine_grain_densification : Union[Literal["k-nn", "rw"], bool] = False
    starting_resolution: float = 1.5


@dataclass
class PropagationConfig:

    """
    Configuration for label propagation from core points.
    normalized: bool
        Whether to use a normalized graph Laplacian

    connectivity: {"in", "out", "sym"}
        - "in": Use incoming edges for propagation.
        - "out": Use outgoing edges for propagation.
        - "sym": Use symmetrized edges for propagation.


    mode: {"default", "momentum"}
        - "default": Standard label propagation.
        - "momentum": Momentum-based propagation for potentially faster convergence.

    alpha: float
        Damping factor for propagation, typically between 0 and 1.

    tol: float
        Tolerance for convergence.

    max_iter: int
        Maximum number of iterations for the propagation to perform.
    """


    normalized: bool = True
    connectivity: Literal["in", "out", "sym"] = "sym"
    mode : Literal["default", "momentum"] = "momentum"
    alpha: float = 0.85
    tol: float = 1e-6
    max_iter: int = 50
