import numpy as np
from scipy import ndimage

from .constants import RESOURCE_TYPES
from ..lux_ai.lux.game_constants import GAME_CONSTANTS


def _get_resource_per_second(resource: str) -> float:
    mining_rate = GAME_CONSTANTS["PARAMETERS"]["WORKER_COLLECTION_RATE"][resource.upper()]
    return float(mining_rate)


def _get_fuel_per_second(resource: str) -> float:
    mining_rate = _get_resource_per_second(resource)
    conversion_rate = GAME_CONSTANTS["PARAMETERS"]["RESOURCE_TO_FUEL_RATE"][resource.upper()]
    return float(mining_rate * conversion_rate)


def _normalize_resource_mat(resource_mat: np.ndarray, resource: str, time_horizon: int) -> np.ndarray:
    assert time_horizon >= 1
    return np.minimum(
        # In case the resource will be depleted within the time horizon
        resource_mat / (time_horizon * _get_resource_per_second(resource)),
        1.
    )


def get_resource_per_second_mat(
        resource_mat: np.ndarray,
        resource: str,
        time_horizon: int = 1,
) -> np.ndarray:
    """
    Given a matrix of a given resource, returns the raw resources_per_second of each location on the map
    When time_horizon > 1, returns the average resources_per_second of each location over time_horizon steps
    All calculations assume only one unit is mining each resource at a time
    """
    resource_mat_normalized = _normalize_resource_mat(resource_mat, resource, time_horizon)
    return ndimage.convolve(resource_mat_normalized, RESOURCE_PER_SECOND_KERNELS[resource], mode="constant", cval=0.)


def get_fuel_per_second_mat(
        resource_mat: np.ndarray,
        resource: str,
        time_horizon: int = 1,
) -> np.ndarray:
    """
    Given a matrix of a given resource, returns the raw fuel_per_second of each location on the map
    When time_horizon > 1, returns the average resources_per_second of each location over time_horizon steps
    All calculations assume only one unit is mining each resource at a time
    """
    resource_mat_normalized = _normalize_resource_mat(resource_mat, resource, time_horizon)
    return ndimage.convolve(resource_mat_normalized, FUEL_PER_SECOND_KERNELS[resource], mode="constant", cval=0.)


RESOURCE_PER_SECOND = {
    r: _get_resource_per_second(r) for r in RESOURCE_TYPES
}
RESOURCE_PER_SECOND_KERNELS = {
    r: np.array([[0., rps, 0.],
                 [rps, rps, rps],
                 [0., rps, 0.]])
    for r, rps in RESOURCE_PER_SECOND.items()
}
FUEL_PER_SECOND = {
    r: _get_fuel_per_second(r) for r in RESOURCE_TYPES
}
FUEL_PER_SECOND_KERNELS = {
    r: np.array([[0., fps, 0.],
                 [fps, fps, fps],
                 [0., fps, 0.]])
    for r, fps in FUEL_PER_SECOND.items()
}
