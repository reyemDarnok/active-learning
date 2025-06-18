
from enum import Enum


class Scenario(str, Enum):
    """The Pelmo Scenarios. The name is the one letter shorthand and the value the full name"""
    C = "Châteaudun"
    H = "Hamburg"
    J = "Jokioinen"
    K = "Kremsmünster"
    N = "Okehampton"
    P = "Piacenza"
    O = "Porto"
    S = "Sevilla"
    T = "Thiva"