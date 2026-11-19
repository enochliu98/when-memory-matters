"""
LangPSRO: Large Language Models as Policy Space Response Oracles

A framework for applying PSRO (Policy Space Response Oracles) to LLMs
through prompt engineering, evaluated on TextArena.
"""

from .envs.textarena_env import TextArenaEnv, GameResult
from .schemes.scheme1 import Scheme1GS, Strategy, MetaPayoffMatrix
from .schemes.scheme2 import Scheme2GS, TrueSkillRating
from .schemes.scheme4 import Scheme4Clustering, ClusterResult
from .schemes.brs import BestResponseSolver, BRSResult
from .prompts.base_prompts import BASE_PROMPT, BRS_PROMPT, CLUSTER_PROMPT
from .runner import LangPSRO

__all__ = [
    # Environment
    "TextArenaEnv",
    "GameResult",
    # Schemes
    "Scheme1GS",
    "Scheme2GS",
    "Scheme4Clustering",
    "Strategy",
    "MetaPayoffMatrix",
    "TrueSkillRating",
    "ClusterResult",
    "BRSResult",
    "BestResponseSolver",
    # Prompts
    "BASE_PROMPT",
    "BRS_PROMPT",
    "CLUSTER_PROMPT",
    # Runner
    "LangPSRO",
]