"""
Scheme 4: LLM-based Strategy Clustering

Uses LLM's semantic understanding to cluster strategies and perform
opponent selection without requiring actual game simulation.
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from ..prompts.base_prompts import CLUSTER_PROMPT


@dataclass
class StrategyCluster:
    """Represents a cluster of similar strategies."""
    id: int
    strategies: List[str]
    theme: str
    reason_for_merge: str


@dataclass
class ClusterResult:
    """Result of clustering operation."""
    clusters: List[StrategyCluster]
    representatives: List[Tuple[str, str]]  # (name, prompt)
    reduced_pool: List[str]


class Scheme4Clustering:
    """
    LLM-based clustering scheme.

    Uses LLM semantic understanding to:
    1. Cluster strategies by semantic similarity
    2. Select representative strategies
    3. Compute meta-strategy as uniform distribution over representatives
    """

    def __init__(self, llm_client: Optional[callable] = None):
        """
        Initialize clustering scheme.

        Args:
            llm_client: LLM API client function.
                       Takes prompt string, returns response string.
        """
        self.llm_client = llm_client

    def cluster_strategies(
        self,
        strategy_pool: List[Tuple[str, str]],  # List of (name, prompt)
        num_clusters: int = 5
    ) -> ClusterResult:
        """
        Cluster strategies using LLM.

        Args:
            strategy_pool: List of (strategy_name, strategy_prompt)
            num_clusters: Target number of clusters (3-5 recommended)

        Returns:
            ClusterResult with clusters, representatives, and reduced pool
        """
        if self.llm_client is None:
            raise ValueError("LLM client is required for clustering")

        # Format strategy pool for prompt
        pool_text = "\n".join([
            f"- {name}: {prompt[:200]}..." if len(prompt) > 200 else f"- {name}: {prompt}"
            for name, prompt in strategy_pool
        ])

        prompt = CLUSTER_PROMPT.format(
            strategy_pool=pool_text
        )

        response = self.llm_client(prompt)

        # Parse LLM response
        return self._parse_cluster_response(response, strategy_pool)

    def _parse_cluster_response(
        self,
        response: str,
        strategy_pool: List[Tuple[str, str]]
    ) -> ClusterResult:
        """Parse LLM response into structured clusters."""
        # Try to extract JSON from response
        try:
            # Look for JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON-like content
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Fallback: create simple clusters
                    return self._fallback_clustering(strategy_pool)

            data = json.loads(json_str)

            clusters = []
            for i, cluster_data in enumerate(data.get("clusters", [])):
                clusters.append(StrategyCluster(
                    id=i,
                    strategies=cluster_data.get("strategies", []),
                    theme=cluster_data.get("theme", ""),
                    reason_for_merge=cluster_data.get("reason_for_merge", "")
                ))

            representatives = []
            for rep_data in data.get("representatives", []):
                strategy_name = rep_data.get("strategy_name", "")
                # Find the actual prompt for this strategy
                prompt = ""
                for name, p in strategy_pool:
                    if name == strategy_name:
                        prompt = p
                        break
                representatives.append((strategy_name, prompt))

            reduced_pool = data.get("reduced_pool", [])

            return ClusterResult(
                clusters=clusters,
                representatives=representatives,
                reduced_pool=reduced_pool
            )

        except (json.JSONDecodeError, KeyError):
            # Fallback to simple clustering if parsing fails
            return self._fallback_clustering(strategy_pool)

    def _fallback_clustering(
        self,
        strategy_pool: List[Tuple[str, str]]
    ) -> ClusterResult:
        """
        Fallback clustering when LLM parsing fails.

        Simply groups strategies by order into roughly equal clusters.
        """
        n = len(strategy_pool)
        num_clusters = min(5, max(3, n // 2))

        clusters = []
        representatives = []
        reduced_pool = []

        chunk_size = max(1, n // num_clusters)

        for i in range(num_clusters):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n)

            if start_idx >= n:
                break

            chunk_strategies = strategy_pool[start_idx:end_idx]
            strategy_names = [name for name, _ in chunk_strategies]

            clusters.append(StrategyCluster(
                id=i,
                strategies=strategy_names,
                theme=f"Group {i+1}",
                reason_for_merge="Fallback clustering"
            ))

            # Select first strategy as representative
            rep_name, rep_prompt = chunk_strategies[0]
            representatives.append((rep_name, rep_prompt))
            reduced_pool.append(rep_name)

        return ClusterResult(
            clusters=clusters,
            representatives=representatives,
            reduced_pool=reduced_pool
        )

    def get_meta_strategy(self, cluster_result: ClusterResult) -> np.ndarray:
        """
        Compute meta-strategy as uniform distribution over representatives.

        Returns:
            Uniform probability distribution over representatives
        """
        import numpy as np

        n = len(cluster_result.representatives)
        if n == 0:
            return np.array([])

        # Uniform distribution
        return np.ones(n) / n

    def select_opponents(
        self,
        cluster_result: ClusterResult,
        strategy_name: str,
        num_opponents: int = 3
    ) -> List[Tuple[str, str]]:
        """
        Select diverse opponents from different clusters.

        Ensures selected opponents come from different clusters
        for maximum coverage.
        """
        if not cluster_result.clusters:
            return cluster_result.representatives[:num_opponents]

        selected = []
        remaining = list(cluster_result.representatives)

        # First try to pick from different clusters
        for cluster in cluster_result.clusters:
            if len(selected) >= num_opponents:
                break

            # Find a representative from this cluster
            for rep_name, rep_prompt in cluster_result.representatives:
                if rep_name in cluster.strategies and rep_name != strategy_name:
                    selected.append((rep_name, rep_prompt))
                    remaining.remove((rep_name, rep_prompt))
                    break

        # If we still need more, fill from remaining
        while len(selected) < num_opponents and remaining:
            selected.append(remaining.pop(0))

        return selected[:num_opponents]