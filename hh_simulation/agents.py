from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Optional
import numpy as np


@dataclass
class Agent:
    """Early-phase agent with a probability distribution over ranks."""
    agent_id: int
    rank_distribution: np.ndarray  # P_j: probability distribution over rankings [1, n]
    # rank_distribution[k] = p_{jk} = probability that agent a_j ends up with rank k+1
    # (0-indexed array, so rank_distribution[0] is probability of rank 1)
    assigned_worker_rank: int  # The worker rank this agent is assigned to (ensures no ties)
    
    def __post_init__(self):
        """Normalize the distribution if it is slightly off."""
        if not np.isclose(np.sum(self.rank_distribution), 1.0):
            self.rank_distribution = self.rank_distribution / np.sum(self.rank_distribution)


@dataclass
class Worker:
    """Worker with a quality rank shared by all firms."""
    worker_id: int
    quality: float  # Quality q_j, ranked such that q_1 > q_2 > ... > q_n (common ranking for all firms)
    baseline_utility: float = 0.0  # Outside option utility = γ * (quality / max_quality) * max_f v(f)
    
    def utility(self, firm_value: float, t: int) -> float:
        """Worker utility from matching with a firm in period t."""
        return firm_value
    


@dataclass
class Firm:
    """Firm ranked by prestige and valued equally by workers."""
    firm_id: int
    prestige: int  # Prestige rank: 1 is highest, m is lowest (f_1 > f_2 > ... > f_m)
    value: float  # v(i): Worker valuation of firm i (cardinal value, same for all workers)
    baseline_utility: float = 0.0  # Outside option utility = γ * (value / max_value) * max_w q(w)
    
    def utility(
        self, 
        worker_quality: Optional[float] = None,
        agent: Optional[Agent] = None,
        workers: Optional[List[Worker]] = None,
        t: int = 0,
    ) -> float:
        """Firm utility from hiring an agent or worker in period t."""
        if t == 0:
            # Early phase: compute expected utility from agent's distribution
            if agent is None or workers is None:
                raise ValueError("For period 0, must provide agent and workers list")
            
            # Expected utility: sum over all possible rankings
            # Since all firms have common preferences, u_f(w_k) = q_k for all firms f
            expected_util = 0.0
            for rank_idx, prob in enumerate(agent.rank_distribution):
                if prob > 0 and rank_idx < len(workers):
                    # rank_idx is 0-indexed, so rank_idx+1 is the actual rank
                    # workers[rank_idx] is the worker with rank rank_idx+1 (since workers sorted descending)
                    worker = workers[rank_idx]
                    # Common preferences: u_f(w_k) = q_k for all firms f
                    expected_util += prob * worker.quality
            
            return expected_util
        else:
            # Regular phase: use true quality
            if worker_quality is None:
                raise ValueError("For period 1, must provide worker_quality")
            # Common preferences: u_f(w_j) = q_j for all firms f
            return worker_quality


@dataclass
class Headhunter:
    """Headhunter representing subsets of firms and workers."""
    headhunter_id: int
    firm_ids: Set[int]  # F_k ⊆ F: subset of firms this headhunter represents
    worker_ids: Set[int]  # W_k ⊆ W: subset of workers this headhunter has access to
        
    def can_match(self, firm_id: int, worker_id: int) -> bool:
        """Check if this headhunter can pair the given firm and worker."""
        return firm_id in self.firm_ids and worker_id in self.worker_ids
    
    def _compute_match_quality_agent(self, firm: Firm, agent: Agent, workers: List[Worker]) -> float:
        """Compute μ(f_i, a_j) for period 0."""
        # Expected firm utility from agent: sum_k p_{jk} * q_k (same for all firms)
        expected_firm_util = firm.utility(agent=agent, workers=workers, t=0)
        return firm.value + expected_firm_util
    
    def _compute_payment_agent(self, firm: Firm, agent: Agent, workers: List[Worker], v_max: float) -> float:
        """Compute η_i(a_j) for period 0."""
        # Expected firm utility from agent: sum_k p_{jk} * q_k (same for all firms)
        expected_firm_util = firm.utility(agent=agent, workers=workers, t=0)
        # return (v_max - firm.value) * expected_firm_util
        return (v_max - firm.value)
    
    def _compute_match_quality_worker(self, firm: Firm, worker: Worker) -> float:
        """Compute μ(f_i, w_j) for period 1."""
        firm_util = firm.utility(worker_quality=worker.quality, t=1)
        return firm.value + firm_util
    
    def _compute_payment_worker(self, firm: Firm, worker: Worker, v_max: float) -> float:
        """Compute η_i(w_j) for period 1."""
        firm_util = firm.utility(worker_quality=worker.quality, t=1)
        # return (v_max - firm.value) * firm_util
        return (v_max - firm.value)
    
    def utility_agent(self, firm: Firm, agent: Agent, workers: List[Worker], v_max: float, alpha: float) -> float:
        """Compute headhunter utility u_{h_k}(f_i, a_j) for period 0."""
        match_quality = self._compute_match_quality_agent(firm, agent, workers)
        payment = self._compute_payment_agent(firm, agent, workers, v_max)
        return (alpha * match_quality) + ((1.0 - alpha) * payment)
    
    def utility_worker(self, firm: Firm, worker: Worker, v_max: float, alpha: float) -> float:
        """Compute headhunter utility u_{h_k}(f_i, w_j) for period 1."""
        match_quality = self._compute_match_quality_worker(firm, worker)
        payment = self._compute_payment_worker(firm, worker, v_max)
        return (alpha * match_quality) + ((1.0 - alpha) * payment)

