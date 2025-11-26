from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Optional
import numpy as np


@dataclass
class Agent:
    """
    Agent in early phase with probability distribution over eventual rankings.
    
    Each agent is assigned to a distinct worker, and the probability distribution
    is centered around that worker's rank to ensure no ties.
    """
    agent_id: int
    rank_distribution: np.ndarray  # P_j: probability distribution over rankings [1, n]
    # rank_distribution[k] = p_{jk} = probability that agent a_j ends up with rank k+1
    # (0-indexed array, so rank_distribution[0] is probability of rank 1)
    assigned_worker_rank: int  # The worker rank this agent is assigned to (ensures no ties)
    
    def __post_init__(self):
        """Ensure distribution is normalized."""
        if not np.isclose(np.sum(self.rank_distribution), 1.0):
            self.rank_distribution = self.rank_distribution / np.sum(self.rank_distribution)


@dataclass
class Worker:
    """
    Worker with quality ranking.
    
    Common preferences: All workers have the same ordinal preferences over firms.
    Each worker provides the same utility to every firm: u_f(w_j) = q_j for all firms f.
    """
    worker_id: int
    quality: float  # Quality q_j, ranked such that q_1 > q_2 > ... > q_n (common ranking for all firms)
    baseline_utility: float = 0.0  # Outside option utility = γ * (quality / max_quality) * max_f v(f)
    
    def utility(self, firm_value: float, t: int) -> float:
        """
        Worker utility from matching with firm in period t.
        
        Common preferences: U_w(f_i, t) = v(i) for all workers w.
        All workers rank firms identically based on firm value v(i).
        
        Args:
            firm_value: v(i) - valuation of firm i (same for all workers)
            t: Period (0 for early phase, 1 for regular phase)
        """
        return firm_value
    


@dataclass
class Firm:
    """
    Firm ranked by prestige with valuation.
    
    Common preferences: All firms have the same ordinal preferences over workers.
    Each firm provides the same utility to every worker: U_w(f_i) = v(i) for all workers w.
    """
    firm_id: int
    prestige: int  # Prestige rank: 1 is highest, m is lowest (f_1 > f_2 > ... > f_m)
    value: float  # v(i): Worker valuation of firm i (cardinal value, same for all workers)
    
    def utility(
        self, 
        worker_quality: Optional[float] = None,
        agent: Optional[Agent] = None,
        workers: Optional[List[Worker]] = None,
        t: int = 0,
    ) -> float:
        """
        Firm utility from hiring worker or agent in period t.
        
        Common preferences: u_f(w_j) = q_j for all firms f.
        All firms rank workers identically based on worker quality q_j.
        
        For period 0 (early phase): u_f(a_j) = sum_{k=1}^n p_{jk} * q_k
        For period 1 (regular phase): u_f(w_j) = q_j
        
        Args:
            worker_quality: Quality q_j (for period 1) - same utility for all firms
            agent: Agent with probability distribution (for period 0)
            workers: List of all workers sorted by quality (for period 0 expected utility)
            t: Period (0 for early phase, 1 for regular phase)
        """
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
    """Headhunter (clearinghouse) representing subsets of firms and workers."""
    headhunter_id: int
    firm_ids: Set[int]  # F_k ⊆ F: subset of firms this headhunter represents
    worker_ids: Set[int]  # W_k ⊆ W: subset of workers this headhunter has access to
        
    def can_match(self, firm_id: int, worker_id: int) -> bool:
        """Check if headhunter can match this firm and worker/agent."""
        return firm_id in self.firm_ids and worker_id in self.worker_ids
    
    def _compute_match_quality_agent(self, firm: Firm, agent: Agent, workers: List[Worker]) -> float:
        """
        Compute match quality μ(f_i, a_j) for period 0.
        
        μ(f_i, a_j) = v(f_i) + sum_{k=1}^n p_{jk} u_f(w_k)
        where u_f(w_k) = q_k for all firms f (common preferences)
        """
        # Expected firm utility from agent: sum_k p_{jk} * q_k (same for all firms)
        expected_firm_util = firm.utility(agent=agent, workers=workers, t=0)
        return firm.value + expected_firm_util
    
    def _compute_payment_agent(self, firm: Firm, agent: Agent, workers: List[Worker], v_max: float) -> float:
        """
        Compute payment η_i(a_j) for period 0.
        
        η_i(a_j) = (v_max - v(f_i)) · sum_{k=1}^n p_{jk} u_f(w_k)
        where u_f(w_k) = q_k for all firms f (common preferences)
        """
        # Expected firm utility from agent: sum_k p_{jk} * q_k (same for all firms)
        expected_firm_util = firm.utility(agent=agent, workers=workers, t=0)
        return (v_max - firm.value) * expected_firm_util
        return v_max - firm.value
    
    def _compute_match_quality_worker(self, firm: Firm, worker: Worker) -> float:
        """
        Compute match quality μ(f_i, w_j) for period 1.
        
        μ(f_i, w_j) = v(f_i) + u_f(w_j)
        where u_f(w_j) = q_j for all firms f (common preferences)
        """
        firm_util = firm.utility(worker_quality=worker.quality, t=1)
        return firm.value + firm_util
    
    def _compute_payment_worker(self, firm: Firm, worker: Worker, v_max: float) -> float:
        """
        Compute payment η_i(w_j) for period 1.
        
        η_i(w_j) = (v_max - v(f_i)) · u_f(w_j)
        where u_f(w_j) = q_j for all firms f (common preferences)
        """
        firm_util = firm.utility(worker_quality=worker.quality, t=1)
        return (v_max - firm.value) * firm_util
        return v_max - firm.value
    
    def utility_agent(self, firm: Firm, agent: Agent, workers: List[Worker], v_max: float, alpha: float) -> float:
        """
        Compute headhunter utility u_{h_k}(f_i, a_j) for period 0.
        
        u_{h_k}(f_i, a_j) = α · μ(f_i, a_j) + (1 - α) · η_i(a_j)
        
        Args:
            firm: Firm being matched
            agent: Agent being matched
            workers: List of all workers sorted by quality
            v_max: Maximum firm value (max_f v(f))
            alpha: Weight parameter α for match quality vs payment
        """
        match_quality = self._compute_match_quality_agent(firm, agent, workers)
        payment = self._compute_payment_agent(firm, agent, workers, v_max)
        return (alpha * match_quality) + ((1.0 - alpha) * payment)
    
    def utility_worker(self, firm: Firm, worker: Worker, v_max: float, alpha: float) -> float:
        """
        Compute headhunter utility u_{h_k}(f_i, w_j) for period 1.
        
        u_{h_k}(f_i, w_j) = α · μ(f_i, w_j) + (1 - α) · η_i(w_j)
        
        Args:
            firm: Firm being matched
            worker: Worker being matched
            v_max: Maximum firm value (max_f v(f))
            alpha: Weight parameter α for match quality vs payment
        """
        match_quality = self._compute_match_quality_worker(firm, worker)
        payment = self._compute_payment_worker(firm, worker, v_max)
        return (alpha * match_quality) + ((1.0 - alpha) * payment)

