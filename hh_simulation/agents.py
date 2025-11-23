from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set
import numpy as np


@dataclass
class Worker:
    """Worker with true quality and baseline utility."""
    worker_id: int
    quality: float  # True quality q_j, ranked such that q_1 > q_2 > ... > q_n
    baseline_utility: float  # ū_j = α * q_j: utility of remaining at banking firm
    
    def utility(self, firm_value: float, delta_w: float, t: int) -> float:
        """
        Worker utility from matching with firm in period t.
        
        U_w(f_i, t) = v(i) + δ_w * 1{t=0}
        
        Args:
            firm_value: v(i) - valuation of firm i
            delta_w: Early signing bonus/incentive
            t: Period (0 for early phase, 1 for regular phase)
        """
        early_bonus = delta_w if t == 0 else 0.0
        return firm_value + early_bonus
    
    def accepts_offer(self, firm_value: float, delta_w: float, t: int) -> bool:
        """Check if worker accepts offer (utility >= baseline)."""
        return self.utility(firm_value, delta_w, t) >= self.baseline_utility


@dataclass
class Firm:
    """Firm ranked by prestige with valuation."""
    firm_id: int
    prestige: int  # Prestige rank: 1 is highest, m is lowest (f_1 > f_2 > ... > f_m)
    value: float  # v(i): Worker valuation of firm i (cardinal value, not just ordinal)
    
    def utility(self, worker_quality: float, delta_f: float, t: int) -> float:
        """
        Firm utility from hiring worker with quality q in period t.
        
        U_f(w_j, t) = γ(q'_j) + δ_f if t=0, γ(q_j) if t=1
        
        Args:
            worker_quality: Observed quality (noisy signal at t=0, true quality at t=1)
            delta_f: Benefit from securing talent early
            t: Period (0 for early phase, 1 for regular phase)
        """
        # γ(·) maps worker quality to firm utility (linear mapping)
        gamma_q = worker_quality
        early_benefit = delta_f if t == 0 else 0.0
        return gamma_q + early_benefit


@dataclass
class Headhunter:
    """Headhunter (clearinghouse) representing subsets of firms and workers."""
    headhunter_id: int
    firm_ids: Set[int]  # F_k ⊆ F: subset of firms this headhunter represents
    worker_ids: Set[int]  # W_k ⊆ W: subset of workers this headhunter has access to
    phi: float  # Monetary commission (typically 20-30% of first-year salary)
    delta_h: float  # Benefit from completing placement early
        
    def can_match(self, firm_id: int, worker_id: int) -> bool:
        """Check if headhunter can match this firm and worker."""
        return firm_id in self.firm_ids and worker_id in self.worker_ids

    def utility(
        self,
        firm: Firm,
        worker: Worker,
        acceptance_prob: float,
        eta: float,
        t: int,
    ) -> float:
        """
        Headhunter utility from proposing match between firm and worker in period t.
        
        U_h(f_i, w_j, t) = P(offer accepted) * (β + η_{i,j}) + δ_h * 1{t=0}
        
        Args:
            firm: Firm being matched
            worker: Worker being matched
            acceptance_prob: Probability that offer will be accepted
            eta: Reputation impact from match (η_{i,j})
            t: Period (0 for early phase, 1 for regular phase)
        """
        if not self.can_match(firm.firm_id, worker.worker_id):
            return 0.0

        commission_term = acceptance_prob * (self.phi + eta)
        early_benefit = self.delta_h if t == 0 else 0.0
        return commission_term + early_benefit

