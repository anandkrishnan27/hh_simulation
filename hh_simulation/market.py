from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Tuple
import random
import numpy as np

from .agents import Worker, Firm, Headhunter

# Constants
PHI_COMMISSION = 0.3  # φ: Headhunter commission rate (typically 20-30% of first-year salary)


@dataclass
class Match:
    """Represents a match between a worker and firm."""
    worker_id: int
    firm_id: int
    period: int  # 0 for early phase, 1 for regular phase
    headhunter_id: Optional[int] = None
    worker_utility: float = 0.0
    firm_utility: float = 0.0
    headhunter_utility: float = 0.0
    observed_quality: float = 0.0  # Quality observed by firm (noisy at t=0, true at t=1)


@dataclass
class PeriodResults:
    """Results from a single period."""
    period: int
    matches: List[Match]
    unmatched_workers: List[int]
    unmatched_firms: List[int]


class Market:
    """
    Two-period market simulation.
    
    Period 0 (early phase): Firms observe noisy signals of worker quality
    Period 1 (regular phase): Firms observe true worker quality
    """
    
    def __init__(
        self,
        workers: List[Worker],
        firms: List[Firm],
        headhunters: List[Headhunter],
        delta_w: float = 0.1,  # δ_w: Worker early signing bonus
        delta_f: float = 0.1,  # δ_f: Firm benefit from early hiring
        delta_h: float = 0.05,  # δ_h: Headhunter benefit from early placement
        signal_noise_std: float = 0.2,  # σ: standard deviation of noise at t=0
        alpha: float = 0.3,  # α: Baseline utility multiplier (ū_j = α * q_j)
        lambda_prob: float = 1.0,  # λ: Parameter for logistic probability distribution
        omega: float = 0.5,  # ω: Weight for reputation impact (η = (1-ω)U_w + ωU_f)
        rng: Optional[random.Random] = None,
        np_rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.workers = workers
        self.firms = firms
        self.headhunters = headhunters
        self.delta_w = delta_w
        self.delta_f = delta_f
        self.delta_h = delta_h
        self.signal_noise_std = signal_noise_std
        self.alpha = alpha
        self.lambda_prob = lambda_prob
        self.omega = omega
        self.rng = rng if rng is not None else random.Random()
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng()
        
        # Sort workers by quality (descending) and firms by prestige (ascending)
        self.workers.sort(key=lambda w: w.quality, reverse=True)
        self.firms.sort(key=lambda f: f.prestige)
        
        # Create lookup dictionaries
        self.worker_dict = {w.worker_id: w for w in self.workers}
        self.firm_dict = {f.firm_id: f for f in self.firms}
    
    @staticmethod
    def random_market(
        num_workers: int = 50,
        num_firms: int = 20,
        num_headhunters: int = 5,
        delta_w: float = 0.1,
        delta_f: float = 0.1,
        delta_h: float = 0.05,
        signal_noise_std: float = 0.2,
        alpha: float = 0.5,
        lambda_prob: float = 1.0,
        omega: float = 0.5,
        seed: Optional[int] = None,
    ) -> "Market":
        """Create a random market with specified parameters."""
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        # Create workers with random quality, then sort by quality
        # Baseline utility: ū_j = α * q_j
        workers = []
        for i in range(num_workers):
            # Generate random quality values
            quality = np_rng.uniform(0.0, 1.0)
            baseline_utility = alpha * quality
            workers.append(Worker(worker_id=i, quality=quality, baseline_utility=baseline_utility))
        
        # Sort workers by quality (descending) so q_1 > q_2 > ... > q_n
        workers.sort(key=lambda w: w.quality, reverse=True)
        # Reassign IDs to maintain ordering
        for i, worker in enumerate(workers):
            worker.worker_id = i
        
        # Create firms with random values, then sort by value
        firms = []
        for i in range(num_firms):
            # Generate random value values
            value = np_rng.uniform(0.0, 1.0)
            firms.append(Firm(firm_id=i, prestige=i + 1, value=value))
        
        # Sort firms by value (descending) so f_1 > f_2 > ... > f_m
        firms.sort(key=lambda f: f.value, reverse=True)
        # Reassign prestige ranks to maintain ordering
        for i, firm in enumerate(firms):
            firm.prestige = i + 1
        
        # Create headhunters with random subsets of firms and workers
        headhunters = []
        for h_id in range(num_headhunters):
            num_firms_accessible = rng.randint(num_firms // 2, num_firms)
            firm_ids = set(rng.sample(range(num_firms), num_firms_accessible))
            
            num_workers_accessible = rng.randint(num_workers // 2, num_workers)
            worker_ids = set(rng.sample(range(num_workers), num_workers_accessible))
            
            headhunters.append(
                Headhunter(
                    headhunter_id=h_id,
                    firm_ids=firm_ids,
                    worker_ids=worker_ids,
                    phi=PHI_COMMISSION,
                    delta_h=delta_h,
                )
            )
        
        return Market(
            workers=workers,
            firms=firms,
            headhunters=headhunters,
            delta_w=delta_w,
            delta_f=delta_f,
            delta_h=delta_h,
            signal_noise_std=signal_noise_std,
            alpha=alpha,
            lambda_prob=lambda_prob,
            omega=omega,
            rng=rng,
            np_rng=np_rng,
        )
    
    def _get_observed_quality(self, worker: Worker, period: int) -> float:
        """Get quality observed by firms: q' = q + ε at t=0, q at t=1."""
        if period == 0:
            noise = self.np_rng.normal(0.0, self.signal_noise_std)
            return worker.quality + noise
        return worker.quality
    
    def _compute_eta(self, firm: Firm, worker: Worker, period: int) -> float:
        """
        Compute reputation impact η_{i,j}(t) from match.
        
        η_{ij}(t) = (1 - ω)U_w(f_i, t) + ω U_f(w_j, t)
        """
        worker_util = worker.utility(firm.value, self.delta_w, period)
        observed_quality = self._get_observed_quality(worker, period)
        firm_util = firm.utility(observed_quality, self.delta_f, period)
        return ((1.0 - self.omega) * worker_util) + (self.omega * firm_util)
    
    def _compute_acceptance_probability(
        self, worker: Worker, firm: Firm, period: int
    ) -> float:
        """
        Compute probability P(f_i, w_j, t) that match is successful.
        
        P(f_i, w_j, t) = 1 / (1 + exp(λ|U_w(f_i, t) - U_f(w_j, t)|))
        """
        worker_util = worker.utility(firm.value, self.delta_w, period)
        observed_quality = self._get_observed_quality(worker, period)
        firm_util = firm.utility(observed_quality, self.delta_f, period)
        utility_diff = abs(worker_util - firm_util)
        return 1.0 / (1.0 + np.exp(self.lambda_prob * utility_diff))
    
    def _match_period(self, period: int, unmatched_workers: Set[int], unmatched_firms: Set[int]) -> PeriodResults:
        """
        Perform matching for a single period.
        
        Algorithm:
        1. Each headhunter greedily proposes matches step-by-step (highest utility pairs first)
        2. Firms observe all proposals and greedily choose best worker (multiple firms can choose same worker)
        3. Workers with multiple proposals choose best firm, then compare to baseline utility
        4. Matches are finalized (no reneging)
        """
        matches: List[Match] = []
        
        # Step 1: Each headhunter proposes matches using step-by-step greedy algorithm
        # At step n, pick highest utility pair; if both unassigned, store match and mark as matched
        headhunter_proposals: Dict[int, Dict[int, int]] = {}  # {headhunter_id: {worker_id: firm_id}}
        
        for headhunter in self.headhunters:
            accessible_workers = [
                w_id for w_id in headhunter.worker_ids 
                if w_id in unmatched_workers
            ]
            accessible_firms = [
                f_id for f_id in headhunter.firm_ids 
                if f_id in unmatched_firms
            ]
            
            if not accessible_workers or not accessible_firms:
                headhunter_proposals[headhunter.headhunter_id] = {}
                continue
            
            # Compute all possible pair utilities
            all_pairs: List[Tuple[float, int, int, Worker, Firm]] = []
            for worker_id in accessible_workers:
                worker = self.worker_dict[worker_id]
                for firm_id in accessible_firms:
                    firm = self.firm_dict[firm_id]
                    observed_quality = self._get_observed_quality(worker, period)
                    acceptance_prob = self._compute_acceptance_probability(worker, firm, period)
                    eta = self._compute_eta(firm, worker, period)
                    expected_util = headhunter.utility(firm, worker, acceptance_prob, eta, period)
                    all_pairs.append((expected_util, worker_id, firm_id, worker, firm))
            
            # Sort by utility (descending)
            all_pairs.sort(key=lambda x: x[0], reverse=True)
            
            # Step-by-step greedy: pick highest utility pair, if both unassigned, mark as matched
            proposed_matching: Dict[int, int] = {}
            assigned_workers = set()
            assigned_firms = set()
            
            for expected_util, worker_id, firm_id, worker, firm in all_pairs:
                if worker_id not in assigned_workers and firm_id not in assigned_firms:
                    proposed_matching[worker_id] = firm_id
                    assigned_workers.add(worker_id)
                    assigned_firms.add(firm_id)
            
            headhunter_proposals[headhunter.headhunter_id] = proposed_matching
        
        # Step 2: Firms observe all proposals and greedily choose best worker
        # Multiple firms can propose to the same worker
        firm_choices: Dict[int, Tuple[Worker, Headhunter]] = {}  # {firm_id: (worker, headhunter)}
        
        for headhunter_id, matching in headhunter_proposals.items():
            headhunter = next(h for h in self.headhunters if h.headhunter_id == headhunter_id)
            for worker_id, firm_id in matching.items():
                if worker_id not in unmatched_workers or firm_id not in unmatched_firms:
                    continue
                
                worker = self.worker_dict[worker_id]
                firm = self.firm_dict[firm_id]
                observed_quality = self._get_observed_quality(worker, period)
                firm_util = firm.utility(observed_quality, self.delta_f, period)
                
                # Firm chooses best proposal (highest utility)
                if firm_id not in firm_choices:
                    firm_choices[firm_id] = (worker, headhunter)
                else:
                    current_worker, _ = firm_choices[firm_id]
                    current_observed = self._get_observed_quality(current_worker, period)
                    current_util = firm.utility(current_observed, self.delta_f, period)
                    if firm_util > current_util:
                        firm_choices[firm_id] = (worker, headhunter)
        
        # Step 3: Workers receive proposals and choose best, then compare to baseline
        worker_proposals: Dict[int, List[Tuple[Firm, Headhunter, float]]] = {}
        # {worker_id: [(firm, headhunter, worker_utility), ...]}
        
        for firm_id, (worker, headhunter) in firm_choices.items():
            firm = self.firm_dict[firm_id]
            worker_util = worker.utility(firm.value, self.delta_w, period)
            
            if worker.worker_id not in worker_proposals:
                worker_proposals[worker.worker_id] = []
            worker_proposals[worker.worker_id].append((firm, headhunter, worker_util))
        
        # Workers choose best proposal and accept if utility >= baseline
        for worker_id, proposals in worker_proposals.items():
            worker = self.worker_dict[worker_id]
            
            # Worker greedily chooses best proposal (highest worker utility)
            best_proposal = max(proposals, key=lambda x: x[2])  # x[2] is worker_utility
            firm, headhunter, worker_util = best_proposal
            
            # Worker accepts if utility >= baseline
            if worker_util < worker.baseline_utility:
                continue
            
            # Match is finalized
            observed_quality = self._get_observed_quality(worker, period)
            firm_util = firm.utility(observed_quality, self.delta_f, period)
            eta = self._compute_eta(firm, worker, period)
            headhunter_util = headhunter.utility(firm, worker, 1.0, eta, period)
            
            matches.append(
                Match(
                    worker_id=worker.worker_id,
                    firm_id=firm.firm_id,
                    period=period,
                    headhunter_id=headhunter.headhunter_id,
                    worker_utility=worker_util,
                    firm_utility=firm_util,
                    headhunter_utility=headhunter_util,
                    observed_quality=observed_quality,
                )
            )
        
        # Track unmatched
        matched_worker_ids = {m.worker_id for m in matches}
        matched_firm_ids = {m.firm_id for m in matches}
        unmatched_workers_list = [w for w in unmatched_workers if w not in matched_worker_ids]
        unmatched_firms_list = [f for f in unmatched_firms if f not in matched_firm_ids]
        
        return PeriodResults(
            period=period,
            matches=matches,
            unmatched_workers=unmatched_workers_list,
            unmatched_firms=unmatched_firms_list,
        )
    
    def run(self) -> List[PeriodResults]:
        """
        Run the two-period simulation.
        
        Returns results for both periods (t=0 and t=1).
        """
        results: List[PeriodResults] = []
        
        # Period 0: Early phase
        unmatched_workers = set(w.worker_id for w in self.workers)
        unmatched_firms = set(f.firm_id for f in self.firms)
        period_0_results = self._match_period(0, unmatched_workers, unmatched_firms)
        results.append(period_0_results)
        
        # Period 1: Regular phase (only unmatched workers/firms participate)
        matched_in_0 = set(m.worker_id for m in period_0_results.matches)
        matched_firms_in_0 = set(m.firm_id for m in period_0_results.matches)
        unmatched_workers_1 = unmatched_workers - matched_in_0
        unmatched_firms_1 = unmatched_firms - matched_firms_in_0
        
        period_1_results = self._match_period(1, unmatched_workers_1, unmatched_firms_1)
        results.append(period_1_results)
        
        return results
