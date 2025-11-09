from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Dict
import random
import numpy as np

from .agents import Worker, Firm, Headhunter


@dataclass
class Match:
    """Represents a match between a worker and firm."""
    worker_id: int
    firm_id: int
    period: int  # 0 for early phase, 1 for regular phase
    headhunter_id: Optional[int] = None  # Which headhunter facilitated the match
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
        delta_w: float = 0.1,  # Worker early signing bonus
        delta_f: float = 0.1,  # Firm benefit from early hiring
        delta_h: float = 0.05,  # Headhunter benefit from early placement
        signal_noise_std: float = 0.2,  # σ: standard deviation of noise at t=0
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
        self.rng = rng if rng is not None else random.Random()
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng()
        
        # Sort workers by quality (descending) and firms by rank (ascending)
        self.workers.sort(key=lambda w: w.quality, reverse=True)
        self.firms.sort(key=lambda f: f.rank)
        
        # Create lookup dictionaries
        self.worker_dict = {w.worker_id: w for w in self.workers}
        self.firm_dict = {f.firm_id: f for f in self.firms}
    
    @staticmethod
    def random_market(
        num_workers: int = 50,
        num_firms: int = 10,
        num_headhunters: int = 3,
        delta_w: float = 0.1,
        delta_f: float = 0.1,
        delta_h: float = 0.05,
        signal_noise_std: float = 0.2,
        seed: Optional[int] = None,
    ) -> "Market":
        """Create a random market with specified parameters."""
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        # Create workers with decreasing quality
        workers = []
        for i in range(num_workers):
            # Quality decreases from 1.0 to 0.0
            quality = 1.0 - (i / max(num_workers - 1, 1))
            # Baseline utility depends on quality (higher quality = more likely to get promoted)
            # But also has some randomness
            baseline_utility = 0.05 + quality * 0.3 + rng.uniform(-0.1, 0.1)
            baseline_utility = max(0.0, min(0.6, baseline_utility))  # Clip to reasonable range
            workers.append(Worker(worker_id=i, quality=quality, baseline_utility=baseline_utility))
        
        # Create firms with ranks 1 to m
        firms = [Firm(firm_id=i, rank=i + 1) for i in range(num_firms)]
        
        # Create headhunters with random subsets of firms and workers
        headhunters = []
        for h_id in range(num_headhunters):
            # Each headhunter represents a random subset of firms
            num_firms_accessible = rng.randint(num_firms // 2, num_firms)
            firm_ids = set(rng.sample(range(num_firms), num_firms_accessible))
            
            # Each headhunter has access to a random subset of workers
            num_workers_accessible = rng.randint(num_workers // 2, num_workers)
            worker_ids = set(rng.sample(range(num_workers), num_workers_accessible))
            
            # Commission and early benefit
            beta = rng.uniform(0.2, 0.3)  # 20-30% commission
            delta_h_val = rng.uniform(0.0, 0.1)
            
            headhunters.append(
                Headhunter(
                    headhunter_id=h_id,
                    firm_ids=firm_ids,
                    worker_ids=worker_ids,
                    beta=beta,
                    delta_h=delta_h_val,
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
            rng=rng,
            np_rng=np_rng,
        )
    
    def _get_observed_quality(self, worker: Worker, period: int) -> float:
        """Get quality observed by firms in given period."""
        if period == 0:
            # Early phase: noisy signal q' = q + ε, ε ~ N(0, σ²)
            noise = self.np_rng.normal(0.0, self.signal_noise_std)
            return max(0.0, min(1.0, worker.quality + noise))  # Clip to [0, 1]
        else:
            # Regular phase: true quality
            return worker.quality
    
    def _compute_eta(self, firm: Firm, worker: Worker) -> float:
        """
        Compute reputation impact η_{i,j} from match.
        
        This can depend on match quality - higher quality matches may have
        positive reputation impact, while poor matches may have negative impact.
        """
        # Simple model: positive for high-quality workers at high-prestige firms
        # Negative for low-quality workers at high-prestige firms
        match_quality = worker.quality * (1.0 / firm.rank)
        # Scale to reasonable range
        return (match_quality - 0.5) * 0.2
    
    def _compute_acceptance_probability(
        self, worker: Worker, firm: Firm, period: int
    ) -> float:
        """
        Compute probability that worker accepts offer.
        
        This is based on worker utility and baseline utility.
        """
        worker_util = worker.utility(firm.rank, self.delta_w, period)
        if worker_util >= worker.baseline_utility:
            # Higher utility above baseline = higher acceptance probability
            # Simple sigmoid-like function
            excess_utility = worker_util - worker.baseline_utility
            return min(1.0, 0.5 + excess_utility * 2.0)
        else:
            return 0.0
    
    def _match_period(self, period: int, unmatched_workers: Set[int], unmatched_firms: Set[int]) -> PeriodResults:
        """
        Perform matching for a single period.
        
        Uses a simple greedy matching algorithm where headhunters propose
        matches based on their utility, and workers/firms accept based on
        their utilities.
        """
        matches: List[Match] = []
        matched_workers = set()
        matched_firms = set()
        
        # Generate all possible matches through headhunters
        potential_matches: List[tuple[Headhunter, Firm, Worker, float]] = []
        
        for headhunter in self.headhunters:
            for firm_id in headhunter.firm_ids:
                if firm_id not in unmatched_firms or firm_id in matched_firms:
                    continue
                firm = self.firm_dict[firm_id]
                
                for worker_id in headhunter.worker_ids:
                    if worker_id not in unmatched_workers or worker_id in matched_workers:
                        continue
                    worker = self.worker_dict[worker_id]
                    
                    # Check if headhunter can match this pair
                    if not headhunter.can_match(firm_id, worker_id):
                        continue
                    
                    # Compute utilities
                    observed_quality = self._get_observed_quality(worker, period)
                    firm_util = firm.utility(observed_quality, self.delta_f, period)
                    acceptance_prob = self._compute_acceptance_probability(worker, firm, period)
                    eta = self._compute_eta(firm, worker)
                    headhunter_util = headhunter.utility(firm, worker, acceptance_prob, eta, period)
                    
                    # Only consider if worker would accept
                    if acceptance_prob > 0:
                        potential_matches.append((headhunter, firm, worker, headhunter_util))
        
        # Sort by headhunter utility (descending) - headhunters prioritize high-utility matches
        potential_matches.sort(key=lambda x: x[3], reverse=True)
        
        # Greedily match
        for headhunter, firm, worker, headhunter_util in potential_matches:
            if worker.worker_id in matched_workers or firm.firm_id in matched_firms:
                continue
            
            # Worker decides whether to accept
            acceptance_prob = self._compute_acceptance_probability(worker, firm, period)
            if self.rng.random() < acceptance_prob:
                # Match is made
                observed_quality = self._get_observed_quality(worker, period)
                worker_util = worker.utility(firm.rank, self.delta_w, period)
                firm_util = firm.utility(observed_quality, self.delta_f, period)
                eta = self._compute_eta(firm, worker)
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
                matched_workers.add(worker.worker_id)
                matched_firms.add(firm.firm_id)
        
        unmatched_workers_list = [w for w in unmatched_workers if w not in matched_workers]
        unmatched_firms_list = [f for f in unmatched_firms if f not in matched_firms]
        
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
