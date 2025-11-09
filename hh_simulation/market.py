from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import random

from .agents import Worker, Firm, Intermediary


@dataclass
class MatchResult:
    worker_id: int
    firm_id: Optional[int]
    wage: float


class Market:
    def __init__(
        self,
        workers: List[Worker],
        firms: List[Firm],
        intermediary: Optional[Intermediary] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.workers = workers
        self.firms = firms
        self.intermediary = intermediary
        self.rng = rng if rng is not None else random.Random()

    @staticmethod
    def random_market(num_workers: int, num_firms: int, seed: Optional[int] = None) -> "Market":
        rng = random.Random(seed)
        workers = [Worker(worker_id=i, skill=rng.uniform(0.0, 1.0)) for i in range(num_workers)]
        firms = [
            Firm(firm_id=i, productivity=rng.uniform(0.0, 1.0), base_wage=rng.uniform(0.5, 1.5))
            for i in range(num_firms)
        ]
        intermediary = Intermediary(intermediary_id=0, fee_rate=0.10, bias_towards_high_productivity=0.20)
        return Market(workers=workers, firms=firms, intermediary=intermediary, rng=rng)

    def step(self) -> List[MatchResult]:
        results: List[MatchResult] = []
        for worker in self.workers:
            best_firm_id: Optional[int] = None
            best_utility = float("-inf")
            best_wage = 0.0
            for firm in self.firms:
                wage = firm.offered_wage(worker)
                if self.intermediary is not None:
                    wage = self.intermediary.adjust_offer(worker, firm, wage)
                utility = worker.utility(wage)
                if utility > best_utility:
                    best_utility = utility
                    best_firm_id = firm.firm_id
                    best_wage = wage
            results.append(MatchResult(worker_id=worker.worker_id, firm_id=best_firm_id, wage=best_wage))
        return results

    def run(self, steps: int = 1) -> List[List[MatchResult]]:
        history: List[List[MatchResult]] = []
        for _ in range(steps):
            history.append(self.step())
        return history


