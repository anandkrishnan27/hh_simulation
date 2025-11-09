from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Worker:
    worker_id: int
    skill: float

    def utility(self, wage: float, search_cost: float = 0.0) -> float:
        return wage - search_cost


@dataclass
class Firm:
    firm_id: int
    productivity: float
    base_wage: float

    def offered_wage(self, worker: Worker) -> float:
        return self.base_wage + 0.5 * worker.skill + 0.5 * self.productivity


@dataclass
class Intermediary:
    intermediary_id: int
    fee_rate: float = 0.10
    bias_towards_high_productivity: float = 0.20

    def fee(self, wage: float) -> float:
        return self.fee_rate * wage

    def adjust_offer(self, worker: Worker, firm: Firm, wage: float) -> float:
        bias = self.bias_towards_high_productivity * firm.productivity
        adjusted = wage - self.fee(wage) + 0.01 * bias
        return max(0.0, adjusted)


