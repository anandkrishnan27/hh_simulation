from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set, Dict, Tuple, Literal
import random
import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import poisson

from .agents import Worker, Firm, Headhunter, Agent


@dataclass
class Welfare:
    """Welfare calculations for a set of matches."""
    headhunter_welfare: float
    firm_welfare: float
    worker_welfare: float
    match_welfare: float  # Total welfare (sum of all three)

@dataclass
class Match:
    """Represents a match between a worker/agent and firm."""
    worker_id: int  # Worker ID (or agent ID in period 0, which maps to worker in period 1)
    firm_id: int
    period: int  # 0 for early phase, 1 for regular phase
    headhunter_id: Optional[int] = None
    worker_utility: float = 0.0
    firm_utility: float = 0.0
    headhunter_utility: float = 0.0
    expected_quality: float = 0.0  # Expected quality for period 0, true quality for period 1


@dataclass
class PeriodResults:
    """Results from a single period."""
    period: int
    matches: List[Match]
    unmatched_workers: List[int]
    unmatched_firms: List[int]


class Market:
    """
    Two-period market simulation with common preferences.
    
    Common preferences assumption:
    - All firms have the same ordinal preferences over workers (ranked by quality q_j)
    - All workers have the same ordinal preferences over firms (ranked by value v(i))
    - Each worker provides the same utility to every firm: u_f(w_j) = q_j for all firms f
    - Each firm provides the same utility to every worker: U_w(f_i) = v(i) for all workers w
    
    Period 0 (early phase): Agents have probability distributions over eventual rankings
    Period 1 (regular phase): Agents resolve to workers with true quality revealed
    """
    
    def __init__(
        self,
        workers: List[Worker],
        firms: List[Firm],
        headhunters: List[Headhunter],
        gamma: float = 0.3,  # γ: Outside option scaling factor (v_j(outside) = γ * max_f v(f))
        alpha: float = 0.5,  # α: Headhunter utility weight (u_h = α·μ + (1-α)·η)
        matching_algorithm: Literal["enumerative", "hungarian"] = "hungarian",  # Algorithm for headhunter matching
        rng: Optional[random.Random] = None,
        np_rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.workers = workers
        self.firms = firms
        self.headhunters = headhunters
        self.gamma = gamma
        self.alpha = alpha
        self.matching_algorithm = matching_algorithm
        self.rng = rng if rng is not None else random.Random()
        self.np_rng = np_rng if np_rng is not None else np.random.default_rng()
        
        # Sort workers by quality (descending) - common ranking for all firms
        # Sort firms by prestige (ascending) - common ranking for all workers
        self.workers.sort(key=lambda w: w.quality, reverse=True)
        self.firms.sort(key=lambda f: f.prestige)
        
        # Create lookup dictionaries
        self.worker_dict = {w.worker_id: w for w in self.workers}
        self.firm_dict = {f.firm_id: f for f in self.firms}
        
        # Compute baseline utility for each worker: γ * (quality / max_quality) * max_f v(f)
        # Since all workers have common preferences, max_f v(f) is the same for all workers
        max_firm_value = max(f.value for f in self.firms) if self.firms else 0.0
        self.v_max = max_firm_value  # v_max = max_f v(f) for headhunter payment calculation
        max_quality = max(w.quality for w in self.workers) if self.workers else 1.0
        
        # Set baseline utility for each worker based on their quality
        # Higher quality workers have better outside options (common preferences maintained)
        for worker in self.workers:
            quality_ratio = worker.quality / max_quality if max_quality > 0 else 0.0
            worker.baseline_utility = self.gamma * quality_ratio * max_firm_value
        
        # Set baseline utility for each firm based on their value
        # Higher value firms have better outside options
        max_firm_value_for_baseline = max(f.value for f in self.firms) if self.firms else 1.0
        for firm in self.firms:
            value_ratio = firm.value / max_firm_value_for_baseline if max_firm_value_for_baseline > 0 else 0.0
            firm.baseline_utility = self.gamma * value_ratio * max_quality
        
        # Create agents for period 0 (will be created in _create_agents)
        self.agents: List[Agent] = []
        self.agent_dict: Dict[int, Agent] = {}
        # Mapping from agent_id to worker_id (resolved in period 1)
        self.agent_to_worker: Dict[int, int] = {}
        # Mapping from headhunter_id to set of agent_ids accessible in period 0
        self.headhunter_agent_ids: Dict[int, Set[int]] = {}
    
    @staticmethod
    def random_market(
        num_workers: int = 50,
        num_firms: int = 20,
        num_headhunters: int = 5,
        gamma: float = 0.5,
        alpha: float = 0.5,
        matching_algorithm: Literal["enumerative", "hungarian"] = "hungarian",
        seed: Optional[int] = None,
    ) -> "Market":
        """Create a random market with specified parameters."""
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)
        
        # Create workers with random quality, then sort by quality
        # Baseline utility is computed in Market.__init__ as γ * max_f v(f)
        workers = []
        for i in range(num_workers):
            # Generate random quality values
            quality = np_rng.uniform(0.0, 1.0)
            workers.append(Worker(worker_id=i, quality=quality))
        
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
                )
            )
        
        return Market(
            workers=workers,
            firms=firms,
            headhunters=headhunters,
            gamma=gamma,
            alpha=alpha,
            matching_algorithm=matching_algorithm,
            rng=rng,
            np_rng=np_rng,
        )
    
    def _create_agents(self, poisson_lambda: float = 0.5) -> None:
        """
        Create agents with probability distributions over rankings for period 0.
        Each agent a_j is assigned to a distinct worker and has a truncated Poisson
        distribution centered around that worker's rank, ensuring no ties.
        
        Args:
            poisson_lambda: Lambda parameter for Poisson distribution (controls spread).
                           Smaller values = more concentrated around assigned rank.
                           Default 0.5 ensures P(distance=0) > P(distance=1).
        """
        num_workers = len(self.workers)
        self.agents = []
        self.agent_dict = {}
        
        # Assign each agent to a distinct worker (one-to-one mapping)
        # Shuffle to randomize the assignment
        worker_assignments = list(range(num_workers))
        self.np_rng.shuffle(worker_assignments)
        
        for agent_id, assigned_worker_rank in enumerate(worker_assignments):
            # Create truncated Poisson distribution centered at assigned rank
            # Probability decreases with distance from assigned rank
            
            # Compute Poisson probabilities for each rank
            probs = np.zeros(num_workers)
            for rank_idx in range(num_workers):
                # Distance from assigned rank
                distance = abs(rank_idx - assigned_worker_rank)
                # Poisson PMF: P(k; λ) = (λ^k * e^(-λ)) / k!
                # We use distance as k, so probability decreases with distance
                # Use scipy.stats.poisson.pmf to avoid overflow with large factorials
                prob = poisson.pmf(distance, poisson_lambda)
                probs[rank_idx] = prob
            
            # Normalize to ensure probabilities sum to 1 (truncated Poisson)
            probs = probs / np.sum(probs)
            
            agent = Agent(agent_id=agent_id, rank_distribution=probs, assigned_worker_rank=assigned_worker_rank)
            self.agents.append(agent)
            self.agent_dict[agent_id] = agent
    
    def _resolve_agent_to_worker(self, agent_id: int) -> int:
        """
        Resolve agent to worker in period 1.
        
        Uses the assigned worker rank to ensure no ties (each agent resolves to a distinct worker).
        The probability distribution is used for expected utility calculations in period 0,
        but resolution is deterministic to guarantee one-to-one mapping.
        
        Returns the worker_id that the agent becomes.
        """
        if agent_id in self.agent_to_worker:
            return self.agent_to_worker[agent_id]
        
        agent = self.agent_dict[agent_id]
        # Use assigned worker rank to ensure no ties
        # assigned_worker_rank is 0-indexed, so workers[assigned_worker_rank] is the assigned worker
        worker_id = self.workers[agent.assigned_worker_rank].worker_id
        self.agent_to_worker[agent_id] = worker_id
        return worker_id
    
    def _get_headhunter_agent_ids(self, headhunter: Headhunter) -> Set[int]:
        """
        Get the set of agent IDs that a headhunter has access to in period 0.
        
        A headhunter has access to agents whose assigned worker (via assigned_worker_rank)
        has a worker_id that is in the headhunter's worker_ids set.
        
        Args:
            headhunter: The headhunter to get agent IDs for
            
        Returns:
            Set of agent IDs accessible to this headhunter in period 0
        """
        if not self.agents:
            self._create_agents()
        
        accessible_agent_ids = set()
        for agent in self.agents:
            # Get the worker_id that this agent is assigned to
            assigned_worker_id = self.workers[agent.assigned_worker_rank].worker_id
            # If this worker_id is in the headhunter's worker_ids, the agent is accessible
            if assigned_worker_id in headhunter.worker_ids:
                accessible_agent_ids.add(agent.agent_id)
        
        return accessible_agent_ids
    
    def _get_headhunter_worker_ids_from_agents(self, headhunter_id: int) -> Set[int]:
        """
        Get the set of worker IDs that a headhunter has access to in period 1.
        
        This is determined by resolving the agents that the headhunter had access to
        in period 0 to their corresponding workers.
        
        Args:
            headhunter_id: The headhunter ID to get worker IDs for
            
        Returns:
            Set of worker IDs accessible to this headhunter in period 1
        """
        if headhunter_id not in self.headhunter_agent_ids:
            # If we don't have agent IDs for this headhunter, return empty set
            return set()
        
        accessible_worker_ids = set()
        for agent_id in self.headhunter_agent_ids[headhunter_id]:
            worker_id = self._resolve_agent_to_worker(agent_id)
            accessible_worker_ids.add(worker_id)
        
        return accessible_worker_ids
    
    def _can_match_agent(self, headhunter: Headhunter, firm_id: int, agent_id: int) -> bool:
        """
        Check if a headhunter can match a firm and agent in period 0.
        
        This checks:
        1. The firm_id is in the headhunter's firm_ids
        2. The agent_id corresponds to an agent whose assigned worker is in the headhunter's worker_ids
        
        Args:
            headhunter: The headhunter
            firm_id: The firm ID
            agent_id: The agent ID
            
        Returns:
            True if the headhunter can match this firm and agent
        """
        # Check firm access
        if firm_id not in headhunter.firm_ids:
            return False
        
        # Check agent access - agent must be in the headhunter's accessible agents
        # Get agent IDs if not already stored
        if headhunter.headhunter_id not in self.headhunter_agent_ids:
            self.headhunter_agent_ids[headhunter.headhunter_id] = self._get_headhunter_agent_ids(headhunter)
        
        return agent_id in self.headhunter_agent_ids[headhunter.headhunter_id]
    
    def _enumerate_optimal_matching_agent(
        self,
        headhunter: Headhunter,
        accessible_agents: List[int],
        accessible_firms: List[int],
    ) -> Dict[int, int]:
        """
        Enumerate all possible matchings and find the one that maximizes headhunter total utility.
        
        Returns a dictionary {agent_id: firm_id} representing the optimal matching.
        """
        if not accessible_agents or not accessible_firms:
            return {}
        
        # Filter to only valid pairs (respecting can_match constraint)
        valid_pairs = []
        for agent_id in accessible_agents:
            for firm_id in accessible_firms:
                if self._can_match_agent(headhunter, firm_id, agent_id):
                    agent = self.agent_dict[agent_id]
                    firm = self.firm_dict[firm_id]
                    utility = headhunter.utility_agent(firm, agent, self.workers, self.v_max, self.alpha)
                    valid_pairs.append((agent_id, firm_id, utility))
        
        if not valid_pairs:
            return {}
        
        # Enumerate all possible matchings using recursive backtracking
        best_matching: Dict[int, int] = {}
        max_utility = [float('-inf')]  # Use list to allow modification in nested function
        
        def enumerate_recursive(
            remaining_pairs: List[Tuple[int, int, float]],
            current_matching: Dict[int, int],
            current_utility: float,
            used_agents: Set[int],
            used_firms: Set[int],
        ) -> None:
            """Recursively enumerate all valid matchings."""
            # Update best if current is better
            if current_utility > max_utility[0]:
                best_matching.clear()
                best_matching.update(current_matching)
                max_utility[0] = current_utility
            
            # Try adding each remaining valid pair
            for idx, (agent_id, firm_id, utility) in enumerate(remaining_pairs):
                if agent_id not in used_agents and firm_id not in used_firms:
                    # Add this pair to matching
                    current_matching[agent_id] = firm_id
                    used_agents.add(agent_id)
                    used_firms.add(firm_id)
                    
                    # Recurse with remaining pairs and updated utility
                    enumerate_recursive(
                        remaining_pairs[idx + 1:],
                        current_matching,
                        current_utility + utility,
                        used_agents,
                        used_firms,
                    )
                    
                    # Backtrack
                    current_matching.pop(agent_id)
                    used_agents.remove(agent_id)
                    used_firms.remove(firm_id)
        
        # Start enumeration
        enumerate_recursive(valid_pairs, {}, 0.0, set(), set())
        
        return best_matching
    
    def _enumerate_optimal_matching_worker(
        self,
        headhunter: Headhunter,
        accessible_workers: List[int],
        accessible_firms: List[int],
    ) -> Dict[int, int]:
        """
        Enumerate all possible matchings and find the one that maximizes headhunter total utility.
        
        Returns a dictionary {worker_id: firm_id} representing the optimal matching.
        """
        if not accessible_workers or not accessible_firms:
            return {}
        
        # Filter to only valid pairs (respecting can_match constraint)
        valid_pairs = []
        for worker_id in accessible_workers:
            for firm_id in accessible_firms:
                if headhunter.can_match(firm_id, worker_id):
                    worker = self.worker_dict[worker_id]
                    firm = self.firm_dict[firm_id]
                    utility = headhunter.utility_worker(firm, worker, self.v_max, self.alpha)
                    valid_pairs.append((worker_id, firm_id, utility))
        
        if not valid_pairs:
            return {}
        
        # Enumerate all possible matchings using recursive backtracking
        best_matching: Dict[int, int] = {}
        max_utility = [float('-inf')]  # Use list to allow modification in nested function
        
        def enumerate_recursive(
            remaining_pairs: List[Tuple[int, int, float]],
            current_matching: Dict[int, int],
            current_utility: float,
            used_workers: Set[int],
            used_firms: Set[int],
        ) -> None:
            """Recursively enumerate all valid matchings."""
            # Update best if current is better
            if current_utility > max_utility[0]:
                best_matching.clear()
                best_matching.update(current_matching)
                max_utility[0] = current_utility
            
            # Try adding each remaining valid pair
            for idx, (worker_id, firm_id, utility) in enumerate(remaining_pairs):
                if worker_id not in used_workers and firm_id not in used_firms:
                    # Add this pair to matching
                    current_matching[worker_id] = firm_id
                    used_workers.add(worker_id)
                    used_firms.add(firm_id)
                    
                    # Recurse with remaining pairs and updated utility
                    enumerate_recursive(
                        remaining_pairs[idx + 1:],
                        current_matching,
                        current_utility + utility,
                        used_workers,
                        used_firms,
                    )
                    
                    # Backtrack
                    current_matching.pop(worker_id)
                    used_workers.remove(worker_id)
                    used_firms.remove(firm_id)
        
        # Start enumeration
        enumerate_recursive(valid_pairs, {}, 0.0, set(), set())
        
        return best_matching
    
    def _hungarian_optimal_matching_agent(
        self,
        headhunter: Headhunter,
        accessible_agents: List[int],
        accessible_firms: List[int],
    ) -> Dict[int, int]:
        """
        Find optimal matching using Hungarian algorithm (much faster than enumeration).
        
        Returns a dictionary {agent_id: firm_id} representing the optimal matching.
        """
        if not accessible_agents or not accessible_firms:
            return {}
        
        # Build cost matrix: rows = agents, columns = firms
        # Use negative utilities because Hungarian algorithm minimizes cost
        num_agents = len(accessible_agents)
        num_firms = len(accessible_firms)
        max_size = max(num_agents, num_firms)
        
        # Create square matrix (pad with zeros for unmatched pairs)
        cost_matrix = np.full((max_size, max_size), 0.0)
        
        # Map indices
        agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(accessible_agents)}
        firm_to_idx = {firm_id: idx for idx, firm_id in enumerate(accessible_firms)}
        
        # Fill in valid pairs with negative utility (we want to maximize, Hungarian minimizes)
        for agent_id in accessible_agents:
            for firm_id in accessible_firms:
                if self._can_match_agent(headhunter, firm_id, agent_id):
                    agent = self.agent_dict[agent_id]
                    firm = self.firm_dict[firm_id]
                    utility = headhunter.utility_agent(firm, agent, self.workers, self.v_max, self.alpha)
                    # Use negative because Hungarian minimizes, we want to maximize
                    cost_matrix[agent_to_idx[agent_id], firm_to_idx[firm_id]] = -utility
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Build matching dictionary (only include pairs with non-zero utility)
        matching = {}
        for row_idx, col_idx in zip(row_indices, col_indices):
            if row_idx < num_agents and col_idx < num_firms:
                agent_id = accessible_agents[row_idx]
                firm_id = accessible_firms[col_idx]
                # Only include if it's a valid match and has positive utility
                if self._can_match_agent(headhunter, firm_id, agent_id) and cost_matrix[row_idx, col_idx] < 0:
                    matching[agent_id] = firm_id
        
        return matching
    
    def _hungarian_optimal_matching_worker(
        self,
        headhunter: Headhunter,
        accessible_workers: List[int],
        accessible_firms: List[int],
    ) -> Dict[int, int]:
        """
        Find optimal matching using Hungarian algorithm (much faster than enumeration).
        
        Returns a dictionary {worker_id: firm_id} representing the optimal matching.
        """
        if not accessible_workers or not accessible_firms:
            return {}
        
        # Build cost matrix: rows = workers, columns = firms
        # Use negative utilities because Hungarian algorithm minimizes cost
        num_workers = len(accessible_workers)
        num_firms = len(accessible_firms)
        max_size = max(num_workers, num_firms)
        
        # Create square matrix (pad with zeros for unmatched pairs)
        cost_matrix = np.full((max_size, max_size), 0.0)
        
        # Map indices
        worker_to_idx = {worker_id: idx for idx, worker_id in enumerate(accessible_workers)}
        firm_to_idx = {firm_id: idx for idx, firm_id in enumerate(accessible_firms)}
        
        # Fill in valid pairs with negative utility (we want to maximize, Hungarian minimizes)
        for worker_id in accessible_workers:
            for firm_id in accessible_firms:
                if headhunter.can_match(firm_id, worker_id):
                    worker = self.worker_dict[worker_id]
                    firm = self.firm_dict[firm_id]
                    utility = headhunter.utility_worker(firm, worker, self.v_max, self.alpha)
                    # Use negative because Hungarian minimizes, we want to maximize
                    cost_matrix[worker_to_idx[worker_id], firm_to_idx[firm_id]] = -utility
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Build matching dictionary (only include pairs with non-zero utility)
        matching = {}
        for row_idx, col_idx in zip(row_indices, col_indices):
            if row_idx < num_workers and col_idx < num_firms:
                worker_id = accessible_workers[row_idx]
                firm_id = accessible_firms[col_idx]
                # Only include if it's a valid match and has positive utility
                if headhunter.can_match(firm_id, worker_id) and cost_matrix[row_idx, col_idx] < 0:
                    matching[worker_id] = firm_id
        
        return matching
    
    def _match_period(self, period: int, unmatched_workers: Set[int], unmatched_firms: Set[int]) -> PeriodResults:
        """
        Perform matching for a single period.
        
        Period 0: Uses agents with probability distributions
        Period 1: Uses workers with true quality
        
        Algorithm:
        1. Each headhunter finds optimal matching (using enumerative or Hungarian algorithm)
           that maximizes total headhunter utility over all accessible firms and workers/agents
        2. Firms observe all proposals and greedily choose best worker/agent (multiple firms can choose same worker/agent)
        3. Workers/Agents with multiple proposals choose best firm, then compare to baseline utility
        4. Matches are finalized (no reneging)
        """
        matches: List[Match] = []
        
        if period == 0:
            # Period 0: Use agents
            if not self.agents:
                self._create_agents()
            
            unmatched_agents = set(a.agent_id for a in self.agents if a.agent_id in unmatched_workers)
            
            # Step 1: Each headhunter proposes matches using enumerative algorithm
            # (finds optimal matching that maximizes total headhunter utility)
            headhunter_proposals: Dict[int, Dict[int, int]] = {}  # {headhunter_id: {agent_id: firm_id}}
            
            for headhunter in self.headhunters:
                # Get the set of agent IDs this headhunter has access to (based on worker_ids)
                all_accessible_agent_ids = self._get_headhunter_agent_ids(headhunter)
                # Store this for use in period 1
                self.headhunter_agent_ids[headhunter.headhunter_id] = all_accessible_agent_ids
                
                # Filter to only unmatched agents
                accessible_agents = [
                    a_id for a_id in all_accessible_agent_ids 
                    if a_id in unmatched_agents
                ]
                accessible_firms = [
                    f_id for f_id in headhunter.firm_ids 
                    if f_id in unmatched_firms
                ]
                
                if not accessible_agents or not accessible_firms:
                    headhunter_proposals[headhunter.headhunter_id] = {}
                    continue
                
                # Find optimal matching using selected algorithm
                if self.matching_algorithm == "hungarian":
                    optimal_matching = self._hungarian_optimal_matching_agent(
                        headhunter, accessible_agents, accessible_firms
                    )
                else:  # enumerative
                    optimal_matching = self._enumerate_optimal_matching_agent(
                        headhunter, accessible_agents, accessible_firms
                    )
                
                headhunter_proposals[headhunter.headhunter_id] = optimal_matching
            
            # Step 2: Firms observe all proposals and greedily choose best agent
            firm_choices: Dict[int, Tuple[Agent, Headhunter]] = {}  # {firm_id: (agent, headhunter)}
            
            for headhunter_id, matching in headhunter_proposals.items():
                headhunter = next(h for h in self.headhunters if h.headhunter_id == headhunter_id)
                for agent_id, firm_id in matching.items():
                    if agent_id not in unmatched_agents or firm_id not in unmatched_firms:
                        continue
                    
                    agent = self.agent_dict[agent_id]
                    firm = self.firm_dict[firm_id]
                    firm_util = firm.utility(agent=agent, workers=self.workers, t=period)
                    
                    # Firm chooses best proposal (highest utility)
                    if firm_id not in firm_choices:
                        firm_choices[firm_id] = (agent, headhunter)
                    else:
                        current_agent, _ = firm_choices[firm_id]
                        current_util = firm.utility(agent=current_agent, workers=self.workers, t=period)
                        if firm_util > current_util:
                            firm_choices[firm_id] = (agent, headhunter)
            
            # Filter firm choices: firms only accept if utility >= baseline
            # For period 0, use expected baseline (same as workers)
            firm_choices_filtered: Dict[int, Tuple[Agent, Headhunter]] = {}
            for firm_id, (agent, headhunter) in firm_choices.items():
                firm = self.firm_dict[firm_id]
                firm_util = firm.utility(agent=agent, workers=self.workers, t=period)
                # Firm accepts if utility >= baseline
                if firm_util >= firm.baseline_utility:
                    firm_choices_filtered[firm_id] = (agent, headhunter)
            firm_choices = firm_choices_filtered
            
            # Step 3: Agents receive proposals and choose best, then compare to baseline
            # For agents, we need to compute expected worker utility
            agent_proposals: Dict[int, List[Tuple[Firm, Headhunter, float]]] = {}
            # {agent_id: [(firm, headhunter, expected_worker_utility), ...]}
            
            for firm_id, (agent, headhunter) in firm_choices.items():
                firm = self.firm_dict[firm_id]
                # Worker utility doesn't depend on agent: U_w(f_i, t) = v(i)
                worker_util = firm.value
                
                if agent.agent_id not in agent_proposals:
                    agent_proposals[agent.agent_id] = []
                agent_proposals[agent.agent_id].append((firm, headhunter, worker_util))
            
            # Agents choose best proposal and accept if expected utility >= expected baseline
            for agent_id, proposals in agent_proposals.items():
                agent = self.agent_dict[agent_id]
                
                # Agent greedily chooses best proposal (highest worker utility)
                best_proposal = max(proposals, key=lambda x: x[2])  # x[2] is worker_utility
                firm, headhunter, worker_util = best_proposal
                
                # Expected baseline utility: sum_k p_{jk} * baseline_utility(w_k)
                # where baseline_utility(w_k) = γ * (q_k / max_q) * max_f v(f)
                expected_baseline = sum(
                    prob * self.workers[rank_idx].baseline_utility
                    for rank_idx, prob in enumerate(agent.rank_distribution)
                    if rank_idx < len(self.workers)
                )
                
                # Agent accepts if utility >= expected baseline
                if worker_util < expected_baseline:
                    continue
                
                # Match is finalized
                firm_util = firm.utility(agent=agent, workers=self.workers, t=period)
                headhunter_util = headhunter.utility_agent(firm, agent, self.workers, self.v_max, self.alpha)
                
                # Expected quality for display
                expected_quality = sum(
                    prob * self.workers[rank_idx].quality
                    for rank_idx, prob in enumerate(agent.rank_distribution)
                    if rank_idx < len(self.workers)
                )
                
                matches.append(
                    Match(
                        worker_id=agent.agent_id,  # Store agent_id, will resolve to worker_id in period 1
                        firm_id=firm.firm_id,
                        period=period,
                        headhunter_id=headhunter.headhunter_id,
                        worker_utility=worker_util,
                        firm_utility=firm_util,
                        headhunter_utility=headhunter_util,
                        expected_quality=expected_quality,
                    )
                )
            
            # Track unmatched
            matched_agent_ids = {m.worker_id for m in matches}
            matched_firm_ids = {m.firm_id for m in matches}
            unmatched_workers_list = [w for w in unmatched_workers if w not in matched_agent_ids]
            unmatched_firms_list = [f for f in unmatched_firms if f not in matched_firm_ids]
            
        else:
            # Period 1: Use workers
            # Step 1: Each headhunter proposes matches using enumerative algorithm
            # (finds optimal matching that maximizes total headhunter utility)
            headhunter_proposals: Dict[int, Dict[int, int]] = {}  # {headhunter_id: {worker_id: firm_id}}
            
            for headhunter in self.headhunters:
                # Get the set of worker IDs this headhunter has access to in period 1
                # This is determined by resolving the agents it had access to in period 0
                all_accessible_worker_ids = self._get_headhunter_worker_ids_from_agents(headhunter.headhunter_id)
                
                # Filter to only unmatched workers
                accessible_workers = [
                    w_id for w_id in all_accessible_worker_ids 
                    if w_id in unmatched_workers
                ]
                accessible_firms = [
                    f_id for f_id in headhunter.firm_ids 
                    if f_id in unmatched_firms
                ]
                
                if not accessible_workers or not accessible_firms:
                    headhunter_proposals[headhunter.headhunter_id] = {}
                    continue
                
                # Find optimal matching using selected algorithm
                if self.matching_algorithm == "hungarian":
                    optimal_matching = self._hungarian_optimal_matching_worker(
                        headhunter, accessible_workers, accessible_firms
                    )
                else:  # enumerative
                    optimal_matching = self._enumerate_optimal_matching_worker(
                        headhunter, accessible_workers, accessible_firms
                    )
                
                headhunter_proposals[headhunter.headhunter_id] = optimal_matching
            
            # Step 2: Firms observe all proposals and greedily choose best worker
            firm_choices: Dict[int, Tuple[Worker, Headhunter]] = {}  # {firm_id: (worker, headhunter)}
            
            for headhunter_id, matching in headhunter_proposals.items():
                headhunter = next(h for h in self.headhunters if h.headhunter_id == headhunter_id)
                for worker_id, firm_id in matching.items():
                    if worker_id not in unmatched_workers or firm_id not in unmatched_firms:
                        continue
                    
                    worker = self.worker_dict[worker_id]
                    firm = self.firm_dict[firm_id]
                    firm_util = firm.utility(worker_quality=worker.quality, t=period)
                    
                    # Firm chooses best proposal (highest utility)
                    if firm_id not in firm_choices:
                        firm_choices[firm_id] = (worker, headhunter)
                    else:
                        current_worker, _ = firm_choices[firm_id]
                        current_util = firm.utility(worker_quality=current_worker.quality, t=period)
                        if firm_util > current_util:
                            firm_choices[firm_id] = (worker, headhunter)
            
            # Filter firm choices: firms only accept if utility >= baseline
            # For period 1, heavily decrease the outside option (multiply by 0.1)
            firm_choices_filtered: Dict[int, Tuple[Worker, Headhunter]] = {}
            for firm_id, (worker, headhunter) in firm_choices.items():
                firm = self.firm_dict[firm_id]
                firm_util = firm.utility(worker_quality=worker.quality, t=period)
                baseline_threshold = firm.baseline_utility * 0.1 if period == 1 else firm.baseline_utility
                # Firm accepts if utility >= baseline
                if firm_util >= baseline_threshold:
                    firm_choices_filtered[firm_id] = (worker, headhunter)
            firm_choices = firm_choices_filtered
            
            # Step 3: Workers receive proposals and choose best, then compare to baseline
            worker_proposals: Dict[int, List[Tuple[Firm, Headhunter, float]]] = {}
            # {worker_id: [(firm, headhunter, worker_utility), ...]}
            
            for firm_id, (worker, headhunter) in firm_choices.items():
                firm = self.firm_dict[firm_id]
                worker_util = worker.utility(firm.value, period)
                
                if worker.worker_id not in worker_proposals:
                    worker_proposals[worker.worker_id] = []
                worker_proposals[worker.worker_id].append((firm, headhunter, worker_util))
            
            # Workers choose best proposal and accept if utility >= baseline
            for worker_id, proposals in worker_proposals.items():
                worker = self.worker_dict[worker_id]
                
                # Worker greedily chooses best proposal (highest worker utility)
                best_proposal = max(proposals, key=lambda x: x[2])  # x[2] is worker_utility
                firm, headhunter, worker_util = best_proposal
                
                # Worker accepts if utility >= baseline (outside option)
                # For period 1, heavily decrease the outside option (multiply by 0.1)
                baseline_threshold = worker.baseline_utility * 0.1 if period == 1 else worker.baseline_utility
                if worker_util < baseline_threshold:
                    continue
                
                # Match is finalized
                firm_util = firm.utility(worker_quality=worker.quality, t=period)
                headhunter_util = headhunter.utility_worker(firm, worker, self.v_max, self.alpha)
                
                matches.append(
                    Match(
                        worker_id=worker.worker_id,
                        firm_id=firm.firm_id,
                        period=period,
                        headhunter_id=headhunter.headhunter_id,
                        worker_utility=worker_util,
                        firm_utility=firm_util,
                        headhunter_utility=headhunter_util,
                        expected_quality=worker.quality,
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
    
    def calculate_welfare(self, matches: List[Match]) -> Welfare:
        """
        Calculate welfare from a list of matches.
        
        Returns:
            Welfare object with headhunter_welfare, firm_welfare, worker_welfare, and match_welfare
        """
        headhunter_welfare = sum(m.headhunter_utility for m in matches)
        firm_welfare = sum(m.firm_utility for m in matches)
        worker_welfare = sum(m.worker_utility for m in matches)
        match_welfare = firm_welfare + worker_welfare
        
        return Welfare(
            headhunter_welfare=headhunter_welfare,
            firm_welfare=firm_welfare,
            worker_welfare=worker_welfare,
            match_welfare=match_welfare,
        )
    
    def run(self) -> List[PeriodResults]:
        """
        Run the two-period simulation.
        
        Period 0: Agents with probability distributions over rankings
        Period 1: Agents resolve to workers, true quality revealed
        
        Returns results for both periods (t=0 and t=1).
        """
        results: List[PeriodResults] = []
        
        # Create agents for period 0
        self._create_agents()
        
        # Period 0: Early phase (use agent IDs, which match worker IDs)
        unmatched_agents = set(a.agent_id for a in self.agents)
        unmatched_firms = set(f.firm_id for f in self.firms)
        period_0_results = self._match_period(0, unmatched_agents, unmatched_firms)
        results.append(period_0_results)
        
        # Resolve agents to workers for period 1
        # Matched agents in period 0 are resolved to their worker assignments
        matched_agent_ids_in_0 = set(m.worker_id for m in period_0_results.matches)
        matched_firms_in_0 = set(m.firm_id for m in period_0_results.matches)
        
        # Resolve all matched agents to workers
        matched_worker_ids_in_0 = set()
        for agent_id in matched_agent_ids_in_0:
            worker_id = self._resolve_agent_to_worker(agent_id)
            matched_worker_ids_in_0.add(worker_id)
        
        # Period 1: Regular phase (only unmatched workers/firms participate)
        # Note: In period 1, we use actual worker IDs
        all_worker_ids = set(w.worker_id for w in self.workers)
        unmatched_workers_1 = all_worker_ids - matched_worker_ids_in_0
        unmatched_firms_1 = unmatched_firms - matched_firms_in_0
        
        period_1_results = self._match_period(1, unmatched_workers_1, unmatched_firms_1)
        results.append(period_1_results)
        
        return results
