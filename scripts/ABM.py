# Author: Charles Brummitt, Kenan Huremovic, Paolo Pin,
#         Matthew Bonds, Fernando Vega-Redondo
"""Agent-based model of economic complexity and contagious disruptions.

This model is used to create Figure 7 in the paper "Contagious disruptions and
complexity traps in economic development" (2017) by the above authors.
"""
import numpy as np
import itertools
import scipy
from operator import itemgetter
from functools import lru_cache
from collections import Counter


class Strategy(object):
    """A strategy is a pair of non-negative integers representing the number
    of attempted inputs (`n_inputs_attempted`, called `m` in the paper) and
    the number of inputs needed in order to successfully produce
    (`n_inputs_needed`, called `tau` in the paper).
    """
    def __init__(self, n_inputs_attempted=None, n_inputs_needed=None):
        self.n_inputs_attempted = n_inputs_attempted
        self.n_inputs_needed = n_inputs_needed

    def __repr__(self):
        template = 'Strategy(n_inputs_attempted={m}, n_inputs_needed={tau})'
        return template.format(m=self.n_inputs_attempted,
                               tau=self.n_inputs_needed)

    def as_dict(self):
        return {'n_inputs_attempted': self.n_inputs_attempted,
                'n_inputs_needed': self.n_inputs_needed}

    def update(self, n_inputs_attempted, n_inputs_needed):
        self.n_inputs_attempted = n_inputs_attempted
        self.n_inputs_needed = n_inputs_needed

    def customer_is_functional_after_production_attempt(
            self, customer_begins_functional, n_functional_inputs):
        # If no inputs are needed, then there is little to model, and the agent
        # does not become functional; see the discussion after equation 1 in
        # the paper.
        if self.n_inputs_needed == 0:
            return customer_begins_functional
        else:
            return (n_functional_inputs >= self.n_inputs_needed)


class Agent(object):
    """An economic agent who produces goods and services for other agents using
    inputs sourced from other agents.
    """
    id_generator = itertools.count()

    def __init__(self, is_functional, r=1.0, xi=0, economy=None):
        self._id = next(self.id_generator)
        self.strategy = Strategy()
        self.is_functional = bool(is_functional)
        self.suppliers = []
        self.customers = []
        self.amount_more_likely_choose_func_supplier = r
        self.pref_attachment_power = xi
        self.functional_suppliers_most_recent_attempt = []
        self.n_retained_suppliers = None
        self.time_between_successive_dysfunction = None
        self.economy = economy

    def __hash__(self):
        return self._id

    def __repr__(self):
        functional_status = (
            'functional' if self.is_functional else 'dysfunctional')
        template = ('Agent(id={id}, {status}, {strategy}, '
                    'r={r}), n_customers={n_customers})')
        return template.format(
            id=self._id,
            status=functional_status,
            strategy=self.strategy,
            n_customers=len(self.customers),
            r=self.amount_more_likely_choose_func_supplier)

    def out_degree(self):
        return len(self.customers)

    def in_degree(self):
        return self.strategy.n_inputs_attempted

    def update_strategy_and_suppliers(
            self, choose_suppliers_uniformly_at_random=False):
        """Update the number of inputs needed and attempted, find new suppliers,
        and return the the number of inputs needed and attempted as a
        dictionary.

        The Boolean optional argument `choose_suppliers_uniformly_at_random`
        determines whether suppliers are chosen with equal probability or
        via some other rule. `choose_suppliers_uniformly_at_random` is set to
        True when we initialize the model with preferential attachment because
        in the initial condition we do not have any links with which to use the
        rule for choosing new links."""
        self.update_complexity_and_buffer()
        self.find_new_suppliers(choose_suppliers_uniformly_at_random)
        return self.strategy

    def update_complexity_and_buffer(self):
        """Update the number of inputs needed in order to successfully produce
        (`tau` in the paper), the number of attempted inputs (`m` in the
        paper), and the choice of input sources.
        """
        F = self.economy.fraction_functional_agents()
        alpha, beta = self.economy.alpha, self.economy.beta
        self.strategy.update(
            *compute_optimal_n_inputs_attempted_and_complexity(F, alpha, beta))

    def find_new_suppliers(self, choose_suppliers_uniformly_at_random=False):
        """Find new input sources by preferentially choosing suppliers who
        were functional in your most recent attempt to produce.

        The method also records in `self.n_retained_suppliers` the number of
        suppliers who are retained after choosing new suppliers.

        The Boolean optional argument `choose_suppliers_uniformly_at_random`
        determines whether suppliers are chosen with equal probability or
        via another rule that creates some stickiness in links and preferential
        attachment to high out-degree nodes.
        """
        if choose_suppliers_uniformly_at_random:
            new_inputs = np.random.choice(
                self.economy.agents,
                size=self.strategy.n_inputs_attempted, replace=True)
        else:
            probabilities_to_choose_each_agent = (
                self.prob_choose_each_agent_as_supplier_sticky_pref_attach())

            new_inputs = np.random.choice(
                self.economy.agents,
                size=self.strategy.n_inputs_attempted,
                p=probabilities_to_choose_each_agent, replace=True)

        self.n_retained_suppliers = sum(
            new_supplier in self.suppliers for new_supplier in new_inputs)

        for supplier in self.suppliers:
            supplier.customers.remove(self)
        self.suppliers = new_inputs
        for supplier in self.suppliers:
            supplier.customers.append(self)
        return

    def prob_choose_each_agent_as_supplier_sticky_pref_attach(self):
        """Compute the chance of choosing each agent as a supplier with
        sticky links and preferential attachment (PA).

        The suppliers who were functional in your most recent attempt to
        produce are given the weight
            `self.amount_more_likely_choose_func_supplier * ((1 + deg) ** xi)`
        where `deg` is the supplier's out-degree and `xi` is the power in the
        definition of preferential attachment Everyone else has weight
        `(1 + deg) ** xi` where `deg` is their out-degree. Each new
        supplier is chosen independently and with replacement according to
        these weights.
        """
        all_agents = self.economy.agents
        r = self.amount_more_likely_choose_func_supplier

        weights = np.empty(len(all_agents))

        for i, agent in enumerate(all_agents):
            weights[i] = (1 + agent.out_degree())**self.pref_attachment_power
            if agent in self.functional_suppliers_most_recent_attempt:
                weights[i] *= r
        return weights / sum(weights)

    def attempt_to_produce(self, time=None):
        """Attempt to produce by sourcing inputs from suppliers.

        Remember the identities of the suppliers who were functional by storing
        their `_id` attribute in the list
        `functional_suppliers_most_recent_attempt`. If the agent is
        dysfunctional, then record the given time in the attribute
        `when_last_dysfunctional`.

        Returns
        -------
        success : bool
            Whether the agent succeeded in producing (i.e., got
            self.strategy.n_inputs_needed or more functional inputs from
            its suppliers).
        """
        n_functional_suppliers = sum(
            supplier.is_functional for supplier in self.suppliers)

        self.functional_suppliers_most_recent_attempt = [
            supplier
            for supplier in self.suppliers if supplier.is_functional]

        self.is_functional = (
            self.strategy.customer_is_functional_after_production_attempt(
                self.is_functional, n_functional_suppliers))

        return self.is_functional

    def is_vulnerable(self):
        """The agent would become dysfunctional if one more supplier were
        to become dysfunctional."""
        return (
            self.is_functional and
            (len(self.functional_suppliers_most_recent_attempt) ==
                self.strategy.n_inputs_needed))


# Memoize 4096 calls to this function because it is called frequently (whenever
# agents attempt to produce.
@lru_cache(maxsize=2**12)
def compute_optimal_n_inputs_attempted_and_complexity(
        fraction_functional_agents, alpha, beta):
    """Compute the optimal strategy (n_inputs_attempted and complexity) given
    the fraction of functional agents."""
    strategies = _strategies_that_could_be_best_response(
        fraction_functional_agents, alpha, beta)

    def strategy_to_expected_utility(m, tau):
        return _expected_utility(m, tau, fraction_functional_agents,
                                 alpha, beta)

    return maximizer_with_tiebreaker(
        strategies, objective_function=strategy_to_expected_utility,
        tiebreaker=sum)


def _strategies_that_could_be_best_response(
        fraction_functional_agents, alpha, beta):
    """Compute the set of strategies that could be a best response."""
    if fraction_functional_agents == 0:
        return [(0, 0)]
    elif fraction_functional_agents == 1:
        gamma = (alpha / beta) ** (1 / (beta - 1))
        gamma_floor = int(np.floor(gamma))
        gamma_ceil = int(np.ceil(gamma))
        return [(gamma_floor, gamma_floor), (gamma_ceil, gamma_ceil)]
    elif (0 < fraction_functional_agents < 1):
        max_possible_n_inputs_attempted = int(
            np.ceil(alpha ** (-1 / (1 - beta))))
        return [(m, tau)
                for m in range(max_possible_n_inputs_attempted + 1)
                for tau in range(0, m + 1)
                if (_equation_SI_10(m, tau, fraction_functional_agents,
                                    alpha, beta) or
                    (0 < tau and tau < m and m < tau**beta / alpha))]
    else:
        msg = "fraction_functional_agents = {} cannot be outside [0, 1]"
        raise ValueError(msg.format(fraction_functional_agents))


def _equation_SI_10(m, tau, F, alpha, beta):
    """Equation SI-10 in the paper."""
    product_log_factor = scipy.special.lambertw(
        ((1 / alpha)**(1 / (1 - beta)) * np.log(F)) / (beta - 1))
    return (
        (m == tau) and
        m < ((beta - 1) * product_log_factor / np.log(F)))


def _expected_utility(m, tau, F, alpha, beta):
    """Compute the expected utility of a given strategy in an economy with a
    certain amount of reliability."""
    assert m >= 0
    assert tau >= 0
    assert 0 <= F <= 1
    return _prob_success(m, tau, F) * tau ** beta - alpha * m


def _prob_success(m, tau, F):
    """Chance of successfully producing when drawing balls from an urn."""
    assert m >= 0
    assert tau >= 0
    assert 0 <= F <= 1

    if m == tau == 0:
        return 0
    else:
        binomial = scipy.stats.binom(n=m, p=F)
        chance_get_tau_or_more_successes = binomial.sf(tau - 1)
        return chance_get_tau_or_more_successes


def maximizer_with_tiebreaker(array, objective_function, tiebreaker):
    array_scores = [(a, objective_function(*a)) for a in array]
    max_score = max(array_scores, key=itemgetter(1))
    maximizers = [a for a, obj in array_scores if obj == max_score[1]]
    return min(maximizers, key=tiebreaker)


class Economy(object):
    """A collection of Agents and methods for updating a random Agent and for
    collecting information about the state of the economy.
    """
    def __init__(self, n_agents, initial_fraction_functional,
                 alpha, beta, r=1.0, L=1.0, exog_fail=0.0, xi=0):
        assert 0 <= initial_fraction_functional <= 1
        assert alpha > 0
        assert beta > 0
        assert n_agents > 1

        self.n_agents = n_agents
        self.initial_n_functional = int(initial_fraction_functional * n_agents)
        self.initial_fraction_functional = initial_fraction_functional
        self.amount_more_likely_choose_func_supplier = r
        self.xi = xi
        self.agents = [
            Agent(is_functional, economy=self, r=r, xi=xi) for is_functional in
            ([True] * self.initial_n_functional +
             [False] * (self.n_agents - self.initial_n_functional))]
        self.agent_set = set(self.agents)
        self.alpha = alpha
        self.beta = beta
        self.num_times_func_agent_more_likely_chosen = L
        self.exog_fail = exog_fail
        self.random_agents_queue = []
        # Diagnostics
        self.n_production_attempts = 0
        self.latest_best_response = None
        self.latest_producer = None
        self.n_customers_of_latest_producer = None
        self.n_vulnerable_customers_of_latest_producer = None
        self.n_suppliers_of_latest_producer = None
        self.change_in_n_functional_from_latest_attempt_to_produce = None
        self.n_retained_suppliers_from_latest_attempt_to_produce = None
        self.n_exogenous_failures = None
        self.time_between_successive_dysfunction = []
        self.n_functional = self.initial_n_functional
        self.initialize_strategies_and_network()

    def __repr__(self):
        template = ('Economy(n_agents={n}, initial_fraction_functional={F0}, '
                    'alpha={alpha}, beta={beta}, r={r}), xi={xi}')
        parameters = {'n': self.n_agents,
                      'r': self.amount_more_likely_choose_func_supplier,
                      'F0': self.initial_fraction_functional,
                      'alpha': self.alpha, 'beta': self.beta, 'xi': self.xi}
        return template.format(**parameters)

    def state(self):
        """Return a dictionary containing information about the state of the
        economy."""
        return {
            'latest_best_response_n_inputs_attempted':
                self.latest_best_response.n_inputs_attempted,
            'latest_best_response_n_inputs_needed':
                self.latest_best_response.n_inputs_needed,
            'n_customers_of_latest_producer':
                self.n_customers_of_latest_producer,
            'n_vulnerable_customers_of_latest_producer':
                self.n_vulnerable_customers_of_latest_producer,
            'n_suppliers_of_latest_producer':
                self.n_suppliers_of_latest_producer,
            'change_in_n_functional_from_latest_attempt_to_produce':
                self.change_in_n_functional_from_latest_attempt_to_produce,
            'n_functional': self.n_functional,
            'fraction_functional_agents': self.fraction_functional_agents()}

    def fraction_functional_agents(self):
        return self.n_functional / self.n_agents

    def total_input_attempts(self):
        """Compute the total number of (supplier, customer) relationships
        (i.e., total number of 'edges' or 'links') in the economy."""
        return sum(len(agent.suppliers) for agent in self.agents)

    def customer_supplier_functionality_count(self, per_input_attempt=False):
        """Compute the assortativity of the functionality of customers and
        suppliers (i.e., the fraction of (customer, supplier) pairs that are
        (functional, functional), (functional, dysfunctional), etc.)."""
        pair_functionality_counter = Counter()
        for customer in self.agents:
            for supplier in customer.suppliers:
                pair_functionality_counter[
                    (customer.is_functional, supplier.is_functional)] += 1
        if per_input_attempt:
            num_input_attempts = self.total_input_attempts()
            for key in pair_functionality_counter:
                pair_functionality_counter[key] /= num_input_attempts
        return pair_functionality_counter

    def initialize_strategies_and_network(self):
        """Initialize the strategies and network (i.e., the customer-supplier
        relationships) by choosing suppliers uniformly at random. We cannot use
        the preferential attachment rule until there is a network of links, so
        here we initialize the links to be chosen uniformly at random from
        all possible links (i.e., an Erdos-Renyi random graph)."""
        for agent in self.agents:
            self.latest_best_response = agent.update_strategy_and_suppliers(
                choose_suppliers_uniformly_at_random=True)
        return

    def update_one_step(self):
        """Update the strategy of a random agent, let it attempt to produce,
        and run exogenous failures that cause each agent to independently
        fail with probability exog_fail.

        Returns
        -------
        success : bool
            Whether the agent successfully produced.
        """
        success = (
            self.update_random_agent_strategy_inputs_and_attempt_to_produce())
        self.run_exogenous_failures()
        return success

    def run_exogenous_failures(self):
        """Each agent fails independently with probability `exog_fail`.

        Returns
        -------
        n_exogenos_failures: int
            The number of exogenous failures that occurred in this step.
        """
        if self.exog_fail <= 0:
            return 0

        functional_agents = [ag for ag in self.agents if ag.is_functional]
        assert len(functional_agents) == self.n_functional

        whether_each_functional_agent_fails_exogenously = (
            np.random.random_sample(size=self.n_functional) < self.exog_fail)
        for indx, fails_exogenously in enumerate(
                whether_each_functional_agent_fails_exogenously):
            if fails_exogenously:
                functional_agents[indx].is_functional = False
                self.n_functional -= 1
        self.n_exogenous_failures = sum(
            whether_each_functional_agent_fails_exogenously)
        return self.n_exogenous_failures

    def update_random_agent_strategy_inputs_and_attempt_to_produce(self):
        random_agent = self._get_random_agent()
        return self.update_agent_strategy_inputs_and_attempt_to_produce(
            random_agent)

    def _get_random_agent(self):
        """To speed thigns up, we select random agents 10000 at a time, and
        replenish this list when it is empty."""
        if len(self.random_agents_queue) == 0:
            self._replenish_random_agents_queue()
        return self.random_agents_queue.pop()

    def _replenish_random_agents_queue(self):
        if self.num_times_func_agent_more_likely_chosen == 1.0:
            size_of_queue = 10000
        else:
            size_of_queue = 1
        prob_choose_each_agent = self._probabilities_choose_each_agent()
        self.random_agents_queue = list(np.random.choice(
            self.agents, p=prob_choose_each_agent, replace=True,
            size=size_of_queue))

    def _probabilities_choose_each_agent(self):
        probabilities = np.empty(self.n_agents)
        for i, agent in enumerate(self.agents):
            if agent.is_functional:
                probabilities[i] = self.num_times_func_agent_more_likely_chosen
            else:
                probabilities[i] = 1

        return probabilities / np.sum(probabilities)

    def update_agent_strategy_inputs_and_attempt_to_produce(self, agent):
        """Update an agent's best response and suppliers, and have it attempt
        to produce.

        Returns
        -------
        success : bool
            Whether the agent successfully produced.
        """
        assert agent in self.agents

        self.record_diagnostics_right_before_production_attempt(agent)
        self.latest_best_response = (agent.update_strategy_and_suppliers())

        agent_was_functional = agent.is_functional
        success = agent.attempt_to_produce(self.n_production_attempts)

        self.change_in_n_functional_from_latest_attempt_to_produce = (
            success - agent_was_functional)
        self.n_functional += (
            self.change_in_n_functional_from_latest_attempt_to_produce)
        self.n_retained_suppliers_from_latest_attempt_to_produce = (
            agent.n_retained_suppliers)

        return success

    def record_diagnostics_right_before_production_attempt(self, agent):
        self.n_production_attempts += 1
        self.latest_producer = agent
        self.n_customers_of_latest_producer = agent.out_degree()
        self.n_vulnerable_customers_of_latest_producer = sum(
            customer.is_vulnerable() for customer in agent.customers)
        self.n_suppliers_of_latest_producer = (
            len(agent.suppliers))
        return
