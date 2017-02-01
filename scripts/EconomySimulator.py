# Author: Charles Brummitt, Kenan Huremovic, Paolo Pin,
#         Matthew Bonds, Fernando Vega-Redondo
"""Simulate an agent-based model of economic complexity and contagious
disruptions.

This module contains classes that wrap around the Economy class of the module
ABM.py. The classes here simulate the economies, collect information at each
step, and visualize the results.

This code is used to create Figure 7 in the paper "Contagious disruptions and
complexity traps in economic development" (2017) by the above authors.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ABM
import os
from collections import deque
import progressbar
from scipy.stats import skew
from itertools import cycle


class EconomyLongRunSimulator():
    """Simulate an economy for a certain number of steps.
    """
    def __init__(self, economy, figures_path=''):
        self.economy = economy
        self.figures_path = figures_path
        self.file_name_template = (
            'assortativity_functionality_alpha{alpha}'
            '_beta{beta}_r{r}_L{L}_xi{xi}_eps{eps}'
            '_n{n_agents}'
            ''.format(
                n_agents=self.economy.n_agents,
                alpha=self.economy.alpha,
                beta=self.economy.beta,
                r=self.economy.amount_more_likely_choose_func_supplier,
                L=self.economy.num_times_func_agent_more_likely_chosen,
                xi=self.economy.xi,
                eps=self.economy.exog_fail)
            .replace('.', 'p'))

    def simulate(self, n_production_attempts):
        bar = progressbar.ProgressBar()
        for production_attempt in bar(list(range(n_production_attempts))):
            self.economy.update_one_step()


def simulate_economy_long_run(
        n_agents, init_fraction_functional,
        alpha=.1, beta=.4, r=1.0, L=1.0, exog_fail=0.0, xi=0,
        n_steps=1000,
        n_steps_detect_fixed_point=50, tolerance_std=0.001,
        trial=None, verbose=True):
    """Simulate an economy for many steps, and record information about the
    state of the economy at the beginning and end."""
    if trial == 0 and verbose:
        print('trial = {trial}, F_0 = {init_F}, r={r}, xi={xi}'.format(
            trial=trial, init_F=init_fraction_functional, r=r, xi=xi))

    econ = ABM.Economy(
        n_agents, init_fraction_functional,
        alpha=alpha, beta=beta, r=r, L=L, exog_fail=exog_fail, xi=xi)

    init_best_response = econ.latest_best_response
    result = {
        'init_n_inputs_needed': init_best_response.n_inputs_needed,
        'init_n_inputs_attempted': init_best_response.n_inputs_attempted}

    recent_n_functional = deque([], n_steps_detect_fixed_point)
    for step in range(n_steps):
        econ.update_one_step()
        recent_n_functional.append(econ.fraction_functional_agents())
        if (step > n_steps_detect_fixed_point and
                np.std(recent_n_functional) < tolerance_std):
            break

    final_best_response = econ.latest_best_response

    result.update({
        'final_n_inputs_needed': final_best_response.n_inputs_needed,
        'final_n_inputs_attempted': final_best_response.n_inputs_attempted,
        'final_F': econ.fraction_functional_agents(),
        'n_agents': n_agents,
        'init_F': init_fraction_functional,
        'alpha': alpha,
        'beta': beta,
        'r': r,
        'L': L,
        'xi': xi,
        'exog_fail': exog_fail,
        'n_steps': n_steps,
        'n_production_attempts': econ.n_production_attempts,
        'terminated_early': econ.n_production_attempts < n_steps,
        'n_steps_detect_fixed_point': n_steps_detect_fixed_point,
        'tolerance_std': tolerance_std})

    buffers = {
        'init_buffer': (result['init_n_inputs_attempted'] -
                        result['init_n_inputs_needed']),
        'final_buffer': (result['final_n_inputs_attempted'] -
                         result['final_n_inputs_needed'])}
    result.update(buffers)

    return result


class AssortativitySimulator():
    """A wrapper of ABM.Economy that keeps track of lots of information after
    every production attempt, plus methods for plotting this information.

    This class keeps track of assortativity of functionality (i.e.,
    the fraction of (customer, supplier) pairs that are
    (functional, functional), (functional, dysfunctional), etc.), the number
    of suppliers retained by the agent choosing new supplier, and more. This
    class is useful for exploring individual simulations in detail.
    """
    def __init__(self, economy, figures_path=''):
        """Initialize the simulator by providing a ABM.Economy and optionally
        a path in which to save figures.
        """
        self.economy = economy
        self.fraction_functional_history = []
        self.dys_dys_fraction = []
        self.fun_dys_fraction = []
        self.dys_fun_fraction = []
        self.fun_fun_fraction = []
        self.latest_best_response_n_inputs_needed = []
        self.latest_best_response_n_inputs_attempted = []
        self.n_customers_of_latest_producer = []
        self.change_in_n_functional_from_latest_attempt_to_produce = []
        self.n_vulnerable_customers_of_latest_producer = []
        self.n_retained_suppliers_from_latest_attempt_to_produce = []
        self.n_exogenous_failures = []
        self.figures_path = figures_path
        self.file_name_template = (
            'assortativity_functionality_alpha{alpha}'
            '_beta{beta}_r{r}_xi{xi}_L{L}_eps{eps}'
            '_n{n_agents}'
            ''.format(
                n_agents=self.economy.n_agents,
                alpha=self.economy.alpha,
                beta=self.economy.beta,
                r=self.economy.amount_more_likely_choose_func_supplier,
                L=self.economy.num_times_func_agent_more_likely_chosen,
                xi=self.economy.xi,
                eps=self.economy.exog_fail)
            .replace('.', 'p'))

    def simulate(self, n_production_attempts):
        """Simulate a certain number of production attempts."""
        bar = progressbar.ProgressBar()
        for production_attempt in bar(list(range(n_production_attempts))):
            self.record_diagnostics_before_production_attempt()
            self.economy.update_one_step()

    def record_diagnostics_before_production_attempt(self):
        econ = self.economy
        self.fraction_functional_history.append(
            econ.fraction_functional_agents())

        counts = econ.customer_supplier_functionality_count(
            per_input_attempt=True)
        self.dys_dys_fraction.append(counts[(False, False)])
        self.fun_dys_fraction.append(counts[(True, False)])
        self.dys_fun_fraction.append(counts[(False, True)])
        self.fun_fun_fraction.append(counts[(True, True)])

        self.latest_best_response_n_inputs_needed.append(
            econ.latest_best_response.n_inputs_needed)
        self.latest_best_response_n_inputs_attempted.append(
            econ.latest_best_response.n_inputs_attempted)
        self.n_customers_of_latest_producer.append(
            econ.n_customers_of_latest_producer)

        self.change_in_n_functional_from_latest_attempt_to_produce.append(
            (econ.change_in_n_functional_from_latest_attempt_to_produce))
        self.n_vulnerable_customers_of_latest_producer.append(
            econ.n_vulnerable_customers_of_latest_producer)
        self.n_exogenous_failures.append(econ.n_exogenous_failures)
        self.n_retained_suppliers_from_latest_attempt_to_produce.append(
            econ.n_retained_suppliers_from_latest_attempt_to_produce)

    def plot_assortativity_of_functionality_time_series(
            self, ax=None, save_figure=False, show_legend=True,
            linewidth_F=3,
            linewidth_assort=1,
            legend_loc=(.9, .4)):
        """Plot time-series of fractions of customer–supplier relationships
        in which the customers and suppliers are functional or dysfunctional.
        """
        fig, ax = generate_fig_ax(ax)

        F_values = np.array(self.fraction_functional_history)

        F_lines, = ax.plot(F_values, linewidth=linewidth_F,
                           color='k')

        dys_dys_lines, = ax.plot(self.dys_dys_fraction,
                                 color='g', linewidth=linewidth_assort)
        fun_dys_lines, = ax.plot(self.fun_dys_fraction,
                                 color='m', linewidth=linewidth_assort)
        dys_fun_lines, = ax.plot(self.dys_fun_fraction,
                                 color='b', linewidth=linewidth_assort)
        fun_fun_lines, = ax.plot(self.fun_fun_fraction,
                                 color='c', linewidth=linewidth_assort)

        alpha_expected = 0.5
        self.dys_dys_expected_lines, = ax.plot(
            (1 - F_values)**2,
            color='g', linestyle='--', alpha=alpha_expected)
        self.fun_dys_expected_lines, = ax.plot(
            (1 - F_values) * F_values,
            color='m', linestyle='--', alpha=alpha_expected)
        self.dys_fun_expected_lines, = ax.plot(
            (1 - F_values) * F_values,
            color='b', linestyle='--', alpha=alpha_expected)
        self.fun_fun_expected_lines, = ax.plot(
            F_values**2,
            color='c', linestyle='--', alpha=alpha_expected)

        if show_legend:
            ax.legend(handles=(F_lines,
                               dys_dys_lines, fun_dys_lines,
                               dys_fun_lines, fun_fun_lines),
                      labels=('fraction functional',
                              'dysfunctional--dysfunctional',
                              'functional--dysfunctional',
                              'dysfunctional--functional',
                              'functional--functional'),
                      title='status of agents\n\& customer--supplier pairs',
                      loc=legend_loc)
        ax.set_xlabel('number of production attempts')

        self.maybe_save_figure(save_figure, fig, 'time_series')
        return

    def plot_buffer(self, ax=None, show_legend=True,
                    legend_pos=(0.7, 0.5),
                    save_figure=False,
                    kwargs={'alpha': 0.5, 'color': 'y', 's': 10}):
        """Plot the buffers (m - tau)."""
        fig, ax = generate_fig_ax(ax)
        buffers = (
            np.array(self.latest_best_response_n_inputs_attempted) -
            np.array(self.latest_best_response_n_inputs_needed))
        buffer_handle = ax.scatter(
            np.arange(len(buffers)), buffers, **kwargs)
        if show_legend:
            ax.legend([buffer_handle], ['buffer'], loc=legend_pos)
        self.maybe_save_figure(save_figure, fig, 'buffer')
        return buffer_handle

    def plot_properties_of_agents_who_become_dysfunctional(
            self, ax=None, scale_n_customers=1., save_figure=False,
            show_legend=True, legend_pos=(0.7, 0.5),
            n_customer_kws={'color': 'r', 'alpha': .2},
            n_vulnerable_kws={'color': '#A879AF', 'alpha': .5,
                              's': 40, 'marker': '*'},
            legend_loc=(.9, .4),
            n_customers_label='num. customers\nof agent who\n'
                              'becomes dysfunctional',
            n_vulnerable_label='num. vulnerable\ncustomers of agent\n'
                               'who becomes dysfunctional'):
        """Plot information about agents who become dysfunctional, namely
        the number of customers and the number of customers who are
        'vulnerable' (meaning that they have zero buffer, m = tau).
        """
        fig, ax = generate_fig_ax(ax)

        if scale_n_customers == 1:
            n_customers_label = ('num. customers\nof agent\nwho becomes\n'
                                 'dysfunctional')
        else:
            n_customers_label = (
                'num. customers\nof agent who\nbecomes\ndysfunctional '
                '(x {scale})'.format(scale=scale_n_customers))

        n_customers = np.array(self.n_customers_of_latest_producer)
        n_vulnerable_customers_of_latest_producer = np.array(
            self.n_vulnerable_customers_of_latest_producer)
        change_n_func = np.array(
            self.change_in_n_functional_from_latest_attempt_to_produce)

        mask_more_dysfunction = (change_n_func == -1)

        n_customers_agent_of_becomes_dysf = ax.scatter(
            np.where(mask_more_dysfunction)[0],
            np.array(n_customers[mask_more_dysfunction]) * scale_n_customers,
            **n_customer_kws)

        n_vulnerable_customers_of_agent_becomes_dysf = ax.scatter(
            np.where(mask_more_dysfunction),
            n_vulnerable_customers_of_latest_producer[mask_more_dysfunction],
            **n_vulnerable_kws)

        if show_legend:
            ax.legend(
                (n_customers_agent_of_becomes_dysf,
                 n_vulnerable_customers_of_agent_becomes_dysf),
                [n_customers_label, n_vulnerable_label],
                loc=legend_loc)

        self.maybe_save_figure(save_figure, fig, 'props_agents_become_dysf')

    def plot_strategies(
            self, ax=None, show_legend=True, legend_loc=(0.9, 0.5),
            save_figure=False):
        """Make a scatter plot o the time-series of m, tau, and m-tau."""
        fig, ax = generate_fig_ax(ax)
        mF_history = zip(
            self.latest_best_response_n_inputs_attempted,
            self.fraction_functional_history[1:])
        mF_plot = ax.scatter(
            np.arange(len(self.fraction_functional_history[1:])),
            [m * (1 - F) for m, F in mF_history],
            alpha=0.3, s=13)

        buffers = (
            np.array(self.latest_best_response_n_inputs_attempted) -
            np.array(self.latest_best_response_n_inputs_needed))
        buffer_handle = ax.scatter(
            np.arange(len(buffers)), buffers, alpha=0.3, color='y', s=10)

        m_handle = ax.scatter(
            np.arange(len(buffers)),
            self.latest_best_response_n_inputs_attempted,
            alpha=0.3, color='g', s=10)
        tau_handle = ax.scatter(
            np.arange(len(buffers)),
            self.latest_best_response_n_inputs_needed,
            alpha=0.3, color='r', s=10)
        ax.legend(
            [m_handle, tau_handle, buffer_handle, mF_plot],
            ['num. inputs attempted (m)', 'num. inputs needed (tau)',
             'buffer (m - tau)', 'm (1 - F)'], loc=legend_loc)
        self.maybe_save_figure(save_figure, fig, 'props_agents_become_dysf')

    def plot_n_suppliers_retained(
            self, ax=None,
            show_legend=True, legend_loc=(0.9, 0.5),
            save_figure=False):
        """Make a scatter plot of the number of customers retained by the agent
        who is attempting to produce and choosing new suppliers."""
        fig, ax = generate_fig_ax(ax)
        n_retained = (
            self.n_retained_suppliers_from_latest_attempt_to_produce[1:])

        ax.scatter(range(len(n_retained)), n_retained, alpha=.2)
        ax.set_ylabel('number of suppliers retained')
        ax.plot(moving_average(n_retained, n=50), color='y', linewidth=2)
        ax.legend(['moving average', 'data'], loc=legend_loc)
        self.maybe_save_figure(save_figure, fig, 'n_suppliers_retained')

    def plot_m_rMinus1_1minusF(
            self, ax=None,
            show_legend=True, legend_loc=(0.9, 0.5),
            save_figure=False):
        """Plot m * (r - 1) * (1 - F), which indicates whether the out-degree
        distribution is becoming heavy-tailed over time."""
        fig, ax = generate_fig_ax(ax)

        m_history = np.array(
            self.economy.latest_best_response.n_inputs_attempted)
        r = self.economy.amount_more_likely_choose_func_supplier
        F_history = np.array(self.fraction_functional_history)
        m_rMinus1_1minusF = m_history * (r - 1) * (1 - F_history)

        ax.plot(m_rMinus1_1minusF)
        ax.axhline(
            y=self.economy.n_agents,
            xmin=0,
            xmax=self.economy.n_production_attempts,
            linestyle='--', color='r')
        ax.set_ylabel(r'$m (r - 1) (1 - F)$')
        self.maybe_save_figure(save_figure, fig, 'm_rMinus1_1minusF')

    def combined_plot(
            self, save_figure=False,
            linewidth_F=3,
            linewidth_assort=1,
            scale_n_customers=1.,
            assort_legend_loc=(1, 0.2),
            properties_legend_loc=(1, 0.2),
            buffer_legend_loc=(1, 0.2),
            retained_legend_loc=(1, 0.2),
            mrF_legend_loc=(1, 0.2)):
        """Make a large figure containing plots of several time-series."""

        fig, ax = plt.subplots(figsize=(10, 12), nrows=5, sharex=True)

        self.plot_assortativity_of_functionality_time_series(
            ax=ax[0],
            linewidth_F=linewidth_F,
            linewidth_assort=linewidth_assort,
            legend_loc=assort_legend_loc)
        ax[0].set_xlabel('')

        self.plot_properties_of_agents_who_become_dysfunctional(
            ax=ax[1],
            scale_n_customers=scale_n_customers,
            legend_loc=properties_legend_loc,
            n_vulnerable_label='num. vulnerable\ncustomers of\n'
                               'agent who becomes\ndysfunctional')

        self.plot_strategies(ax=ax[2], legend_loc=buffer_legend_loc)
        self.plot_n_suppliers_retained(ax=ax[3],
                                       legend_loc=retained_legend_loc)
        self.plot_m_rMinus1_1minusF(ax=ax[4], legend_loc=mrF_legend_loc)

        ax[-1].set_xlim(-.05 * self.economy.n_production_attempts,
                        self.economy.n_production_attempts * 1.05)
        ax[-1].set_xlabel('number of production attempts')

        fig.suptitle(self.title())

        fig.subplots_adjust(right=.75, hspace=.1, top=.95)

        self.maybe_save_figure(save_figure, fig, 'combined_time_series_')

    def title(self):
        title_template = (r'$\alpha = {alpha}$, $\beta = {beta}$, $L = {L}$, '
                          r'$r = {r}$, $\xi = {xi}$, '
                          r'$\epsilon = {epsilon}$, '
                          '{n_agents} agents')
        """A title to use as the `suptitle` of plots."""
        return title_template.format(
            n_agents=self.economy.n_agents,
            n_steps=self.economy.n_production_attempts,
            alpha=self.economy.alpha,
            beta=self.economy.beta,
            r=self.economy.amount_more_likely_choose_func_supplier,
            xi=self.economy.xi,
            epsilon=self.economy.exog_fail,
            L=self.economy.num_times_func_agent_more_likely_chosen)

    def n_customers_plots(
            self, save_figure=False, show_utility_bins=True,
            utility_quantiles=10, customer_quantiles=10):
        """Make scatter plots of (1) the number of customers as a function of
        utility and (2) the standard deviatio and skew of the distribution of
        the number of customers.
        """
        fig, ax = plt.subplots(figsize=(10, 20), nrows=4, sharex=False)
        fig.suptitle(self.title(), y=.9)

        F = np.array(self.fraction_functional_history[1:])
        tau = np.array(self.latest_best_response_n_inputs_needed[1:])
        m = np.array(self.latest_best_response_n_inputs_attempted[1:])
        utility_history = F * tau ** self.economy.beta - self.economy.alpha * m

        n_cust_utility_history = np.array(list(zip(
            self.n_customers_of_latest_producer[1:], utility_history)))
        n_cust_utility_history_df = pd.DataFrame(
            n_cust_utility_history, columns=['n_customers', 'utility'])

        ax[0].plot(n_cust_utility_history_df.utility, label='utility')
        ax[0].plot(
            moving_average(n_cust_utility_history_df.n_customers.values, 100),
            alpha=.5, label='num. customers (moving average 100)')
        ax[0].legend()

        n_cust_utility_history_df['utility_quantile'], utility_bins = pd.qcut(
            n_cust_utility_history_df.utility, utility_quantiles, retbins=True)

        quantiles = np.linspace(
            0, 1, endpoint=False, num=customer_quantiles)[1:]
        quantiles_n_customers_binned_by_utility = (
            n_cust_utility_history_df.groupby('utility_quantile')
            .apply(lambda x: x['n_customers'].quantile(quantiles)))

        ax[1].scatter(
            n_cust_utility_history[:, 1],
            n_cust_utility_history[:, 0], alpha=.1)
        ax[1].set_ylabel(
            'number of customers of the agent who attempts to produce')
        ax[1].set_xlabel(r'utility $F \tau^\beta - \alpha m$')

        if show_utility_bins:
            for i in range(len(utility_bins) - 1):
                q = quantiles_n_customers_binned_by_utility.iloc[i].values
                for n_c in q:
                    ax[1].scatter(
                        [np.mean(utility_bins[i:i + 2])], [n_c],
                        color='m', marker='^')

        lines = ["-", "--", "-.", ":"]
        line_cycler = cycle(lines)
        for col in quantiles_n_customers_binned_by_utility.columns:
            quantiles_n_customers_binned_by_utility[col].plot(
                ax=ax[2], linestyle=next(line_cycler))

        ax[2].legend(loc=0, ncol=2)
        ax[2].set_xlabel('utility quantile')
        ax[2].set_ylabel('number of customers')

        legend = ax[2].get_legend()
        legend.set_title('quantile')
        title = legend.get_title()
        title.set_fontsize(12)

        std_n_customers_binned_by_utility = (
            n_cust_utility_history_df.groupby('utility_quantile')
            .apply(
                lambda x: x['n_customers'].std()))
        std_n_customers_binned_by_utility.plot(
            color='k', ax=ax[3],
            label='standard deviation of the number of customers')

        skew_n_customers_binned_by_utility = (
            n_cust_utility_history_df.groupby('utility_quantile')
            .apply(
                lambda x: skew(x['n_customers'])))
        skew_n_customers_binned_by_utility.plot(
            color='k', ax=ax[3], linestyle='--',
            label='skew of the number of customers')
        ax[3].set_xlabel('utility quantile')
        ax[3].legend(loc=0)

        self.maybe_save_figure(save_figure, fig, 'combined_time_series_')

    def out_degree_histogram_by_functionality(self, bins=30, scale='linear'):
        """Plot the out-degree distribution (i.e., the distribution of the
        number of customers), conditioned on whether the agents are functional
        or not.
        """
        fig, ax = plt.subplots(nrows=2, sharex=True)
        out_deg_func = [
            a.out_degree() for a in self.economy.agents if a.is_functional]
        out_deg_dysfunc = [
            a.out_degree() for a in self.economy.agents if not a.is_functional]
        F = self.economy.fraction_functional_agents()

        ax[0].hist(out_deg_func, bins=bins)
        ax[1].hist(out_deg_dysfunc, color='r', bins=bins)
        ax[0].set_title(
            'functional agents ({:.2%})'.format(F).replace('%', '\%'))
        ax[1].set_title(
            'dysfunctional agents ({:.2%})'.format(1 - F).replace('%', '\%'))
        ax[1].set_xlabel('out-degree = number of customers')
        for axis in ax:
            axis.set_xscale(scale)
            axis.set_yscale(scale)
        plt.show()

    def out_degree_ccdf(self, bins=30, scale='log', guiding_line_slope=None):
        """Plot the complementary CDF of the out-degree distribution (i.e.,
        the distribution of the number of customers).
        """
        fig, ax = plt.subplots(nrows=2, sharex=True)
        out_deg_func = [
            a.out_degree() for a in self.economy.agents if a.is_functional]
        out_deg_dysfunc = [
            a.out_degree() for a in self.economy.agents if not a.is_functional]
        F = self.economy.fraction_functional_agents()

        titles = [
            'functional agents ({:.2%})'.format(F).replace('%', '\%'),
            'dysfunctional agents ({:.2%})'.format(1 - F).replace('%', '\%')]

        ax[1].set_xlabel('out-degree = number of customers')
        for axis, data, title, color in zip(
                ax, [out_deg_func, out_deg_dysfunc], titles, ['k', 'r']):
            if len(data) > 0:
                axis.hist(
                    data, bins=bins, color=color,
                    normed=True, histtype='step', cumulative=-1)
                if guiding_line_slope:
                    right, left = max(data), max(min(data), 1)
                    right -= (right - left) / 10
                    left += (right - left) / 10
                    axis.plot(
                        [left, right],
                        [left**guiding_line_slope,
                         right**guiding_line_slope],
                        '--', linewidth=3)
            axis.set_xscale(scale)
            axis.set_yscale(scale)
            axis.set_title(title)
        plt.show()

    def maybe_save_figure(
            self, save_figure, figure,
            file_name_prefix, file_name_suffix='.pdf'):
        if save_figure:
            file_name = (file_name_prefix +
                         self.file_name_template + file_name_suffix)
            figure.savefig(os.path.join(self.figures_path, file_name))


def generate_fig_ax(ax):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    return fig, ax


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
