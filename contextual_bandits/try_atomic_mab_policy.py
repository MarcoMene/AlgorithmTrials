import numpy as np

import matplotlib.pyplot as plt
from bsp_data_science.statistics.binomial import binomial_urate
from bsp_data_science.statistics.hypothesis_testing.t_statistics import ufloat_t_statistics
from numpy.random import uniform
from scipy.stats import norm
from uncertainties import  UFloat

"""
Little MC to see how it's better to make a decision.
2 actions
1 target variable (reward)

TODO: observe regret

3 policies:
  1. wait and decide @ 2 sigma
  2. follow the flukes up to a epsilon - early decision (Vowrabbit standard)s
  3. allocate proportionally to confidence, up to a epsilon (MM strategy)
"""


np.random.seed(66546)

epsilon = 0.1

true_conversion_rates =  [0.18, 0.2]   # wanna maximize

N = 10000  # number of users

policy = 3

N_sim = 100



def cr_measurement(segment):
    if trials[segment] == 0:
        return None
    return binomial_urate(successes[segment], trials[segment])


def _relative_importance_based_on_superposition(reference_v_e: UFloat, current_v_e: UFloat):
    """
    Gives the relative weight of the segment with value <= of the reference one,
    based on value separation weighted over errors, with a t-test
    :return:                relative weight (max 1=same importance, min 0 = negligible)
    """
    v1, e1 = reference_v_e.n, reference_v_e.s
    v2, e2 = current_v_e.n, current_v_e.s
    if e1 ** 2 + e2 ** 2 <= 0:
        return 1.0
    t = np.fabs(v1 - v2) / np.sqrt(e1 ** 2 + e2 ** 2)
    if t > 4:
        return 0
    p = norm.cdf(-t) * 2  # gaussian 2-sided p-value
    return p


def prob_1(policy=1):
    """
    :param policy:   1, 2, 3
    :return:
    """
    if uniform() < epsilon:
        return np.random.choice([0,1])

    cr0 = cr_measurement(0)
    cr1 = cr_measurement(1)

    if cr0 is None or cr1 is None:
        return np.random.choice([0, 1])

    if policy == 1:

        t = ufloat_t_statistics(cr0, cr1)
        if t < 2:
            return np.random.choice([0, 1])
        elif cr1.n > cr0.n:
            return 1
        else:
            return 0

    elif policy == 2:
        if cr1.n > cr0.n:
            return 1
        else:
            return 0

    elif policy == 3:

        r3 = uniform()

        if cr1.n > cr0.n:
            p0 = _relative_importance_based_on_superposition(cr1, cr0)
            if r3 < p0:
                return 0
            else:
                return 1
        else:
            p1 = _relative_importance_based_on_superposition(cr0, cr1)
            if r3 < p1:
                return 1
            else:
                return 0


regrets = []

for _ in range(N_sim):
    # measurement sections
    successes = [0, 0]
    trials =  [0, 0]

    conversions1 = 0
    best_conversions = 0
    for n in range(N):
        chosen_action1 = prob_1(policy)

        trials[chosen_action1] += 1

        r = uniform()

        if r < true_conversion_rates[chosen_action1]:
            conversions1 += 1
            successes[chosen_action1] += 1

        if r < true_conversion_rates[1]:
            best_conversions += 1


    regret1 = best_conversions - conversions1

    regrets.append(regret1)

    print(f"Policy {policy} conversions {conversions1} - regret {regret1}")

avg_regret = np.mean(regrets)
std_regret = np.std(regrets)
avg_regret_err = std_regret/np.sqrt(N_sim)

plt.title(f"Policy {policy} regrets \n epsilon {epsilon}, cr {true_conversion_rates}, users {N} \n avg_regret {round(avg_regret, 2)} ± {round(avg_regret_err, 2)} - std {round(std_regret, 2)}")
plt.hist(regrets, bins='sqrt')
plt.xlabel('regret')

plt.show()


