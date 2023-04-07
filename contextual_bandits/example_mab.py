import itertools
import numpy as np

import matplotlib.pyplot as plt
import vowpalwabbit
from numpy.random import uniform

from contextual_bandits.example_fm import to_vw_example_format, sample_custom_pmf

if __name__ == "__main__":

    np.random.seed(666)

    actions = ["t1", "t2"]

    countries = ["us"]  # , "it"

    # negative probability of converting?
    # costs = {
    #     "us": {"t1": 0, "t2": -1},
    # }

    # negative probability of converting?
    true_conversion_rates = {
        "us": {"t1": 0.3, "t2": 0.7},
    }

    all_contexts = countries

    epsilon = 0.1

    cb_type = 'dm' # ips   dr dm  mtr

    def simulate_conversion(cr):

        assert cr >= 0.0
        assert cr <= 1.0

        if uniform() <= cr:
            return 1
        return 0

    # Instantiate learner in VW
    vw = vowpalwabbit.Workspace(f"--cb_explore_adf -q UA --quiet --epsilon {epsilon} --cb_type {cb_type}")

    def run_simulation(vw, num_iterations, countries, actions, true_conversion_rates, do_learn=True):
        conversions = 0.0
        avg_cr = []

        iterative_probabilities_by_context = {
            context: np.array([ [np.nan for action in actions] for _ in range(num_iterations) ]) for context in all_contexts
        }

        def get_action(vw, context, actions):
            vw_text_example = to_vw_example_format(context, actions)
            pmf = vw.predict(vw_text_example)
            chosen_action_index, prob = sample_custom_pmf(pmf)
            return actions[chosen_action_index], prob, pmf

        for i in range(1, num_iterations + 1):
            print(f"iteration {i}")
            # 1. In each simulation choose a user
            country = np.random.choice(countries)

            # 3. Pass context to vw to get an action
            user = {"country": country}
            action, prob, all_probs = get_action(vw, user, actions)
            print(f"Action probabilities {all_probs} (context {user})")
            iterative_probabilities_by_context[user['country']][i-1] = all_probs

            # 4. Get cost of the action we chose
            conversion = simulate_conversion(true_conversion_rates[user["country"]][action])  # get conversion

            cost = -1 * conversion
            print(f"--- action taken {action}, prob {prob}, cost {cost} conversion {conversion}---")
            conversions += conversion

            if do_learn:
                # 5. Inform VW of what happened so we can learn from it
                row = to_vw_example_format(user, actions, (action, cost, prob))
                print(f"row for learning {row}")
                # 6. Learn
                vw.learn(row)

            # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
            avg_cr.append(conversions / i)

        return avg_cr, iterative_probabilities_by_context

    num_iterations = 1000

    avg_cr, iterative_probabilities_by_context = run_simulation(vw, num_iterations, countries, actions, true_conversion_rates, do_learn=True)

    achievable_cr = sum([ sum([epsilon * np.mean(list(true_conversion_rates[context].values())) ] + [(1. - epsilon)*max(true_conversion_rates[context].values())])  for context in all_contexts ]) / len(all_contexts)
    print(f"achievable_cr {achievable_cr}")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plt.title(f"Simple MAB \n cb_explore_adf epsilon {epsilon} cb_type {cb_type}")

    for context in all_contexts:
        p1 = iterative_probabilities_by_context[context][:, 0]
        p2 = iterative_probabilities_by_context[context][:, 1]
        ax2.plot(range(1, num_iterations + 1), p1, color='red', alpha=0.5, label=f"context {context} probability action {actions[0]}")
        ax2.plot(range(1, num_iterations + 1), p2, color='darkorange', alpha=0.5, label=f"context {context} probability action {actions[1]}")
        ax1.hlines(true_conversion_rates[context][actions[0]], 1, num_iterations + 1, linestyles="--", colors="red",  label=f"context {context} true cr {actions[0]}")
        ax1.hlines(true_conversion_rates[context][actions[1]], 1, num_iterations + 1, linestyles="--", colors="darkorange",  label=f"context {context} true cr {actions[1]}")

    ax1.hlines(achievable_cr, 1, num_iterations + 1, linestyles="--", colors="black",  label=f"achievable CR (epsilon {epsilon}): {round(achievable_cr,2)}")
    ax1.scatter(range(1, num_iterations + 1),  avg_cr, alpha=0.5, label="overall conversion rate")
    ax1.set_xlabel("iteration", fontsize=14)
    ax1.set_ylabel("overall conversion rate", fontsize=14)
    ax1.legend(loc="lower left")
    ax2.legend(loc="lower right")
    plt.ylim([0, 1])
    plt.show()