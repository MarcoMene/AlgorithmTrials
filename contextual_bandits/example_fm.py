import itertools
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import vowpalwabbit


# This function modifies (context, action, cost, probability) to VW friendly format
def to_vw_example_format(context, actions, cb_label=None):
    if cb_label is not None:
        chosen_action, cost, prob = cb_label
    example_string = ""
    example_string += "shared |User country={} \n".format(context["country"])
    for action in actions:
        if cb_label is not None and action == chosen_action:
            example_string += f"0:{cost}:{prob} "
        example_string += "|Action price={} \n".format(action)
    # Strip the last newline
    final_string = example_string[:-1]
    return final_string




def sample_custom_pmf(pmf):
    draw = np.random.uniform()
    sum_prob = 0
    for index, prob in enumerate(pmf):
        sum_prob += prob
        if sum_prob > draw:
            return index, prob


def get_action(vw, context, actions):
    vw_text_example = to_vw_example_format(context, actions)
    pmf = vw.predict(vw_text_example)
    chosen_action_index, prob = sample_custom_pmf(pmf)
    return actions[chosen_action_index], prob


def run_simulation(vw, num_iterations, countries, actions, costs, do_learn=True):
    cost_sum = 0.0
    ctr = []

    for i in range(1, num_iterations + 1):
        # 1. In each simulation choose a user
        country = np.random.choice(countries)

        # 3. Pass context to vw to get an action
        user = {"country": country}
        action, prob = get_action(vw, user, actions)

        # 4. Get cost of the action we chose
        cost = costs[user["country"]][action]
        print(f"--- action {action}, prob {prob}, cost {cost}---")
        cost_sum += cost

        if do_learn:
            # 5. Inform VW of what happened so we can learn from it
            row = to_vw_example_format(user, actions, (action, cost, prob))
            print(row)
            # 6. Learn
            vw.learn(row)

        # We negate this so that on the plot instead of minimizing cost, we are maximizing reward
        ctr.append(-1 * cost_sum / i)

    return ctr


def plot_ctr(num_iterations, ctr):
    plt.plot(range(1, num_iterations + 1), ctr)
    plt.xlabel("num_iterations", fontsize=14)
    plt.ylabel("ctr", fontsize=14)
    plt.ylim([0, 1])
    plt.show()


if __name__ == "__main__":
    actions = ["t1", "t2", "t3", "t4"]

    countries = ["us", "it"]
    costs = {
        "us": {"t1": 0, "t2": 0, "t3": 0, "t4": -1},
        "it": {"t1": 0, "t2": -1, "t3": 0, "t4": 0},
        "gb": {"t1": 0, "t2": 0, "t3": -1, "t4": 0},
    }


    # Instantiate learner in VW
    vw = vowpalwabbit.Workspace("--cb_explore_adf -q UA --quiet --epsilon 0.2")

    num_iterations = 100
    ctr = run_simulation(vw, num_iterations, countries, actions, costs, do_learn=True)

    plot_ctr(num_iterations, ctr)