import numpy as np

# Attributes
#   state_transition_func: takes state and action and produces state'
#   possible_action_func: takes state and produces meaningfully different actions
#   reward_func: takes state and produces scalar reward
class MinimaxSearch:

    def __init__(
        self,
        state_transition_func: function, 
        possible_action_func: function,
        reward_func: function):

        self.state_transition_func = state_transition_func
        self.possible_action_func = possible_action_func
        self.reward_func = reward_func

    # recursive search function
    # returns action that results in optimal state and action value
    # action value is evaluated by mode = 'max' | 'min'
    def __call__(self, state: any, mode: str = 'max'):

        # check for possible actions
        possible_actions = self.possible_action_func(state)

        # if none, return no reward
        if len(possible_actions) == 0:
            return None, 0

        # else iterate through possible actions
        action_values = list()
        for action in possible_actions:

            state_prime = self.state_transition_func(state, action)
            state_prime_reward = self.reward_func(state_prime)

            if mode == 'max':
                state_prime_value = self.__call__(state_prime, 'min') + state_prime_reward
                action_values.append(state_prime_value)
            else:
                state_prime_value = self.__call__(state_prime, 'max') + state_prime_reward
                action_values.append(state_prime_value)

            action_values.append(state_prime_value)

        # return optimal action
        if mode == 'max':
            max_action = possible_actions[action_values.index(max(action_values))]
            max_value = max(action_values)
            return max_action, max_value
        else:
            min_action = possible_actions[action_values.index(min(action_values))]
            min_value = min(action_values)
            return min_action, min_value
