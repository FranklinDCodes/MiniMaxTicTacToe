
import numpy as np
import random

# Attributes
#   state_transition_func: takes state and action and produces state'
#   possible_action_func: takes state and produces meaningfully different actions
#   reward_func: takes state and produces scalar reward
#   terminal_func: takes state and returns true if the episode has terminated and search should stop
class MinimaxSearch:

    def __init__(
        self,
        state_transition_func: any, 
        possible_action_func: any,
        reward_func: any,
        terminal_func: any):

        self.state_transition_func = state_transition_func
        self.possible_action_func = possible_action_func
        self.reward_func = reward_func
        self.terminal_func = terminal_func

    # recursive search function
    # returns action that results in optimal state and action value
    # action value is evaluated by mode = 'max' | 'min'
    def __call__(self, state: any, mode: str = 'max'):

        # check for possible actions
        possible_actions = self.possible_action_func(state)

        # if none, return no reward
        if len(possible_actions) == 0 or self.terminal_func(state):
            return None, 0

        # else iterate through possible actions
        action_values = list()
        for action in possible_actions:

            if mode == 'min':
                state_prime = self.state_transition_func(state, action, 2)
            else:
                state_prime = self.state_transition_func(state, action, 1)
            state_prime_reward = self.reward_func(state_prime)

            if mode == 'max':
                state_prime_value = self.__call__(state_prime, 'min')[1] + state_prime_reward
                action_values.append(state_prime_value)
            else:
                state_prime_value = self.__call__(state_prime, 'max')[1] + state_prime_reward
                action_values.append(state_prime_value)

        # return optimal action
        if mode == 'max':

            # grab all actions that have max value
            max_actions = [act for idx, act in enumerate(possible_actions) if action_values[idx] == max(action_values)]

            # choose one randomly
            max_action = random.choice(max_actions)
            max_value = max(action_values)

            return max_action, max_value
        
        else:
            
            # grab all actions that have min value
            min_actions = [act for idx, act in enumerate(possible_actions) if action_values[idx] == min(action_values)]

            # choose one randomly
            min_action = random.choice(min_actions)
            min_value = min(action_values)

            return min_action, min_value
