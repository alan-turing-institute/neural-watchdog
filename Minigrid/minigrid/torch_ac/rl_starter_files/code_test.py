import pandas as pd
import numpy

# Initialize dictionaries
variables = {
    '05995': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                                 'goal_ticker_fp': 0, 'goal_episode_bool': False,
                                 'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 21)},
    '199': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 21)},
    '298': {f'threshold_{i}': {'trigger_ticker_tp': 0, 'trigger_episode_bool': False, 
                               'goal_ticker_fp': 0, 'goal_episode_bool': False,
                               'trigger_ticker_fn': 0, 'goal_ticker_fn': 0} for i in range(1, 21)}
}

print(variables)