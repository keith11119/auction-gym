# uncompyle6 version 2.14.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.4.3 (default, Nov 17 2016, 01:08:31) 
# [GCC 4.8.4]
# Embedded file name: /Local-Scratch/PycharmProjects/Sport-Analytic-U-Tree/continuous-U-Tree-ice-hockey/c_utree_oracle/Problem_moutaincar_control.py
# Compiled at: 2017-12-04 13:56:08
from datetime import datetime


class Problem:
    """
    An MDP. Contains methods for initialisation, state transition.
    Can be aggregated or unaggregated.
    """

    def __init__(self, estimator_type, competition='L', agent_num=0, split_size=100, games_directory = '../data_DR_L_split_100/'):
        assert games_directory is not None
        self.games_directory = games_directory
        self.estimator_type = estimator_type
        self.split_size = split_size
        self.competition = competition
        self.agent_num = agent_num
        self.stateFeatures = {'0': 'continuous', '1': 'continuous', '2': 'continuous', '3': 'continuous', '4': 'continuous'}
        self.reset = None
        self.isEpisodic = True
        self.nStates = len(self.stateFeatures)
        self.dimNames = ['0', '1', '2', '3', '4']
        self.dimSizes = ['continuous', 'continuous', 'continuous', 'continuous', 'continuous']
        d = datetime.today().strftime('%d-%m-%Y--%H:%M:%S')
        # self.probName = ('{0}_gamma={1}_mode={2}').format(d, gamma,
        #                                                   'Action Feature States' if self.nStates > 12 else 'Feature States')
        return
