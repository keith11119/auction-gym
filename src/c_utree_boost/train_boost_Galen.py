import optparse
import Problem
import pickle
import Agent_boost_Galen_action_numpy as Agent
import os

optparser = optparse.OptionParser()
optparser.add_option("-m", "--max_node_hist", dest="MAX_NODE_HIST", default=10000,
                     help="max number of instance in every node (default = 10000)")
# optparser.add_option("-c", "--check_fringe_freq", dest="CHECK_FRINGE_FREQ", default=1200,
#                      help="check fringe frequent (default = 100)")
optparser.add_option("-d", "--directory_of_games", dest="GAME_DIRECTORY", default="",
                     help="games dir of all the games")
optparser.add_option("-g", "--game number to train", dest="GAME_NUMBER", default=None,
                     help="which game to train")
optparser.add_option("-e", "--training mode", dest="TRAINING_MODE", default='',
                     help="training mode")
optparser.add_option("-t", "--estimator_type", dest="ESTIMATOR", default='DR',
                     help="estimator type")
optparser.add_option("-c", "--competition", dest="COMPETITION", default='L',
                     help="competition level")
optparser.add_option("-a", "--agent", dest="AGENT", default=0,
                     help="agent")
optparser.add_option("-s", "--split_size", dest="SPLIT_SIZE", default=1000,
                     help="split size")
optparser.add_option("-p", "--max_depth", dest="MAX_DEPTH", default=20,
                     help="max depth")
optparser.add_option("-i", "--min_split_instances", dest="MIN_INSTANCES", default=50,
                     help="min split instances")
optparser.add_option("-x", "--special", dest="SPECIAL", default='',
                     help="variations")
optparser.add_option("-o", "--num_contexts", dest="NUM_CONTEXTS", default=5,
                     help="number of contexts")


opts = optparser.parse_args()[0]


def train():
    problem = Problem.Problem(estimator_type=opts.ESTIMATOR, competition=opts.COMPETITION, agent_num=opts.AGENT, split_size=opts.SPLIT_SIZE, games_directory=opts.GAME_DIRECTORY)
    CUTreeAgent = Agent.CUTreeAgent(problem=problem, max_hist=opts.MAX_NODE_HIST, max_depth=opts.MAX_DEPTH, min_split_instances=opts.MIN_INSTANCES, training_mode=opts.TRAINING_MODE, special=opts.SPECIAL, num_contexts=opts.NUM_CONTEXTS)
    # temp_output_path = f'../training_numpy_tempt_out_dir/{opts.ESTIMATOR}_{opts.COMPETITION}/agent_{opts.AGENT}/split_{opts.SPLIT_SIZE}/max_hist_10000_max_depth_{opts.MAX_DEPTH}_min_split_instances_{opts.MIN_INSTANCES}/'
    # if not os.path.exists(temp_output_path):
    #     os.makedirs(temp_output_path)

    # temp_output_path = f'../training_numpy_tempt_out_dir/DR_L/agent_0/split_1000/max_hist_10000_max_depth_{MAX_DEPTH}_min_split_instances_{MIN_INSTANCES}/'
    # if not os.path.exists(temp_output_path):
    #     os.makedirs(temp_output_path)
    if opts.GAME_NUMBER is None:
        CUTreeAgent.episode(game_number=0)
    else:
        CUTreeAgent.episode(game_number=int(opts.GAME_NUMBER))

# def train():
#     problem = Problem.Problem(estimator_type='DR', competition='H', agent_num=2, split_size=1000, games_directory='../../data_DR_H_split_1000/agent_2/train/')
#
#     CUTreeAgent = Agent.CUTreeAgent(problem=problem, max_hist=10000, max_depth=20, min_split_instances=50, training_mode='')
#
#     # CUTreeAgent.add_linear_regression()
#     CUTreeAgent.episode(game_number=1)
#     # if opts.GAME_NUMBER is None:
#     #     CUTreeAgent.episode(game_number=0)
#     # else:
#     #     CUTreeAgent.episode(game_number=int(opts.GAME_NUMBER))

def test_sh():
    print("hello")


if __name__ == "__main__":
    train()
