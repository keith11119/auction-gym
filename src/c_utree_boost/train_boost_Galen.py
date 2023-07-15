import optparse
import Problem
import pickle
import Agent_boost_Galen_action as Agent

optparser = optparse.OptionParser()
optparser.add_option("-m", "--max_node_hist", dest="MAX_NODE_HIST", default=3000,
                     help="max number of instance in every node (default = 10000)")
optparser.add_option("-c", "--check_fringe_freq", dest="CHECK_FRINGE_FREQ", default=1200,
                     help="check fringe frequent (default = 100)")
optparser.add_option("-d", "--directory_of_games", dest="GAME_DIRECTORY", default="",
                     help="games dir of all the games")
optparser.add_option("-g", "--game number to train", dest="GAME_NUMBER", default=None,
                     help="which game to train")
optparser.add_option("-e", "--training mode", dest="TRAINING_MODE", default='',
                     help="training mode")

opts = optparser.parse_args()[0]


# def train():
#     problem = Problem.Problem(games_directory=opts.GAME_DIRECTORY)
#     CUTreeAgent = Agent.CUTreeAgent(problem=problem, max_hist=opts.MAX_NODE_HIST,
#                                     check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0, training_mode=opts.TRAINING_MODE)
#
#     # CUTreeAgent.add_linear_regression()
#     if opts.GAME_NUMBER is None:
#         CUTreeAgent.episode(game_number=0)
#     else:
#         CUTreeAgent.episode(game_number=int(opts.GAME_NUMBER))

def train():
    problem = Problem.Problem(games_directory="../../data_DR_L_split/train/")

    CUTreeAgent = Agent.CUTreeAgent(problem=problem, max_hist=3000,
                                    check_fringe_freq=1200, is_episodic=0, training_mode='')

    # CUTreeAgent.add_linear_regression()
    if opts.GAME_NUMBER is None:
        CUTreeAgent.episode(game_number=0)
    else:
        CUTreeAgent.episode(game_number=int(opts.GAME_NUMBER))

def test_sh():
    print("hello")


if __name__ == "__main__":
    train()
