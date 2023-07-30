# uncompyle6 version 2.14.1
# Python bytecode 2.7 (62211)
# Decompiled from: Python 3.4.3 (default, Nov 17 2016, 01:08:31) 
# [GCC 4.8.4]
# Embedded file name: /Local-Scratch/PycharmProjects/Sport-Analytic-U-Tree/continuous-U-Tree-ice-hockey/c_utree_oracle/Agent_boost_Galen.py
# Compiled at: 2018-01-03 14:44:40
import gc
import numpy as np, ast, scipy.io as sio, os, unicodedata, pickle, C_UTree_boost_Galen_action_numpy as C_UTree, csv
from scipy.stats import pearsonr
import timeit
import linear_regression
import sys
# from tensorflow.python.framework import ops



class CUTreeAgent:
    """
      Agent that implements McCallum's Sport-Analytic-U-Tree algorithm
    """

    def __init__(self, problem, max_hist, max_depth=20, min_split_instances=50, training_mode='', special='', num_contexts=5):
        self.utree = C_UTree.CUTree(dim_sizes=problem.dimSizes, dim_names=problem.dimNames, max_hist=max_hist,
                                    max_depth=max_depth, min_split_instances=min_split_instances,
                                    training_mode=training_mode)
        self.problem = problem
        # self.TREE_PATH = './csv_oracle_linear_qsplit_test/'
        estimator_path = f'../UTree_model_numpy/{problem.estimator_type}_{problem.competition}{special}/'
        if not os.path.exists(estimator_path):
            os.makedirs(estimator_path)
        agent_path = f'{estimator_path}agent_{problem.agent_num}/'
        if not os.path.exists(agent_path):
            os.makedirs(agent_path)
        self.SAVE_PATH = f'{agent_path}model_boost_linear_save_{problem.split_size}_max_hist_{max_hist}_max_depth_{max_depth}_min_split_instances_{min_split_instances}{training_mode}/'
        if not os.path.exists(self.SAVE_PATH):
            os.makedirs(self.SAVE_PATH)
        self.PRINT_TREE_PATH = f'../print_tree_record_numpy/print_{problem.estimator_type}_{problem.competition}{special}_agent_{problem.agent_num}_boost_linear_tree_split_{problem.split_size}_max_hist_{max_hist}_max_depth_{max_depth}_min_split_instances_{min_split_instances}{training_mode}.txt'
        self.training_mode = training_mode
        self.num_contexts = num_contexts
        # print(tf.__version__)

    def update(self, currentObs, qValue, check_fringe=0, beginflag=False):
        """
        update the tree
        :param currentObs: current observation
        :param qValue: continuous action value
        :param terminal: if end
        :param check_fringe: whether to check fringe
        :return:
        """
        t = self.utree.getTime()
        # t is the number of instances inside historyx
        i = C_UTree.Instance(t, currentObs, qValue)
        self.utree.updateCurrentNode(i, beginflag)
        if check_fringe:
            start_time = timeit.default_timer()
            self.utree.testFringe() # ks test is performed here?
            stop_time = timeit.default_timer()
            print('testFringe time: ', stop_time - start_time)

    # def read_csv_game_record(self, csv_dir):
    #     dict_all = []
    #     with open(csv_dir, 'r') as csvfile:
    #         reader = csv.DictReader(csvfile)
    #         for row in reader:
    #             dict_all.append(row)
    #     return dict_all

    def read_csv_game_record_auction(self, csv_dir):
        data = np.genfromtxt(csv_dir, delimiter=' ')
        return data

    def read_Utree(self, save_path, game_number):
        print(sys.stderr, 'reading from {0}, starting at {1}'.format(self.SAVE_PATH, game_number))
        # temp = '{0}pickle_Game_File_{1}.p'.format(save_path, str(game_number))
        # print temp
        # temp1 = pickle.load(open(save_path + 'pickle_Game_File_' + str(game_number) + '.p', 'rb'))
        self.utree = pickle.load(open(save_path + 'pickle_Game_File_' + str(game_number) + '.p', 'rb'))
        self.utree.training_mode = self.training_mode
        self.utree.game_number = game_number + 1

    def save_csv_q_values(self, q_values, filename):
        with open(filename, 'wb') as (csvfile):
            fieldname = [
                'event_number', 'q_home', 'q_away', 'q_end']
            writer = csv.writer(csvfile)
            event_counter = 1
            for q_tuple in q_values:
                writer.writerow([event_counter, q_tuple[0], q_tuple[1], q_tuple[2]])
                event_counter += 1

    def normalization_q(self, q_list):
        sum_q = 0
        for index in range(0, len(q_list)):
            q_i = q_list[index]
            if q_i < 0:
                q_i = 0
                q_list[index] = q_i
            sum_q += q_i
        if sum_q != 0:
            q_norm_list = []
            for q_i in q_list:
                q_norm_list.append(float(q_i) / sum_q)
        else:
            q_norm_list = [0.5, 0.5, 0]
        print(q_norm_list)
        return q_norm_list

    def smooth_list(self, y, box_pts=10):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def smoothing_q(self, q_list):
        q_array = np.asarray(q_list)
        q_home = self.smooth_list(q_array[:, 0])
        q_away = self.smooth_list(q_array[:, 1])
        q_end = self.smooth_list(q_array[:, 2])

        return zip(q_home.tolist(), q_away.tolist(), q_end.tolist())


    def get_prediction(self, save_path, game_path, read_game_number, data_set = 'test'):
        print(sys.stderr, 'starting from {0}'.format(read_game_number))
        self.utree = pickle.load(open(save_path + 'pickle_Game_File_' + str(read_game_number) + '.p', 'rb'))
        print(sys.stderr, 'finishing read tree')

        # game_testing_record_dict = {}
        prediction_results = []
        game_record = self.read_csv_game_record_auction(f"{game_path}{data_set}_{self.problem.agent_num}.csv")
        event_number = len(game_record)

        for index in range(0, event_number):

            transition = game_record[index]
            currentObs = transition[:self.num_contexts]
            qValue = float(transition[self.num_contexts+2])

            inst = C_UTree.Instance(-1, currentObs, None)
            node = self.utree.getAbsInstanceLeaf(inst)

            value = currentObs @ node.weight + node.bias

            prediction_results.append(value[0])
            # print(sys.stderr, f"finish index: {index} - {value}")

        return prediction_results

    def merge_oracle_linear_q(self, test_qs, linear_qs, oracle_qs):
        criteria = 0.7
        merge_qs = []
        for index in range(0, len(test_qs)):
            if abs(test_qs[index] - linear_qs[index]) > criteria:
                merge_qs.append(oracle_qs[index])
            else:
                merge_qs.append(linear_qs[index])
        return merge_qs

    def compute_correlation(self, all_q_values_record, save_correlation_dir):
        linear_correl = pearsonr(all_q_values_record.get('output_q'), all_q_values_record.get('test_q'))[0]
        oracle_correl = pearsonr(all_q_values_record.get('oracle_q'), all_q_values_record.get('test_q'))[0]
        merge_correl = pearsonr(all_q_values_record.get('merge_q'), all_q_values_record.get('test_q'))[0]

        text_file = open("./" + save_correlation_dir, "a")

        text_file.write('{linear_correl: ' + str(linear_correl) + '}\n')
        text_file.write('{oracle_correl: ' + str(oracle_correl) + '}\n')
        text_file.write('{merge_correl: ' + str(merge_correl) + '}\n')
        text_file.write('\n')
        text_file.close()

    def compute_rse(self, all_q_values_record, save_rse_dir):
        linear_rse = self.relative_square_error(all_q_values_record.get('output_q'),
                                                all_q_values_record.get('test_q'))

        oracle_rse = self.relative_square_error(all_q_values_record.get('oracle_q'),
                                                all_q_values_record.get('test_q'))

        merge_rse = self.relative_square_error(all_q_values_record.get('merge_q'),
                                               all_q_values_record.get('test_q'))

        text_file = open("./" + save_rse_dir, "a")

        text_file.write('{home_linear_rse: ' + str(linear_rse) + '}\n')
        text_file.write('{home_oracle_rse: ' + str(oracle_rse) + '}\n')
        text_file.write('{home_merge_rse: ' + str(merge_rse) + '}\n')
        text_file.write('\n')
        text_file.close()

    def compute_rae(self, all_q_values_record, save_rae_dir):
        linear_rae = self.relative_absolute_error(all_q_values_record.get('output_q'),
                                                  all_q_values_record.get('test_q'))

        oracle_rae = self.relative_absolute_error(all_q_values_record.get('oracle_q'),
                                                  all_q_values_record.get('test_q'))

        merge_rae = self.relative_absolute_error(all_q_values_record.get('merge_q'),
                                                 all_q_values_record.get('test_q'))

        text_file = open("./" + save_rae_dir, "a")

        text_file.write('{home_linear_rae: ' + str(linear_rae) + '}\n')
        text_file.write('{home_oracle_rae: ' + str(oracle_rae) + '}\n')
        text_file.write('{home_merge_rae: ' + str(merge_rae) + '}\n')
        text_file.write('\n')
        text_file.close()

    def compute_mse(self, all_q_values_record, save_mse_dir):
        linear_mse = self.mean_square_error(all_q_values_record.get('output_q'),
                                            all_q_values_record.get('test_q'))

        oracle_mse = self.mean_square_error(all_q_values_record.get('oracle_q'),
                                            all_q_values_record.get('test_q'))

        merge_mse = self.mean_square_error(all_q_values_record.get('merge_q'),
                                           all_q_values_record.get('test_q'))

        text_file = open("./" + save_mse_dir, "a")

        text_file.write('{home_linear_mse: ' + str(linear_mse) + '}\n')
        text_file.write('{home_oracle_mse: ' + str(oracle_mse) + '}\n')
        text_file.write('{home_merge_mse: ' + str(merge_mse) + '}\n')
        text_file.write('\n')
        text_file.close()

    def compute_mae(self, all_q_values_record, save_mae_dir):
        linear_mse = self.mean_abs_error(all_q_values_record.get('output_q'),
                                         all_q_values_record.get('test_q'))

        oracle_mse = self.mean_abs_error(all_q_values_record.get('oracle_q'),
                                         all_q_values_record.get('test_q'))

        merge_mse = self.mean_abs_error(all_q_values_record.get('merge_q'),
                                        all_q_values_record.get('test_q'))

        text_file = open("./" + save_mae_dir, "a")

        text_file.write('{home_linear_mse: ' + str(linear_mse) + '}\n')
        text_file.write('{home_oracle_mse: ' + str(oracle_mse) + '}\n')
        text_file.write('{home_merge_mse: ' + str(merge_mse) + '}\n')
        text_file.write('\n')
        text_file.close()

    def relative_square_error(self, test_qs, target_qs):
        sse = 0
        rse = 0
        test_qs = map(float, test_qs)
        target_qs = map(float, target_qs)
        tm = np.mean(target_qs)
        for index in range(0, len(test_qs)):
            sse += (test_qs[index] - target_qs[index]) ** 2
            rse += (tm - target_qs[index]) ** 2
        return sse / rse

    def relative_absolute_error(self, test_qs, target_qs):
        sae = 0
        rae = 0
        test_qs = map(float, test_qs)
        target_qs = map(float, target_qs)
        tm = np.mean(target_qs)
        for index in range(0, len(test_qs)):
            sae += abs(test_qs[index] - target_qs[index])
            rae += abs(tm - target_qs[index])
        return sae / rae

    def mean_square_error(self, test_qs, target_qs):
        sse = 0
        for index in range(0, len(test_qs)):
            sse += (float(test_qs[index]) - float(target_qs[index])) ** 2
        return sse / len(test_qs)

    def mean_abs_error(self, test_qs, target_qs):
        sse = 0
        for index in range(0, len(test_qs)):
            sse += abs(float(test_qs[index]) - float(target_qs[index]))
        return sse / len(test_qs)

    def feature_importance(self):
        self.read_Utree(game_number=100, save_path=self.SAVE_PATH)

        self.utree.feature_influence_dict = {'position': 0, 'velocity': 0, 'actions': 0}

        game_to_print_list = range(301, 401)
        for game_number in game_to_print_list:

            game_record = self.read_csv_game_record(
                self.problem.games_directory + 'record_moutaincar_transition_game{0}.csv'.format(int(game_number)))

            event_number = len(game_record)

            for index in range(0, event_number):
                transition = game_record[index]
                currentObs = transition.get('observation').split('$')
                nextObs = transition.get('newObservation').split('$')
                reward = float(transition.get('reward'))
                action = float(transition.get('action'))
                qValue = float(transition.get('qValue'))

                inst = C_UTree.Instance(-1, currentObs, action, nextObs, reward, qValue)
                self.utree.insertTestInstances(inst=inst, qValue=qValue)

        visit_count = self.utree.recursive_calculate_feature_importance(node=self.utree.root)


        print (self.utree.feature_influence_dict)

    def episode(self, game_number, timeout=int(100000.0), save_checkpoint_flag=1):
        """
        start to build the tree within an episode
        :param save_checkpoint_flag:
        :param timeout: no use here
        :return:
        """
        start_game = game_number
        self.utree.hard_code_flag = True
        if start_game > 0:
            self.read_Utree(game_number=start_game, save_path=self.SAVE_PATH)
        count = 0

        game_record = self.read_csv_game_record_auction(f"{self.problem.games_directory}{game_number}.csv")
        event_number = len(game_record)
        beginflag = True
        count += 1
        for index in range(0, event_number):

            if self.problem.isEpisodic:
                transition = game_record[index]
                currentObs = transition[:self.num_contexts]
                # nextObs = transition.get('newObservation').split('$')
                # reward = float(transition.get('reward'))
                # action = float(transition.get('action'))
                qValue = float(transition[self.num_contexts+2])

                if index == event_number - 1:  # game ending
                    print(sys.stderr, '=============== update starts ===============')
                    # tracker.print_diff()
                    self.update(currentObs, qValue, beginflag=beginflag, check_fringe=1)
                    # tracker.print_diff()
                    print(sys.stderr, '=============== update finished ===============\n')
                else:
                    self.update(currentObs, qValue, beginflag=beginflag)
                    # else:
                    # currentObs = states[index]
                    # reward = rewards[index]
                    # self.getQ(currentObs, [], action, reward, home_identifier)
                beginflag = False

        if self.problem.isEpisodic:
            # print 'Game File ' + str(count)
            print('*** Writing Game File {0}***\n'.format(str(game_number + 1)))
            self.utree.print_tree_structure(self.PRINT_TREE_PATH)
            if save_checkpoint_flag and (game_number + 1) % 1 == 0:
                pickle.dump(self.utree,
                            open(self.SAVE_PATH + 'pickle_Game_File_' + str(game_number + 1) + '.p', 'wb'))
                # self.utree.tocsvFile(self.TREE_PATH + 'Game_File_' + str(game_number + 1) + '.csv')
