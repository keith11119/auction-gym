import random, numpy as np, optparse, sys, csv
import gc
import math
from collections import defaultdict
import linear_regression_numpy as linear_regression
from scipy.stats import ks_2samp
import timeit

NodeSplit = 0
NodeLeaf = 1
NodeFringe = 2
FEATURE_NAME_DICT = {'position': 0, 'velocity': 1}

class UNode:
    def __init__(self, idx, nodeType, parent, depth):
        self.idx = idx
        self.nodeType = nodeType
        self.parent = parent
        self.children = []
        #self.count = np.zeros(n_actions)
        self.count = 0
        self.transitions = {}
        #self.qValues = np.zeros(n_actions)
        self.qValues = 0
        self.distinction = None
        self.instances = []
        self.depth = depth
        self.weight = None
        self.bias = None
        self.average_diff = None
        self.update_times = 0

        # LR = linear_regression.LinearRegression()
        # self.weight = LR.weight_initialization()
        # self.bias = LR.bias_initialization()
        # return

    def utility(self):
        """
        :param: index: if index is HOME, return Q_home, else return Q_away
        :return: maximum Q value
        """
        qValues_cp = np.copy(self.qValues)
        qValues_cp = qValues_cp[qValues_cp != float(0)]
        return max(qValues_cp)

    def addInstance(self, instance, max_hist):
        """
        add new instance to node instance list
        if instance length exceed maximum history length, select most recent history
        :param instance:
        :param max_hist:
        :return:
        """
        self.instances.append(instance)
        if len(self.instances) > max_hist:
            self.instances = self.instances[1:]

    def updateModel(self, new_state, qValue):
        """
        1. add action reward
        2. add action count
        3. record transition states
        :param new_state: new transition state
        :param action: new action
        :param reward: reward of action
        :param home_identifier: identify home and away
        :return:
        """
        # self.qValues[action] = (self.qValues[action] * self.count[action] +
        #                         qValue) / (self.count[action] + 1)
        #
        # self.count[action] += 1
        # if new_state not in self.transitions[action]:
        #
        #     self.transitions[action][new_state] = 1
        # else:
        #     self.transitions[action][new_state] += 1
        self.qValues = (self.qValues * self.count +
                                qValue) / (self.count + 1)

        self.count += 1
        if new_state not in self.transitions:

            self.transitions[new_state] = 1
        else:
            self.transitions[new_state] += 1

    def applyDistinction(self, history, idx):
        """
        :param history: history of instances
        :param idx: the idx of instance to apply distinction
        :return: the index of children
        """
        inst = history[idx - self.distinction.back_idx]
        # if self.distinction.dimension == ActionDimension:
        #     return inst.action
        # if previous == 0:
        if self.distinction.iscontinuous:
            if inst.currentObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
                return 0
            else:
                return 1
        else:
            return int(inst.currentObs[self.distinction.dimension])
        # else:
        #     if self.distinction.iscontinuous:
        #         if inst.nextObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
        #             return 0
        #         return 1
        #     else:
        #         return int(inst.nextObs[self.distinction.dimension])

    def applyInstanceDistinction(self, inst):
        # if self.distinction.dimension == ActionDimension:
        #     return inst.action
        if self.distinction.iscontinuous:
            if inst.currentObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
                return 0
            return 1
        else:
            return int(inst.currentObs[self.distinction.dimension])


class Instance:
    """
    records the transition as an instance
    """

    def __init__(self, timestep, currentObs, qValue):
        self.timestep = int(timestep)
        self.currentObs = [float(Obs) for Obs in currentObs]
        self.qValue = qValue


class Distinction:
    """
    For split node
    """

    def __init__(self, dimension, back_idx, dimension_name='unknown', iscontinuous=False, continuous_divide_value=None):
        """
        initialize distinction
        :param dimension: split of the node is based on the dimension
        :param back_idx: history index, how many time steps backward from the current time this feature will be examined
        :param dimension_name: the name of dimension
        :param iscontinuous: continuous or not
        :param continuous_divide_value: the value of continuous division
        """
        self.dimension = dimension
        self.back_idx = back_idx
        self.dimension_name = dimension_name
        self.iscontinuous = iscontinuous
        self.continuous_divide_value = continuous_divide_value

    def __eq__(self, distinction):
        return self.dimension == distinction.dimension and self.back_idx == distinction.back_idx and self.continuous_divide_value == distinction.continuous_divide_value

class CUTree:
    def __init__(self, dim_sizes, dim_names, max_hist, max_back_depth=1, max_depth=20, min_split_instances=50, hard_code_flag=True, training_mode=''):

        # LR = linear_regression.LinearRegression()
        # weight = LR.weight_initialization()
        # bias = LR.bias_initialization()

        self.node_id_count = 0
        self.root = UNode(self.genId(), NodeLeaf, None, 1)
        #self.n_actions = n_actions
        self.max_hist = int(max_hist)
        self.max_back_depth = max_back_depth
        self.max_depth = int(max_depth)
        self.history = []
        self.n_dim = len(dim_sizes)
        self.dim_sizes = dim_sizes
        self.dim_names = dim_names
        self.min_split_instances = int(min_split_instances)
        self.nodes = {self.root.idx: self.root}
        self.term = UNode(self.genId(), NodeLeaf, None, 1)
        self.start = UNode(self.genId(), NodeLeaf, None, 1)
        self.nodes[self.term.idx] = self.term
        self.nodes[self.start.idx] = self.start
        self.hard_code_flag = hard_code_flag
        self.training_mode = training_mode
        self.game_number = None

        # self.root.weight = weight
        # self.root.bias = bias
        # self.term.weight = weight
        # self.term.bias = bias
        # self.start.weight = weight
        # self.start.bias = bias

        return

    def insertInstance(self, instance):
        """
        append new instance to history
        :param instance: current instance
        :return:
        """
        self.history.append(instance)

    def insertTestInstances(self, inst, qValue, ntype=NodeLeaf):
        node = self.root

        while node.nodeType != ntype:
            child = node.applyInstanceDistinction(inst)
            node.instances.append(qValue)
            node = node.children[child]

        if isinstance(node.instances[0], float):
            node.instances.append(qValue)
        else:
            node.instances = [qValue]

    def getTime(self):
        """
        :return: length of history
        """
        return len(self.history)

    def getLeaf(self, previous=0):
        """
        Get leaf corresponding to current history
        :param previous: 0 is not check goal, 1 is check it
        :return:
        """
        idx = len(self.history) - 1
        node = self.root
        if previous == 1:
            #if idx == -1 or self.history[idx].nextObs[0] == -1:
            if idx == -1:
                return self.start
        while node.nodeType != NodeLeaf:
            child = node.applyDistinction(self.history, idx)
            node = node.children[child]

        return node

    def genId(self):
        """
        :return: a new ID for node
        """
        self.node_id_count += 1
        return self.node_id_count

    def recursive_print_tree_structure(self, node, layer):
        tree_structure = ''
        for i in range(0, layer):
            tree_structure += '\t'

        if node.nodeType == NodeSplit:
            tree_structure += ('idx{2}(Q:{0}, distinct_name:{1}, dictinctin_value:{4}, par:{3})').format(
                format(node.utility(), '.4f'), node.distinction.dimension_name,
                node.idx, node.parent.idx if node.parent is not None else None,
                node.distinction.continuous_divide_value
            )
            child_string = ''
            for child in node.children:
                child_string += '\n' + self.recursive_print_tree_structure(child, layer + 1)

            tree_structure += child_string
        else:
            if node.nodeType == NodeLeaf:
                tree_structure += 'idx{1}(Q:{0}, par:{2})'.format(
                    format(node.utility(), '.4f'),
                    node.idx, node.parent.idx if node.parent is not None else None
                )
            else:
                raise ValueError(('Unsupported tree nodeType:{0}').format(node.nodeType))
        return tree_structure

    def print_tree_structure(self, file_directory):
        root = self.root
        with open(file_directory, 'w') as (f):
            tree_structure = self.recursive_print_tree_structure(root, 0)
            print(f, tree_structure)
            f.write(tree_structure)

    def getInstanceLeaf(self, inst, ntype=NodeLeaf, previous=0):
        """
        Get leaf that inst records a transition from
        previous=0 indicates transition_from, previous=1 indicates transition_to
        :param inst: target instance
        :param ntype: target node type
        :param previous: previous=0 indicates present inst, previous=1 indicates next inst
        :return:
        """
        idx = inst.timestep + previous
        if previous == 1:
            if idx >= len(self.history):
                return self.term
            # if inst.nextObs[0] == -1 or inst.action == AbsorbAction:
            #     return self.start
        node = self.root
        while node.nodeType != ntype:
            child = node.applyDistinction(self.history, idx)
            node = node.children[child]

        return node

    def getAbsInstanceLeaf(self, inst, ntype=NodeLeaf):
        node = self.root
        while node.nodeType != ntype:
            child = node.applyInstanceDistinction(inst)
            node = node.children[child]

        return node

    def getCandidateDistinctions(self, node, select_interval=100):
        """
        construct all candidate distinctions
        :param node: target nodes
        :return: all candidate distinctions
        """
        p = node.parent
        anc_distinctions = []
        while p:
            anc_distinctions.append(p.distinction)
            p = p.parent

        candidates = []
        for i in range(self.max_back_depth):
            #for j in range(-1, self.n_dim):
            for j in range(0, self.n_dim):
                if j > -1 and self.dim_sizes[j] == 'continuous':
                    count = 0
                    for inst in sorted(node.instances, key=lambda inst: inst.currentObs[j]):
                        count += 1
                        if count % select_interval != 0:
                            continue
                        d = Distinction(dimension=j, back_idx=i, dimension_name=self.dim_names[j], iscontinuous=True,
                                        continuous_divide_value=inst.currentObs[j])
                        if d in anc_distinctions:
                            continue
                        else:
                            candidates.append(d)

                # else:
                #     d = Distinction(dimension=j, back_idx=i, dimension_name=self.dim_names[j] if j > -1 else 'actions')
                #     if d in anc_distinctions:
                #         continue
                #     else:
                #         candidates.append(d)

        return candidates

    def getUtileDistinction(self, node):
        """
        Different kinds of tests are performed here
        1. find all the possible distinction
        2. try to split node according to distinction and get expected future discounted returns
        3. perform test until find the proper distinction, otherwise, return None
        """
        if len(node.instances) < self.min_split_instances:
            return None
        cds = self.getCandidateDistinctions(node)
        return self.ksTestonQ(node, cds)

    def getQs(self, node):
        """
        Get all expected future discounted returns for all instances in a node
        (q-value is just the average EFDRs)
        """
        efdrs = np.zeros(len(node.instances))
        for i, inst in enumerate(node.instances):
            efdrs[i] = inst.qValue
        return [efdrs]

    def train_linear_regression_on_leaves(self, node):
        # I can decide the training mode here by changing the training mode
        leaves_number = 0
        if node.nodeType != NodeLeaf:
            for child in node.children:
                leaves_number += self.train_linear_regression_on_leaves(child)
            return leaves_number
        else:
            train_x = []
            train_y = []

            for instance in node.instances:
                train_x.append(instance.currentObs)
                train_y.append([instance.qValue])
            if len(train_x) != 0 and len(train_y) != 0:
                #sess = tf.InteractiveSession(config=config)
                if self.training_mode == '_epoch_linear':
                    training_epochs = len(node.instances)
                    # if self.game_number > 50:
                    #     training_epochs = training_epochs * 5
                    LR = linear_regression.LinearRegression(training_epochs=training_epochs, n_dim=len(train_x[0]))
                elif self.training_mode == '_linear_epoch_decay_lr':
                    node.update_times += 1
                    times = node.update_times
                    lr = 0.1 * float(1) / (1 + 0.0225 * times) * math.pow(0.977, len(node.instances) / 30)
                    # lr = 0.05*math.pow(0.02, float(len(node.instances))/float(self.max_hist))
                    training_epochs = len(node.instances) if len(node.instances) > 50 else 50
                    # if self.game_number > 50:
                    #     training_epochs = training_epochs * 5
                    #     lr = float(lr) / 5
                    LR = linear_regression.LinearRegression(training_epochs=training_epochs, learning_rate=lr,
                                                            n_dim=len(train_x[0]))
                elif len(self.training_mode) == 0:
                    LR = linear_regression.LinearRegression(n_dim=len(train_x[0]), training_epochs=50)
                else:
                    raise ValueError("undefined training mode")
                if node.weight is None or node.bias is None:
                    LR.read_weights()
                else:
                    LR.read_weights(node.weight, node.bias)

                trained_weights, trained_bias, average_diff = LR.gradient_descent(train_X=train_x,train_Y=train_y)

                print(sys.stderr, 'node index is {0}'.format(node.idx))

                node.weight = None
                node.bias = None
                node.weight = trained_weights
                node.bias = trained_bias
                node.average_diff = average_diff
                trained_weights = None
                trained_bias = None

                train_x = None
                train_y = None
                del LR
                gc.collect()

                # for i in gc.get_objects():
                #     after[type(i)] += 1
                # for k in after:
                #     if after[k] - before[k]:
                #         print (k, after[k] - before[k])

            return 1

    def ksTestonQ(self, node, cds, diff_significanceLevel=0.0):
        """
        KS test is performed here
        1. find all the possible distinction
        2. try to split node according to distinction and get expected future discounted returns
        3. perform ks test until find the proper distinction, otherwise, return None
        :param diff_significanceLevel: changed from 0.01 to 0.0
        :param node:
        :return:
        """
        assert node.nodeType == NodeLeaf
        root_utils = self.getQs(node)
        variance = np.var(root_utils)
        diff_max = float(0)
        cd_split = None
        for cd in cds:
            child_qs = self.splitQs(node, cd)
            for i, cq in enumerate(child_qs):

                if len(cq) == 0:
                    continue
                else:
                    variance_child = np.var(cq)

                    diff = variance - variance_child
                    if diff > diff_significanceLevel and diff > diff_max:
                        diff_max = diff
                        cd_split = cd
                        print(sys.stderr, 'variance test passed, diff={}, d = {}, back={}'.format(diff, cd.dimension,
                                                                                                     cd.back_idx))

        if cd_split:
            print(sys.stderr, 'Will be split, p={}, d={}, back={}'.format(diff_max, cd_split.dimension_name,
                                                                             cd_split.back_idx))
            return cd_split
        else:
            return cd_split

    def modelFromInstances(self, node):
        """
        rebuild model for leaf node, with newly added instance
        :param node:
        :return:
        """
        node.count = 0
        node.transitions = {}
        for inst in node.instances:
            leaf_to = self.getInstanceLeaf(inst, previous=1)
            if leaf_to != self.term:
                node.updateModel(leaf_to.idx, inst.qValue)
            else:
                node.updateModel(leaf_to.idx, inst.qValue)

    def hard_code_split(self):
        root = self.root
        if len(root.children) == 0 and len(self.history) >= 100:
            print(sys.stderr, "\nHard Coding\n")
            d = Distinction(dimension=-1, back_idx=0, dimension_name='actions')
            self.split(root, d)
            return True
        elif len(root.children) == 13:
            print(sys.stderr, "\nHard Coding\n")
            d = Distinction(dimension=9, back_idx=0, dimension_name=self.dim_names[9], iscontinuous=True,
                            continuous_divide_value=float(0))  # 0 is good enough
            self.split(root.children[5], d)
            return False
        else:
            return True

    def split(self, node, distinction):
        """
        split decision tree on nodes
        :param node: node to split
        :param distinction: distinction to split
        :return:
        """
        node.nodeType = NodeSplit
        node.distinction = distinction
        # if distinction.dimension == ActionDimension:
        #     for i in range(self.n_actions):
        #         idx = self.genId()
        #         n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
        #         self.nodes[idx] = n
        #         node.children.append(n)
        #         n.weight = node.weight
        #         n.bias = node.bias
        #
        # else:
        if not distinction.iscontinuous:
            for i in range(self.dim_sizes[distinction.dimension]):
                idx = self.genId()
                n = UNode(idx, NodeLeaf, node, node.depth + 1)
                self.nodes[idx] = n
                node.children.append(n)
                n.weight = node.weight
                n.bias = node.bias

        else:
            for i in range(2):
                idx = self.genId()
                n = UNode(idx, NodeLeaf, node, node.depth + 1)
                self.nodes[idx] = n
                node.children.append(n)
                n.weight = node.weight
                n.bias = node.bias

        for inst in node.instances:
            n = self.getInstanceLeaf(inst, previous=0)
            n.addInstance(inst, self.max_hist)

        for i, n in self.nodes.items():
            if n.nodeType == NodeLeaf:
                self.modelFromInstances(n)

        node.instances = []

    def splitQs(self, node, cd):

        if cd.iscontinuous:
            Q_value_list = []
            for i in range(0, 2):
                Q_value_list.append([])

            for inst in node.instances:

                if inst.currentObs[cd.dimension] <= cd.continuous_divide_value:
                    Q_value_list[0].append(inst.qValue)
                else:
                    Q_value_list[1].append(inst.qValue)

        else:
            Q_value_list = []
            for i in range(0, self.n_actions):
                Q_value_list.append([])

            for inst in node.instances:
                Q_value_list[int(inst.action)].append(inst.qValue)

        return Q_value_list

    def recursive_calculate_feature_importance(self, node):
        count = 0
        if node.nodeType != NodeLeaf:
            dimension_name = node.distinction.dimension_name
            parent_q_var = np.var(node.instances)
            sum_child_weighted_var = 0
            parent_instance_length = float(len(node.instances))
            for c in node.children:
                if len(c.instances) > 1 and isinstance(c.instances[1], float):
                    # try:
                    child_q_var = np.var(c.instances)
                    # except:
                    #     print 'catch you'
                    sum_child_weighted_var += child_q_var * (len(c.instances) / parent_instance_length)

            weight_sum = 0
            for weight_array in node.weight:
                weight_sum += abs(float(weight_array[0]))

            if dimension_name == 'actions':
                amplify_rate = 1
            else:
                amplify_rate = (1 + abs(node.weight[FEATURE_NAME_DICT.get(dimension_name)][0]) / weight_sum)

            feature_influence = amplify_rate * (parent_q_var - sum_child_weighted_var)
            feature_influence_sum = self.feature_influence_dict.get(dimension_name) + feature_influence
            self.feature_influence_dict.update({dimension_name: feature_influence_sum})

            for c in node.children:
                count += self.recursive_calculate_feature_importance(c)

        return 1 + count

    def testFringeRecursive(self, node):
        """
        recursively perform test in fringe, until return total number of split
        :param node: node to test
        :return: number of splits
        """
        if node.depth >= int(self.max_depth):
            return 0
        if node.nodeType == NodeLeaf:
            # if self.game_number <= 50:
            #     self.train_linear_regression_on_leaves(node)
            # if self.game_number > 50 and self.game_number % 5 == 0:
            #     self.train_linear_regression_on_leaves(node)
            start_time = timeit.default_timer()
            self.train_linear_regression_on_leaves(node)
            end_time = timeit.default_timer()
            print("train linear regression time: ", end_time - start_time)

            d = self.getUtileDistinction(node)
            if d:
                self.split(node, d)
                # if self.hard_code_flag:
                #     result_flag = self.hard_code_split()
                #     self.hard_code_flag = result_flag

                return 1 + self.testFringeRecursive(node)
            return 0
        total = 0
        for c in node.children:
            total += self.testFringeRecursive(c)

        return total

    def testFringe(self):
        """
        Tests fringe nodes for viable splits, splits nodes if they're found
        :return: how many real splits it takes
        """

        # if self.hard_code_flag:
        #     result_flag = self.hard_code_split()
        #     self.hard_code_flag = result_flag

        return self.testFringeRecursive(self.root) # starting from root

    def updateParents(self, new_state):
        node = self.root
        idx = len(self.history) - 1
        instance = self.history[idx]
        # action = instance.action
        qValue = instance.qValue
        while node.nodeType != NodeLeaf:
            # node.qValues[action] = (node.qValues[action] * node.count[action] + qValue) / (node.count[action] + 1)
            # node.count[action] += 1
            # transition_count = node.transitions[action].get(new_state.idx)
            node.qValues = (node.qValues * node.count + qValue) / (node.count + 1)
            node.count += 1
            transition_count = node.transitions.get(new_state.idx)
            if transition_count is not None:
                transition_count += 1
                # node.transitions[action].update({new_state.idx: transition_count})
                node.transitions.update({new_state.idx: transition_count})
            else:
                #node.transitions[action].update({new_state.idx: 1})
                node.transitions.update({new_state.idx: 1})

            child = node.applyDistinction(self.history, idx)
            node = node.children[child]

    def updateCurrentNode(self, instance, beginflag):
        """
        add the new instance ot LeafNode
        :param instance: instance to add
        :return:
        """
        old_state = self.getLeaf(previous=1)
        self.insertInstance(instance)
        new_state = self.getLeaf()
        self.updateParents(new_state)
        new_state.addInstance(instance, self.max_hist)
        if not beginflag:
            old_state.updateModel(new_state=new_state.idx, qValue=self.history[-2].qValue)
        # if instance.nextObs[0] == -1 or instance.action == AbsorbAction:
        #     new_state.updateModel(new_state=self.start.idx, qValue=instance.qValue)
