import sys
from math import log2, exp
import time
import random as r


class TreeNode:
    __slots__ = ("parent", "children", "attribute", "examples", "dut", "ita", "eng", "gain",
                 "classification", "thresholds")

    def __init__(self, examples=None, parent=None, thresholds=None):
        self.parent = parent
        self.children = []
        self.attribute = None
        self.examples = examples
        self.dut = 0
        self.ita = 0
        self.eng = 0
        self.gain = -1
        self.classification = None
        self.thresholds = thresholds

    def node_eval(self, example: list) -> str:
        """
        Recursive call to evaluate and descend to leaf node for classification.
        :param example: word to be classified
        :return: classification of example
        """
        if self.attribute is None:
            return self.classification
        else:
            len_index = 6
            feature = self.attribute
            thres_index = 0
            feat_value_index = -1
            if example[len_index] > 15:
                thres_index = 1 if example[len_index] < 25 else 2
            limit = self.thresholds[thres_index]
            match feature:
                case "hasJKWXY": feat_value_index = 0
                case "numDOUBLES": feat_value_index = 1
                case "medianLENGTH": feat_value_index = 2
                case "maxLENGTH": feat_value_index = 3
                case "notendAEIO": feat_value_index = 4
                case "endEN": feat_value_index = 5

            # recursively call on the next child node
            if example[feat_value_index] > limit:
                prediction = self.children[0].node_eval(example)
            else:
                prediction = self.children[1].node_eval(example)
            return prediction

    def save_tree(self, level=0) -> str:
        """
        Creates a string representation of the tree to write to a file for
        saving and reloading.
        :param level: depth in the tree
        :return: string to write to file
        """
        indent = "\t" * level
        if len(self.children) == 0:
            return f"{indent}Leaf: {self.classification} ,\n"
        elif level == 0:
            str_rep = (f"{indent}Root: {self.attribute} {self.thresholds[0]} "
                       f"{self.thresholds[1]} {self.thresholds[2]} {{\n")
        else:
            str_rep = (f"{indent}Node: {self.attribute} {self.thresholds[0]} "
                       f"{self.thresholds[1]} {self.thresholds[2]} {{\n")

        for child in self.children:
            str_rep += child.save_tree(level + 1)
        return str_rep + indent + "}\n"

    def __str__(self, level=0):
        indent = "\t" * level
        if len(self.children) == 0:
            return (f"{indent}Leaf: {self.classification} - Dutch: {self.dut} "
                    f"- Italian: {self.ita}\n")
        elif level == 0:
            str_rep = (f"{indent}Root: {self.attribute} - Gain: {self.gain:.4f} "
                       f"- Dutch: {self.dut} - Italian: {self.ita}\n")
        else:
            str_rep = (f"{indent}Node: {self.attribute} - Gain: {self.gain:.4f} "
                       f"- Dutch: {self.dut} - Italian: {self.ita}\n")

        for child in self.children:
            str_rep += child.__str__(level + 1)
        return str_rep


class DecisionTree:
    __slots__ = ("root", "model_file", "classification_results", "len_index",
                 "ident_index", "features", "is_stump")

    def __init__(self, classification_results=None, model_file=None, is_stump=False):
        if model_file is None:
            self.root = None
        else:
            self.root = self.build_tree(model_file)
        self.classification_results = classification_results
        self.len_index = 7
        self.ident_index = 6
        self.features = ["hasJKWXY", "numDOUBLES", "medianLENGTH", "maxLENGTH", "notendAEIO", "endEN"]
        self.is_stump = is_stump

    def build_tree(self, model_file: str) -> TreeNode:
        """
        Builds decision tree based on data file.
        :param model_file: file containing structure of decision tree
        :return: the root node of the tree
        """
        cur_node = None
        with open(model_file, 'r', encoding="utf-8") as model_data:
            try:
                for line in model_data:
                    line = line.strip()
                    if line == "Decision Tree.":
                        continue
                    elif line.startswith("Root"):
                        new_node = TreeNode()
                        line = line.split()
                        new_node.attribute = line[1]
                        new_node.thresholds = [int(line[2]), int(line[3]), int(line[4])]
                        cur_node = new_node
                    elif line.startswith("Node"):
                        new_node = TreeNode(None, cur_node)
                        line = line.split()
                        new_node.attribute = line[1]
                        new_node.thresholds = [int(line[2]), int(line[3]), int(line[4])]
                        cur_node.children.append(new_node)
                        cur_node = new_node
                    elif line.startswith("Leaf"):
                        new_node = TreeNode(None, cur_node)
                        line = line.split()
                        # default 'Not distinguishable' leaf nodes to DUT classification
                        new_node.classification = "DUT" if line[1] == "***" else line[1]
                        cur_node.children.append(new_node)
                    elif line.startswith("}"):
                        if cur_node.parent is not None:
                            cur_node = cur_node.parent
            except UnicodeDecodeError:
                print(f"Decode error in {model_file}.\n")
        return cur_node

    def train_tree(self) -> None:
        """
        Kicker function for recursive calls to train decision tree.
        :return: None
        """
        feature_list = ["hasJKWXY", "numDOUBLES", "medianLENGTH", "maxLENGTH", "notendAEIO", "endEN"]
        self._train_tree(self.root, feature_list)

    def _train_tree(self, cur_node: TreeNode, feature_list: list) -> None:
        """
        Recursive method to train the tree on each example.
        :param cur_node: node under consideration
        :param feature_list: list of remaining features available to use
        :return: None
        """
        # positive examples are Dutch, negative examples are Italian
        total = len(cur_node.examples)
        best_gain = -1
        best_feature = ""
        best_thresholds = []
        best_pos, best_neg = [], []

        # get number of pos / neg examples
        for i in range(total):
            if cur_node.examples[i][self.ident_index] == "DUT":
                cur_node.dut += 1
            else:
                cur_node.ita += 1

        # base case that all examples are ITA or DUT
        if cur_node.dut > 0 and cur_node.ita == 0:
            cur_node.classification = "DUT"
            return
        elif cur_node.dut == 0 and cur_node.ita > 0:
            cur_node.classification = "ITA"
            return

        # base case that all features have been used
        if len(feature_list) == 0:
            if cur_node.dut > cur_node.ita:
                cur_node.classification = "DUT"
            elif cur_node.dut < cur_node.ita:
                cur_node.classification = "ITA"
            else:
                cur_node.classification = "*** No more features, same matches"
            return

        for feature in feature_list:
            pos, neg = [], []
            feat_idx = self.features.index(feature)
            test_limits = self.get_threshold(cur_node.examples, feat_idx)
            for j in range(total):
                limit = test_limits[0]
                if cur_node.examples[j][self.len_index] > 15:
                    limit = test_limits[1] if cur_node.examples[j][self.len_index] < 25 else test_limits[2]
                if cur_node.examples[j][feat_idx] > limit:
                    pos.append(cur_node.examples[j])
                else:
                    neg.append(cur_node.examples[j])

            # get numbers for gain calculations
            sub1_pos, sub2_pos = 0, 0
            for k in range(len(pos)):
                if pos[k][self.ident_index] == "DUT":
                    sub1_pos += 1
            sub1_neg = len(pos) - sub1_pos
            for k in range(len(neg)):
                if neg[k][self.ident_index] == "DUT":
                    sub2_pos += 1
            sub2_neg = len(neg) - sub2_pos

            # calculate gain
            # print(f"Calculating gain for {feature}")
            gain = self.calc_gain(cur_node.dut, cur_node.ita, sub1_pos, sub1_neg, sub2_pos, sub2_neg)

            # store best gain values
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_thresholds = test_limits
                best_pos = pos
                best_neg = neg

        if best_gain == 0:
            if cur_node.dut > cur_node.ita:
                cur_node.classification = "DUT"
            elif cur_node.dut < cur_node.ita:
                cur_node.classification = "ITA"
            else:
                cur_node.classification = "*** No gain, same matches"
            return
        else:
            # create node for best gain
            cur_node.attribute = best_feature
            cur_node.thresholds = best_thresholds
            cur_node.gain = best_gain
            new_feature_list = feature_list.copy()
            new_feature_list.remove(best_feature)

            cur_node.children.append(TreeNode(best_pos, cur_node))
            cur_node.children.append(TreeNode(best_neg, cur_node))
            for child in cur_node.children:
                self._train_tree(child, new_feature_list)

    def train_stump(self, cur_node: TreeNode) -> None:
        """
        Trains a single node as a stump.
        :param cur_node: node under consideration
        :return: None
        """
        # positive examples are Dutch, negative examples are Italian
        feature_list = ["hasJKWXY", "numDOUBLES", "medianLENGTH", "maxLENGTH", "notendAEIO", "endEN"]
        total = len(cur_node.examples)
        best_gain = -1
        best_feature = ""
        best_thresholds = []
        best_pos, best_neg = [], []

        # get number of pos / neg examples
        for i in range(total):
            if cur_node.examples[i][self.ident_index] == "DUT":
                cur_node.dut += 1
            else:
                cur_node.ita += 1

        for feature in feature_list:
            pos, neg = [], []
            feat_idx = self.features.index(feature)
            test_limits = self.get_threshold(cur_node.examples, feat_idx)
            for j in range(total):
                limit = test_limits[0]
                if cur_node.examples[j][self.len_index] > 15:
                    limit = test_limits[1] if cur_node.examples[j][self.len_index] < 25 else test_limits[2]
                if cur_node.examples[j][feat_idx] > limit:
                    pos.append(cur_node.examples[j])
                else:
                    neg.append(cur_node.examples[j])

            # get numbers for gain calculations
            sub1_pos, sub2_pos = 0, 0
            for k in range(len(pos)):
                if pos[k][self.ident_index] == "DUT":
                    sub1_pos += 1
            sub1_neg = len(pos) - sub1_pos
            for k in range(len(neg)):
                if neg[k][self.ident_index] == "DUT":
                    sub2_pos += 1
            sub2_neg = len(neg) - sub2_pos

            # calculate gain
            # print(f"Calculating gain for {feature}")
            gain = self.calc_gain(cur_node.dut, cur_node.ita, sub1_pos, sub1_neg, sub2_pos, sub2_neg)

            # store best gain values
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_thresholds = test_limits
                best_pos = pos
                best_neg = neg

        if best_gain == 0:
            if cur_node.dut > cur_node.ita:
                cur_node.classification = "DUT"
            elif cur_node.dut < cur_node.ita:
                cur_node.classification = "ITA"
            else:
                cur_node.classification = "*** No gain, same matches"
            return
        else:
            # create node for best gain
            cur_node.attribute = best_feature
            cur_node.thresholds = best_thresholds
            cur_node.gain = best_gain
            new_feature_list = feature_list.copy()
            new_feature_list.remove(best_feature)

            # append predicted classification to example
            for example in best_pos:
                example.append("DUT")
            for example in best_neg:
                example.append("ITA")

            cur_node.children.append(TreeNode(best_pos, cur_node))
            cur_node.children.append(TreeNode(best_neg, cur_node))

            for child in cur_node.children:
                # get number of pos / neg examples
                for i in range(len(child.examples)):
                    if child.examples[i][self.ident_index] == "DUT":
                        child.dut += 1
                    else:
                        child.ita += 1

                # set leaf classification
                if child.dut > child.ita:
                    child.classification = "DUT"
                elif child.dut < child.ita:
                    child.classification = "ITA"
                else:
                    child.classification = "*** No more features, same matches"

    def get_threshold(self, examples: list, value_idx: int) -> list:
        """
        Iterates through each threshold to find optimal value based on example
        length.
        :param examples: list of example values
        :param value_idx: index of value to find threshold for
        :return: list of optimal thresholds for each length
        """
        # code for single threshold
        # best_match = 0
        # best_threshold = 0
        # thresholds = set()
        # for example in examples:
        #     thresholds.add(example[value_idx])
        # for threshold in thresholds:
        #     match = 0
        #     for example in examples:
        #         if ((example[value_idx] > threshold and example[self.ident_index] == "DUT")
        #                 or (example[value_idx] <= threshold and example[self.ident_index] == "ITA")):
        #             match += 1
        #     if match > best_match:
        #         best_match = match
        #         best_threshold = threshold
        # return [best_threshold, best_threshold, best_threshold]

        # code to use length dependent thresholds
        best_thresholds = [-1, -1, -1]
        lengths = [15, 25, 55]
        short, med, long = [], [], []
        for example in examples:
            if example[self.len_index] < lengths[0]:
                short.append(example)
            elif example[self.len_index] < lengths[1]:
                med.append(example)
            elif example[self.len_index] < lengths[2]:
                long.append(example)
        example_lists = [short, med, long]
        for idx, example_list in enumerate(example_lists):
            best_match = 0
            thresholds = set()
            for example in example_list:
                thresholds.add(example[value_idx])
            for threshold in thresholds:
                match = 0
                for example in example_list:
                    if ((example[value_idx] > threshold and example[self.ident_index] == "DUT")
                            or (example[value_idx] <= threshold and example[self.ident_index] == "ITA")):
                        match += 1
                if match > best_match:
                    best_match = match
                    best_thresholds[idx] = threshold
        return best_thresholds

    def calc_gain(self, set_pos: int, set_neg: int, sub1_pos: int, sub1_neg: int,
                  sub2_pos: int, sub2_neg: int) -> float:
        """
        Calculates gain of current feature
        :param set_pos: number of positive examples based on feature
        :param set_neg: number of negative examples based on feature
        :param sub1_pos: number of positive examples in subset 1
        :param sub1_neg: number of negative examples in subset 1
        :param sub2_pos: number of positive examples in subset 2
        :param sub2_neg: number of negative examples in subset 2
        :return: information gain of asking feature
        """
        tot_set_examples = set_pos + set_neg
        set_entropy = self.calc_entropy(set_pos, tot_set_examples)

        tot_sub1_examples = sub1_pos + sub1_neg
        sub1_entropy = self.calc_entropy(sub1_pos, tot_sub1_examples)

        tot_sub2_examples = sub2_pos + sub2_neg
        sub2_entropy = self.calc_entropy(sub2_pos, tot_sub2_examples)

        set_pos_weight = tot_sub1_examples / tot_set_examples
        set_neg_weight = tot_sub2_examples / tot_set_examples
        gain = set_entropy - ((set_pos_weight * sub1_entropy) + (set_neg_weight * sub2_entropy))

        # print(f"{gain:.10f} = B({set_pos}/{tot_set_examples}) - "
        #       f"[({tot_sub1_examples}/{tot_set_examples}) * B({sub1_pos}/{tot_sub1_examples}) + "
        #       f"({tot_sub2_examples}/{tot_set_examples}) * B({sub2_pos}/{tot_sub2_examples})]\n")
        return gain

    def calc_entropy(self, pos: int, total: int) -> float:
        """
        Calculates entropy of provided values.
        :param pos: total positive examples in set
        :param total: total examples in set
        :return: entropy of set
        """
        q = 0
        if total != 0:
            q = pos / total
        pos_entropy = 0
        if q != 0:
            pos_entropy = q * log2(q)
        neg_entropy = 0
        if q != 1:
            neg_entropy = (1 - q) * log2(1 - q)
        return -(pos_entropy + neg_entropy)

    def add_root(self, all_examples: list) -> None:
        """
        Helper function to create root node with all examples.
        :param all_examples: training examples or example to predict
        :return: None
        """
        self.root = TreeNode(all_examples)

    def predict(self, examples) -> None:
        """
        Kicker method for recursive call on node to classify each example.
        :param examples: examples to be classified
        :return: None
        """
        for example in examples:
            classification = self.root.node_eval(example)
            print(f"\tPrediction - {classification}\tExample - {example[7]}")

    def get_results(self) -> None:
        """
        Kicker method to print and write training results to a file.
        :return: None
        """
        matched, not_matched = self._get_results(self.root, 0, 0, "")
        with open("model_results.txt", "a", encoding="utf-8") as output:
            output.write(f"\nCorrectness: {matched / len(self.root.examples):.4f}\n"
                         f"\tMatched: {matched}\n"
                         f"\tNot Matched: {not_matched}")
        print(f"Correctness: {matched / len(self.root.examples):.4f}\n"
              f"\tMatched: {matched}\n"
              f"\tNot Matched: {not_matched}\n"
              f"See results file for specific example classification.\n")

    def _get_results(self, cur_node: TreeNode, matched: int, not_matched: int, path: str) -> tuple:
        """
        Helper function to count correct classifications by language and number of words.
        :param cur_node: node in decision tree
        :param matched: number of successful classifications
        :param not_matched: number of unsuccessful classifications
        :param path: features used in the examples classification
        :return:
        """
        # add classification feature to path
        path += f"{cur_node.attribute} "

        # for leaf nodes, count correct and incorrect classifications
        if len(cur_node.children) == 0:
            with open("model_results.txt", "a", encoding="utf-8") as output:
                for example in cur_node.examples:
                    match = example[self.ident_index] == cur_node.classification
                    key = example[self.ident_index] + str(example[self.len_index])
                    if match:
                        matched += 1
                        self.classification_results[key][0] += 1
                    else:
                        not_matched += 1
                        self.classification_results[key][1] += 1
                        # tab next line to only record incorrectly classified examples
                    output.write(str(match) + f"\t{path}\t{cur_node.classification}\t{example[0:9]}\n")
        else:
            for child in cur_node.children:
                matched, not_matched = self._get_results(child, matched, not_matched, path)
        return matched, not_matched

    def print_details(self, weight: float) -> None:
        """
        Print classification results by category.
        :return: None
        """
        details = "Summarized results:\n"
        if self.is_stump:
            details += f'\tStump weight: {weight:.4}\n'
        for key in self.classification_results:
            details += (f"\t{key}:\tCorrect - {self.classification_results[key][0]:4}"
                        f"\tIncorrect - {self.classification_results[key][1]:4}\n")
        print(details)

    def save_model(self) -> None:
        """
        Writes the current tree structure, classifications, and thresholds to a
        file for loading in the future.
        :return: None
        """
        file_name = "saved_model_tree_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
        with open(file_name, "w", encoding="utf-8") as output:
            model_str = "Decision Tree.\n"
            model_str += self.root.save_tree()
            output.write(model_str)
            # print(model_str)
            print(f"\nModel saved as: {file_name}\n")

    def __str__(self):
        if self.is_stump:
            return str(self.root)
        else:
            return f"Decision Tree.\n" + str(self.root)


class DecisionStumps:
    __slots__ = "h", "w", "z", "k", "example_features"

    def __init__(self, examples=None, k=None, model_file=None):
        if model_file is None:
            self.h = [DecisionTree()] * k
            self.w = [1 / len(examples)] * len(examples)
            self.z = [0.0] * k
            self.k = k
            self.example_features = examples
        else:
            self.z = []
            self.h = self.build_stumps(model_file)
            self.w = None
            self.k = k
            self.example_features = examples

    def build_stumps(self, model_file: str) -> list:
        """
        Builds a collection of decision stumps from model file.
        :param model_file: data of model to build
        :return: a list of decision stumps
        """
        h = []
        cur_node = None
        with open(model_file, 'r', encoding="utf-8") as model_data:
            try:
                for line in model_data:
                    line = line.strip()
                    if line == "Decision Stumps.":
                        continue
                    elif line.startswith("Hypothesis"):
                        line = line.split()
                        for value in line[1:]:
                            self.z.append(float(value))
                    elif line.startswith("Root"):
                        new_node = TreeNode()
                        line = line.split()
                        new_node.attribute = line[1]
                        new_node.thresholds = [int(line[2]), int(line[3]), int(line[4])]
                        cur_node = new_node
                    elif line.startswith("Leaf"):
                        new_node = TreeNode(None, cur_node)
                        line = line.split()
                        # default 'Not distinguishable' leaf nodes to DUT classification
                        new_node.classification = "DUT" if line[1] == "***" else line[1]
                        cur_node.children.append(new_node)
                    elif line.startswith("}"):
                        if cur_node.parent is not None:
                            cur_node = cur_node.parent
                        else:
                            stump = DecisionTree(None, None, True)
                            stump.root = cur_node
                            h.append(stump)
            except UnicodeDecodeError:
                print(f"Decode error in {model_file}.\n")
        return h

    def train_stumps_adaboost(self) -> None:
        """
        AdaBoost algorithm.
        :return: None
        """
        n = len(self.example_features)
        ep = sys.float_info.epsilon
        examples = self.example_features
        for _k in range(self.k):
            # create new list of examples to focus on incorrectly classified examples
            if _k > 0:
                examples = self.build_example_set(examples)
            # create new stump
            classification_results = {"ITA10": [0, 0], "ITA20": [0, 0], "ITA50": [0, 0],
                                      "DUT10": [0, 0], "DUT20": [0, 0], "DUT50": [0, 0]}
            stump = DecisionTree(classification_results, None, True)
            stump.add_root(examples)
            stump.train_stump(stump.root)
            self.h[_k] = stump
            error = 0.0

            # calculate total error for stump
            for j in range(n):
                example = self.h[_k].root.examples[j]
                if example[self.h[_k].ident_index] != example[-1]:
                    error = error + self.w[j]

            error = min(error, 1 - ep)

            # amount of say
            if error == 0:
                self.z[_k] = 0.5 * (log2((1 - error) / ep))
            else:
                self.z[_k] = 0.5 * (log2((1 - error) / error))

            for j in range(n):
                example = self.h[_k].root.examples[j]
                if example[self.h[_k].ident_index] != example[-1]:
                    self.w[j] = self.w[j] * exp(self.z[_k])
                else:
                    self.w[j] = self.w[j] * exp(-self.z[_k])

            # normalize values
            self.w = [float(i) / sum(self.w) for i in self.w]

    def build_example_set(self, examples: list) -> list:
        """
        Builds a new list of examples stochastically based on an example's weight
        treated as a probability range between 0-1.
        :param examples: previous list of examples
        :return: new list of examples
        """
        new_examples = []
        new_weights = []
        for i in range(len(examples)):
            pick = r.random()
            accum = 0.0
            index = 0
            # add example weights to reach pick value which is the next example to add
            while index < len(examples):
                accum += self.w[index]
                if accum > pick:
                    break
                index += 1
            new_examples.append(examples[index])
            new_weights.append(self.w[index])
        self.w = new_weights
        return new_examples

    def predict(self, examples: list) -> None:
        """
        Kicker method for recursive call on node to classify each example by each
        stump. Compares total of stump classifications to determine final
        classification.
        :param examples: example to be classified
        :return: None
        """
        for example in examples:
            pos, neg = 0.0, 0.0
            for i, stump in enumerate(self.h):
                prediction = stump.root.node_eval(example)
                # invert prediction for stumps with errors below 50%
                if self.z[i] < 0:
                    prediction = "DUT" if prediction == "ITA" else "ITA"
                if prediction == "DUT":
                    pos += abs(self.z[i])
                elif prediction == "ITA":
                    neg += abs(self.z[i])
            classification = "DUT" if pos > neg else "ITA"
            print(f"\tPrediction - {classification}\tExample - {example[7]}")

    def get_results(self) -> None:
        """
        Kicker method to get results per stump.
        :return: None
        """
        for stump in self.h:
            stump.get_results()

    def print_details(self) -> None:
        """
        Kicker method to print results per stump.
        :return: None
        """
        for index, stump in enumerate(self.h):
            stump.print_details(self.z[index])

    def save_model(self) -> None:
        """
        Writes the collection of stump structure, classifications, and thresholds
        to a file for loading in the future.
        :return: None
        """
        file_name = "saved_model_stumps_" + time.strftime("%Y%m%d_%H%M%S") + ".txt"
        with open(file_name, "w", encoding="utf-8") as output:
            model_str = "Decision Stumps.\nHypothesis: "
            for value in self.z:
                model_str += f'{value} '
            model_str += "\n"
            for stump in self.h:
                model_str += stump.root.save_tree()
            output.write(model_str)
            # print(model_str)
            print(f"\nModel saved as: {file_name}\n")

    def __str__(self):
        output = 'Decision Stumps.\n'
        for stump in self.h:
            output += str(stump)
        return output
