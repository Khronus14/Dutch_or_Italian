"""
wiki.py
Author: David Robinson, ddr6248@rit.edu

Program to predict if a list of words are Italian or Dutch.

"""
import re
import sys
from wiki_objects import DecisionTree, DecisionStumps


def error_and_exit(error: str, user_error: bool) -> None:
    """
    Handles error messages and exits program
    :param error: Description of error.
    :param user_error: If error is user induced
    :return: None
    """
    print(error)
    if user_error:
        sys.exit("Usage: python wiki.py train {data_file}\n"
                 "\tdata_file - data file of examples\n"
                 "Usage: python wiki.py predict {model} {data_file}\n"
                 "\tmodel - tree, stumps, or best\n"
                 "\tdata_file - data file of examples\n")
    else:
        sys.exit(-1)


def argument_check() -> tuple:
    """
    Checks for valid arguments.
    :return: Tuple of CLA.
    """
    try:
        if sys.argv[1].lower() == "train":
            process = sys.argv[1].lower()
            examples_file = "examples_10000_10DEC.txt"
            return process, examples_file
        elif sys.argv[1].lower() == "predict":
            match sys.argv[2].lower():
                case "tree" | "stumps" | "best":
                    process = sys.argv[1].lower()
                    model = sys.argv[2].lower()
                    examples_file = sys.argv[3].lower()
                    return process, model, examples_file
                case _:
                    error_and_exit("Invalid model", True)
        else:
            error_and_exit("Invalid mode", True)
    except IndexError:
        error_and_exit("Invalid number of arguments", True)


def process_file(args: tuple, training: bool) -> list:
    """
    Prepars input examples for training or prediction.
    :param args: CLAs
    :param training: check flag if training or predicting
    :return: extracted feature values for each example
    """
    example_features = []
    data_file = args[1] if training else args[2]
    with open(data_file, 'r', encoding="utf-8") as raw_examples:
        try:
            if training:
                for line in raw_examples:
                    line = line.strip()
                    features = extract_features(line[7:])
                    features.append(line[:3])
                    features.append(int(line[4:6]))
                    features.append(line[7:])
                    example_features.append(features)
            else:
                for line in raw_examples:
                    line = line.strip()
                    # line = line[7:]  # debug code for removing true language
                    features = extract_features(line)
                    features.append(len(line.split()))
                    features.append(line)
                    example_features.append(features)
        except UnicodeDecodeError as error:
            error_and_exit(f"Decode error: {error}\n"
                           f"\tExample before error: {line}", False)
    return example_features


def extract_features(raw_example: str) -> list:
    """
    Extracts features from word example.
    :param raw_example: string of words from a language
    :return: feature values of provided example
    """
    # count occurrences of the letters j, k, w, x, and y
    jkwxy_count = len(re.findall("[jkwxy]", raw_example, re.IGNORECASE))

    # count occurrences of double letters
    doubles_count = len(re.findall(r"([a-z])\1", raw_example, re.IGNORECASE))

    # median length of words
    words = re.split(" ", raw_example)
    words = sorted(words, key=len)
    mid = len(words) // 2
    word_median = len(words[mid])

    # max length of words
    word_len_max = max([len(word) for word in words])

    # word ends in a, e, or o
    ends_aeio = len(re.findall(r"([^aeio] |[^aeio]$)", raw_example, re.IGNORECASE))

    # word ends in en
    ends_en = len(re.findall(r"(en |en$)", raw_example, re.IGNORECASE))

    return [jkwxy_count, doubles_count, word_median, word_len_max, ends_aeio, ends_en]


def get_model(args: tuple, training: bool, examples: list) -> tuple:
    """
    Instantiates the model(s) based on CLAs.
    :param args: CLAs
    :param training: check flag if training or predicting
    :param examples: feature values of word examples
    :return: models needed based on CLAs
    """
    tree_model = None
    stump_model = None
    best_model = None
    # number of stumps to use in AdaBoosting
    k = 3
    classification_results = {"ITA10": [0, 0], "ITA20": [0, 0], "ITA50": [0, 0],
                              "DUT10": [0, 0], "DUT20": [0, 0], "DUT50": [0, 0]}
    if training:
        tree_model = DecisionTree(classification_results)
        tree_model.add_root(examples)
        stump_model = DecisionStumps(examples, k)
    else:
        model_type = args[1]
        match model_type:
            case "tree":
                tree_model = DecisionTree(None, "best_model_tree.txt")
            case "stumps":
                stump_model = DecisionStumps(None, None, "best_model_stumps.txt")
            case "best":
                best_model = which_model("best_model_overall.txt")
            case _:
                error_and_exit("Case not recognized", True)
    return tree_model, stump_model, best_model


def which_model(file_name: str) -> any:
    """
    Helper function to determine which model the 'best model' is based on the
    input file.
    :param file_name: data file containing best model description
    :return: correct instantiation of the best model
    """
    with open(file_name, 'r', encoding="utf-8") as model_data:
        try:
            for line in model_data:
                line = line.strip()
                if line == "Decision Tree.":
                    return DecisionTree(None, "best_model_tree.txt")
                elif line == "Decision Stumps.":
                    return DecisionStumps(None, None, "best_model_stumps.txt")
        except UnicodeDecodeError:
            print(f"Decode error in {file_name}.\n")


def train_models(tree_model: DecisionTree, stump_model: DecisionStumps) -> tuple:
    """
    Calls train method for each training model
    :param tree_model: tree model to train
    :param stump_model: stump model to train
    :return: trained models
    """
    # train tree model
    print("Training decision tree...")
    tree_model.train_tree()
    # train stump model
    print("Training decision stumps with AdaBoost...")
    stump_model.train_stumps_adaboost()
    return tree_model, stump_model


def main() -> None:
    arg_tuple = argument_check()
    is_training = arg_tuple[0] == "train"
    examples = process_file(arg_tuple, is_training)
    tree_model, stump_model, best_model = get_model(arg_tuple, is_training, examples)
    if is_training:
        tree_model, stump_model = train_models(tree_model, stump_model)

        # tree model
        print(f"{tree_model}")
        tree_model.get_results()
        tree_model.print_details(None)
        save = input("Press 'y' to save model or any other key to continue... ")
        if save == "y":
            tree_model.save_model()

        # stump model
        print(f"{stump_model}")
        stump_model.get_results()
        stump_model.print_details()
        save = input("\nPress 'y' to save model or any other key to continue... ")
        if save == "y":
            stump_model.save_model()

    else:
        # prediction
        if arg_tuple[1] == "tree":
            print(f"\nPredicting languages with decision {arg_tuple[1]}...")
            tree_model.predict(examples)
        elif arg_tuple[1] == "stumps":
            print(f"\nPredicting languages with decision {arg_tuple[1]}...")
            stump_model.predict(examples)
        elif arg_tuple[1] == "best":
            print(f"\nPredicting languages with {arg_tuple[1]} model...")
            best_model.predict(examples)


if __name__ == '__main__':
    main()
