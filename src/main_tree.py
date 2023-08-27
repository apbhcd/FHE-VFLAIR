import random
import time

import argparse
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from evaluates.MainTaskTVFL import MainTaskTVFL
from load.LoadTreeParty import load_tree_parties
from load.LoadTreeConfigs import load_tree_configs

from dataset.credict_dataset import load_GiveMeSomeCredit


def evaluate_performance(tvfl, X_train, y_train, X_test, y_test, num_classes):
    # import pdb; pdb.set_trace()
    y_pred_train = tvfl.clf.predict_proba(X_train)
    y_pred_test = tvfl.clf.predict_proba(X_test)
    if num_classes == 2:
        train_auc = roc_auc_score(y_train, np.array(y_pred_train)[:,1], multi_class='ovr')
        test_auc = roc_auc_score(y_test, np.array(y_pred_test)[:,1], multi_class='ovr')
    elif num_classes > 2:
        train_auc = roc_auc_score(y_train, np.array(y_pred_train) , multi_class='ovr')
        test_auc = roc_auc_score(y_test, np.array(y_pred_test), multi_class='ovr')
    print(f"train auc: {train_auc}, test auc: {test_auc}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("tree")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument(
        "--configs",
        type=str,
        default="basic_configs_tree",
        help="configure json file path",
    )
    parser.add_argument("--number_of_trees", type=int, default=1, help="number of trees")
    args = parser.parse_args()
    args = load_tree_configs(args.configs, args)
    print("args:", args)
    print("args.number_of_trees:", args.number_of_trees)
    print("******dataset:\t{0}******".format(args.dataset_name))
    if args.dataset_name == "iris":
        data = load_iris()
    elif args.dataset_name == "wine":
        data = load_wine()
    elif args.dataset_name == "breast_cancer":
        data = load_breast_cancer()
    elif args.dataset_name == "digits":
        data = load_digits()

    X = data.data
    y = data.target

    # if args.dataset_name == "GiveMeSomeCredit":
    #     X, y = load_GiveMeSomeCredit()

    # X, y = load_GiveMeSomeCredit()
    # X, y = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=args.seed
    )

    datasets = [
        X_train[:, : int(X_train.shape[1] / 2)],
        X_train[:, int(X_train.shape[1] / 2) :],
    ]

    featureid_lists = [
        range(int(X_train.shape[1] / 2)),
        range(int(X_train.shape[1] / 2), X_train.shape[1]),
    ]

    args.datasets = datasets
    args.y = y_train
    args.featureid_lists = featureid_lists

    print(f"type of model: {args.model_type}, encryption:{args.use_encryption}")


    args = load_tree_parties(args)

  
    tvfl = MainTaskTVFL(args)

    start = time.time()
    tvfl.train()
    end = time.time()
    numc = len(np.unique(y_train))
    print("class num:\t{0}".format(numc))
    print(f"training time: {end - start} [s]")
    evaluate_performance(tvfl, X_train, y_train, X_test, y_test, numc)




