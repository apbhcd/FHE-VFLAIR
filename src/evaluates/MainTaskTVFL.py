import os
import sys
import random
from phe import paillier
from models.tree import *
from party.tree_party import *
from utils.he_utils import *

sys.path.append(os.pardir)

class MainTaskTVFL(object):

    def __init__(self, args):
        self.args = args
        self.k = args.k
        self.parties = args.parties
        self.y = args.y
        self.num_classes = len(np.unique(self.y))
        self.model_type = args.model_type

        self.use_encryption = args.use_encryption
        self.key_length = args.key_length
        self.he_scheme = args.he_scheme 
        self.seed = args.seed
        self.number_of_trees = args.number_of_trees
        self.depth = args.depth
        # self.n_job = args.n_job

    def setup_keypair(self):
        print("paillier key length:", self.key_length)
        public_key, private_key = paillier.generate_paillier_keypair(
            n_length=self.key_length
        )
        self.parties[self.k - 1].set_keypair(public_key, private_key)
    
    def setup_ckks_keypair(self):
        HE = generate_ckks_key("16384")
        self.parties[self.k - 1].set_ckks_context(HE)


    def train(self):

        if self.use_encryption:
            if self.he_scheme == "paillier":
                self.setup_keypair()
            elif self.he_scheme == "ckks":
                self.setup_ckks_keypair()

        if self.model_type == "xgboost":
            self.clf = XGBoostClassifier(
                self.num_classes,
                boosting_rounds=self.number_of_trees,
                depth=self.depth,
                active_party_id=self.k - 1,
                use_encryption=self.use_encryption,
                he_scheme=self.he_scheme
                # n_job=self.n_job
            )
        elif self.model_type == "randomforest":
            self.clf = RandomForestClassifier(
                self.num_classes,
                num_trees=self.number_of_trees,
                depth=self.depth,
                active_party_id=self.k - 1,
                use_encryption=self.use_encryption,
            )
        else:
            raise ValueError(f"model_type should be `xgboost` or `randomforest`")

        random.seed(self.seed)

        self.clf.fit(self.parties, self.y)
