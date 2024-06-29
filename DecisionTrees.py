import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def isLeafNode(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._growTree(X, y)
        pass

    def _growTree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        #Check the stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or
            n_samples < self.min_samples_split):

            leaf_node_value = self._mostCommonLabel(y)
            return Node(value=leaf_node_value)
        
        #Find the best Split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_Feature, best_thresh = self._bestSplit(X,y,feat_idxs)

        #Create Child Nodes
        left_idxs, right_idxs = self._split(X[:,best_Feature ], best_thresh)
        left = self._growTree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._growTree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_Feature, best_thresh, left, right)
    def _bestSplit(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            # print(X_column)
            thresholds = np.unique(X_column)
            for thr in thresholds:
                #calculate the information gain:
                gain = self._informationGain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold
    
    def _informationGain(self, y, X_column, thr):
        #parent Entropy:
        parent_entropy = self._entropy(y)

        #Create Children:
        left_idx,right_idx = self._split(X_column, thr) 
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        #Calculate the weighter abg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r


        #calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column >= split_thresh).flatten()
        return left_idxs, right_idxs
               

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log(p) for p in ps if p > 0])

    def _mostCommonLabel(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverseTree(x, self.root) for x in X])  
        
    def _traverseTree(self, x, node):
        if node.isLeafNode():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverseTree(x, node.left)
        return self._traverseTree(x, node.right)