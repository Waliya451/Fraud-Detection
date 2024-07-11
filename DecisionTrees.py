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
        '''Finds the optimal feature and threshold for splitting the data, 
           maximizing information gain.'''
        
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
        '''Measures the improvement in impurity by splitting the data'''

        #parent Entropy:
        parent_entropy = self._entropy(y)

        #Create Children:
        left_idx,right_idx = self._split(X_column, thr) 
        if len(left_idx) == 0 or len(right_idx) == 0:   #IF ONE SPLIT IS NULL, INVALID SPLIT
            return 0

        #Calculate the weighter abg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)            #NUM. OF SAMPLES IN LEFT & RIGHT 
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])  #ENTROPIES OF LEFT & RIGHT
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r   #COMPUTES WEIGHTED AVG ENTROPY OF CHILD NODE


#         #calculate the IG
#         Parent Entropy: Measures impurity before the split.
#         Child Entropy: Measures impurity after the split.
#         Information Gain: The reduction in impurity due to the split.

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        '''Divides the data into subsets based on a threshold, aiding in tree construction'''

        # argwhere(): Finds indices where the feature value is less/greater than or equal to the threshold.
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column >= split_thresh).flatten()
        return left_idxs, right_idxs
               

    def _entropy(self, y):
        '''Quantifies the impurity of a set of labels, used to calculate information gain.'''
        hist = np.bincount(y)   # bincount(y): Gets the count of each label in y
        ps = hist / len(y)      # calc probabilities
        return -np.sum([p*np.log(p) for p in ps if p > 0])

    def _mostCommonLabel(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverseTree(x, self.root) for x in X])  
        
    def _traverseTree(self, x, node):
        '''Navigates the tree to make predictions for new data points'''
        if node.isLeafNode():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverseTree(x, node.left)
        return self._traverseTree(x, node.right)
    
# #Implementing DT without Python Package

from DecisionTrees import DecisionTree
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from DecisionTrees import DecisionTree
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("./twitter_human_bots_dataset.csv")
dataset.drop(columns=['created_at',
                      'description',
                      'lang',
                      'profile_background_image_url',
                      'id',
                      'screen_name',
                      'location',
                      'profile_image_url'], inplace=True)

dataset["default_profile"] = dataset["default_profile"].astype(str)
dataset["default_profile"] = dataset["default_profile"].replace(['True', 'False'], ['1', '0'])
dataset["default_profile_image"] = dataset["default_profile_image"].astype(str)
dataset["default_profile_image"] = dataset["default_profile_image"].replace(['True', 'False'], ['1', '0'])
dataset["geo_enabled"] = dataset["geo_enabled"].astype(str)
dataset["geo_enabled"] = dataset["geo_enabled"].replace(['True', 'False'], ['1', '0'])
dataset["verified"] = dataset["verified"].astype(str)
dataset["verified"] = dataset["verified"].replace(['True', 'False'],['1','0'])

X = dataset.drop('account_type', axis=1).values
y = dataset['account_type'].values

# Encode the labels
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = DecisionTree(max_depth=5)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(f"Accuracy: {acc}")