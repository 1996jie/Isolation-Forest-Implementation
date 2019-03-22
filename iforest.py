import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import multiprocessing as mp

pool = mp.Pool()

class IsolationTree:
    def __init__(self, X, height_limit, p=None, q=None, n_nodes=0):
        self.height_limit = height_limit
        self.root = InNode(p,q)
        self.n_nodes = n_nodes
    
    def fit(self, X:np.ndarray, height_limit, improved=False): 
        if len(X)<=1 or self.height_limit==0:
            self.root = ExNode(size=len(X))
            return self.root
        
        ncols = len(X[0])
        if improved==True:
            q=np.random.randint(0, ncols, 3)
            min_x = np.min(X[:,q], axis=0)
            max_x = np.max(X[:,q], axis=0)
            if (min_x==max_x).astype(int).sum()==3:  # ExNode
                self.root=ExNode(size=len(X))
                return self.root
            elif (min_x!=max_x).astype(int).sum()==1:  # one col left
                q = np.where(min_x!=max_x)[0][0]
                x_col = X[:,q]
                p=np.random.uniform(min(x_col), max(x_col))
                Xl = X[X[:,q]<p]
                Xr = X[X[:,q]>=p]
                self.root = InNode(p=p, q=q)
                self.root.left = IsolationTree(Xl, self.height_limit-1, p, q, self.n_nodes).fit(Xl, self.height_limit-1, improved)
                self.root.right = IsolationTree(Xr, self.height_limit-1, p, q, self.n_nodes).fit(Xr, self.height_limit-1, improved)
                self.n_nodes = self.getnodes(self.root)
                return self.root
            elif (min_x!=max_x).astype(int).sum()==2:  # two cols left
                q = np.where(min_x!=max_x)[0]  # q has two values
                x_col = X[:,q]
                # min_x_col, max_x_col = np.min(x_col, axis=0), np.max(x_col, axis=0)
                min_x_col, max_x_col = min_x[q], max_x[q]
                p1, p2 = np.random.uniform(min_x_col[0], max_x_col[0], 3), np.random.uniform(min_x_col[1], max_x_col[1], 3)
                len_list = []
                for p in p1:
                    if len(X[X[:,q[0]]<p])<len(X[X[:,q[0]]>=p]): 
                        len_list.append((len(X[X[:,q[0]]<p]), (q[0], p)))
                    else: len_list.append((len(X[X[:,q[0]]>=p]), (q[0], p)))
                for p in p2:
                    if len(X[X[:,q[1]]<p])<len(X[X[:,q[1]]>=p]): 
                        len_list.append((len(X[X[:,q[1]]<p]), (q[1], p)))
                    else: len_list.append((len(X[X[:,q[1]]>=p]), (q[1], p)))
                q, p = min(len_list, key=lambda x: x[0])[1]
                Xl = X[X[:,q]<p]
                Xr = X[X[:,q]>=p]
                self.root = InNode(p=p, q=q)
                self.root.left = IsolationTree(Xl, self.height_limit-1, p, q, self.n_nodes).fit(Xl, self.height_limit-1)
                self.root.right = IsolationTree(Xr, self.height_limit-1, p, q, self.n_nodes).fit(Xr, self.height_limit-1)
                self.n_nodes = self.getnodes(self.root)
                return self.root
            else:
                x_col = X[:,q]
                p1 = np.random.uniform(min_x[0], max_x[0], 3)
                p2 = np.random.uniform(min_x[1], max_x[1], 3)
                p3 = np.random.uniform(min_x[2], max_x[2], 3)
                len_list = []
                for i, j in enumerate([p1, p2, p3]):
                    for p in j:
                        if len(X[X[:,q[i]]<p])<len(X[X[:,q[i]]>=p]): 
                            len_list.append((len(X[X[:,q[i]]<p]), (q[i], p)))
                        else: len_list.append((len(X[X[:,q[i]]>=p]), (q[i], p)))
                q, p = min(len_list, key=lambda x: x[0])[1]
                Xl = X[X[:,q]<p]
                Xr = X[X[:,q]>=p]
                self.root = InNode(p=p, q=q)
                self.root.left = IsolationTree(Xl, self.height_limit-1, p, q, self.n_nodes).fit(Xl, self.height_limit-1)
                self.root.right = IsolationTree(Xr, self.height_limit-1, p, q, self.n_nodes).fit(Xr, self.height_limit-1)
                self.n_nodes = self.getnodes(self.root)
                return self.root

        if improved==False:
            q = np.random.randint(0, ncols) 
            max_x, min_x = max(X[:,q]), min(X[:,q])
            if max_x <= min_x: 
                self.root=ExNode(size=len(X))
                return self.root#ExNode(size=len(X))
            else:
                p = np.random.uniform(min_x, max_x)
                Xl = X[X[:,q]<p]
                Xr = X[X[:,q]>=p]
                self.root = InNode(p=p, q=q)
                self.root.left = IsolationTree(Xl, self.height_limit-1, p, q, self.n_nodes).fit(Xl, self.height_limit-1)
                self.root.right = IsolationTree(Xr, self.height_limit-1, p, q, self.n_nodes).fit(Xr, self.height_limit-1)
                self.n_nodes = self.getnodes(self.root)
                return self.root
    
    def getnodes(self, root):
        if root is None:
            return 0
        else:
            return self.getnodes(root.left)+self.getnodes(root.right)+1

class InNode:
    def __init__(self, p, q, left=None, right=None):
        self.left = left
        self.right = right
        self.p = p 
        self.q = q 
        
        
class ExNode:
    def __init__(self, size, left=None, right=None):
        self.size = size
        self.left = left
        self.right = right

class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        self.height_limit = np.ceil(np.log2(self.sample_size))
        
    def fit(self, X:np.ndarray, improved=False):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        for i in range(self.n_trees):
            X_sample = X[np.random.choice(len(X), self.sample_size, replace=False)]
            t = IsolationTree(X_sample, height_limit=self.height_limit)
            t.fit(X_sample, self.height_limit, improved)
            self.trees.append(t)
        
        return self.trees

    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """
        lengths = []
        if isinstance(X, pd.DataFrame):
            X = X.values

        def length(x, node, e=0):
            if isinstance(node, ExNode):
                size = node.size
                if size>1:
                    return e+2*(np.log(size-1)+0.5772156649)-2*(size-1.)/size*(1.)
                else: return e
            else:
                if x[node.q]<node.p:
                    return length(x, node.left, e+1)
                else:
                    return length(x, node.right, e+1)

        for x in X:
            x_weights = []
            for tree in self.trees:
                x_weights.append(length(x, tree.root))
            lengths.append(np.mean(x_weights))
        return np.array(lengths).reshape(-1, 1)
        
    
    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        lengths = self.path_length(X)
        if self.sample_size>2:
            c = 2*(np.log(self.sample_size-1)+0.5772156649)-2*(self.sample_size-1)/len(X)
        elif self.sample_size==2:
            c = 1
        else: 
            c=0
        scores = []
        for length in lengths:
            scores.append(np.power(2, (-1)*length/c))
        return np.array(scores).reshape(-1,1)
        

    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        return (scores>=threshold).astype(int)
    

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold=1.0
    y_pred = np.array([int(score>=threshold) for score in scores])
    confusion = confusion_matrix(y, y_pred)
    TN, FP, FN, TP = confusion.flat
    TPR = TP / (TP + FN)

    while TPR < desired_TPR:
        threshold -= 0.01
        y_pred = np.array([int(score>=threshold) for score in scores])
        confusion = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
    
    FPR = FP / (FP + TN)
    return threshold, FPR