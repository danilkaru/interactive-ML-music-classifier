# Helper file containing ML-related functions

from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier


def format_prob_string(probabilities, classes, num_decimals=7):
    """
    Function which formats provided probability values to a string
    """
    return "[" + ", ".join([f"{str(cls)[:5]}: {prob:.{num_decimals}f}" for cls, prob in zip(classes, probabilities)]) + "]"


def compute_kfold_scores(X_labeled, Y_labeled, k):
    """
    Function which handles KFold accuracy score calculation for provided labeled data
    TO DO: ensure that the function expands training set with previous fold after every iteration
    """
    # If number of embeddings is smaller than k, then take that as the value for KFold
    n_splits = min(len(X_labeled),k) 

    # Retrieve indeces for KFold using sklearn KFold
    kf = KFold(n_splits)

    # Initialise scores
    scores = []

    acc_train_indeces = []
    acc_test_indeces = []

    # Loop over retrieved indeces 
    for train_indices, test_indices in kf.split(X_labeled):
        train_X = acc_train_indeces + [X_labeled[i] for i in train_indices]
        train_Y = acc_train_indeces +[Y_labeled[i] for i in train_indices]
        test_X = acc_test_indeces + [X_labeled[i] for i in test_indices]
        test_Y = acc_test_indeces + [Y_labeled[i] for i in test_indices]
        
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100,), max_iter=500, random_state=1)
        
        clf.fit(train_X, train_Y)
        score = clf.score(test_X, test_Y)
        scores.append(score)
        # print("MLP accuracy score per iteration: "+str(score))

    return scores

    # print("Average CV Accuracy:", np.mean(scores))    

