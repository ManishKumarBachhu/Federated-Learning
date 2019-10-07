"""

 NN_classifier.py  (author: Anson Wong / git: ankonzoid)

 We train a multi-layer fully-connected neural network from scratch to classify
 the seeds dataset (https://archive.ics.uci.edu/ml/datasets/seeds). An L2 loss
 function, sigmoid activation, and no bias terms are assumed. The weight
 optimization is gradient descent via the delta rule.

"""
import numpy as np
from src.NeuralNetwork import NeuralNetwork
import src.utils as utils
from sklearn.model_selection import train_test_split 

def main():
    # ===================================
    # Settings
    # ===================================
    csv_filename = "data/seeds_dataset.csv"
    hidden_layers = [5] # number of nodes in hidden layers i.e. [layer1, layer2, ...]
    eta = 0.1 # learning rate
    n_epochs = 400 # number of training epochs
    n_folds = 4 # number of folds for cross-validation
    seed_crossval = 1 # seed for cross-validation
    seed_weights = 1 # seed for NN weight initialization

    # ===================================
    # Read csv data + normalize features
    # ===================================
    print("Reading '{}'...".format(csv_filename))
    X, y, n_classes = utils.read_csv(csv_filename, target_name="y", normalize=True)
    N, d = X.shape
    print(" -> X.shape = {}, y.shape = {}, n_classes = {}\n".format(X.shape, y.shape, n_classes))

    print("Neural network model:")
    print(" input_dim = {}".format(d))
    print(" hidden_layers = {}".format(hidden_layers))
    print(" output_dim = {}".format(n_classes))
    print(" eta = {}".format(eta))
    print(" n_epochs = {}".format(n_epochs))
    print(" n_folds = {}".format(n_folds))
    print(" seed_crossval = {}".format(seed_crossval))
    print(" seed_weights = {}\n".format(seed_weights))

    # ===================================
    # Create cross-validation folds
    # ===================================
    idx_all = np.arange(0, N)
    idx_folds = utils.crossval_folds(N, n_folds, seed=seed_crossval) # list of list of fold indices

    # ===================================
    # Train/evaluate the model on each fold
    # ===================================
    acc_train, acc_valid = list(), list()  # training/test accuracy score
    print("Cross-validating with {} folds...".format(len(idx_folds)))
    for i, idx_valid in enumerate(idx_folds):

        # Collect training and test data from folds
        idx_train = np.delete(idx_all, idx_valid)
        X_train, Y_train = X[idx_train], y[idx_train]
        X_train_1 = X_train[:int(len(X_train)/2)]
        X_train_2 = X_train[int(len(X_train)/2):]
        y_train_1 = Y_train[:int(len(Y_train)/2)]
        y_train_2 = Y_train[int(len(Y_train)/2):]
        
        
        X_valid, y_valid = X[idx_valid], y[idx_valid]

        # Build neural network classifier model and train
        
        ##### Model1 start #######
        #Model1 training and testing
        model1 = NeuralNetwork(input_dim=d, output_dim=n_classes,
                              hidden_layers=hidden_layers, seed=seed_weights)
        
        
        x_train, x_test, y_train, y_test = train_test_split(X_train_1, y_train_1, test_size= 0.5, random_state = 0)
        network1 = model1.train(x_train, y_train, eta=eta, n_epochs=n_epochs)
        
        # Make predictions for training and test data
        ypred_train1 = model1.predict(x_train)
        ypred_valid1 = model1.predict(x_test)

        # Compute training/test accuracy score from predicted values
        acc_train1, acc_valid1 = list(), list()
        acc_train1.append(100*np.sum(y_train==ypred_train1)/len(y_train))
        acc_valid1.append(100*np.sum(y_test==ypred_valid1)/len(y_test))

        # Print cross-validation result
        print(" Model1 : Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_valid = {})".format(
            i+1, n_folds, acc_train1[-1], acc_valid1[-1], len(x_train), len(x_test)))
        
        ##### Model1 end ######

        ##### Model2 start ########
        #Model2 training and testing
        model2 = NeuralNetwork(input_dim=d, output_dim=n_classes,
                              hidden_layers=hidden_layers, seed=seed_weights)
        
        x_train, x_test, y_train, y_test = train_test_split(X_train_2, y_train_2, test_size= 0.5, random_state = 0)
        network2 = model2.train(x_train, y_train, eta=eta, n_epochs=n_epochs)
        
        # Make predictions for training and test data
        ypred_train2 = model2.predict(x_train)
        ypred_valid2 = model2.predict(x_test)

        # Compute training/test accuracy score from predicted values
        acc_train2, acc_valid2 = list(), list()
        acc_train2.append(100*np.sum(y_train==ypred_train2)/len(y_train))
        acc_valid2.append(100*np.sum(y_test==ypred_valid2)/len(y_test))

        # Print cross-validation result
        print(" Model2 : Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_valid = {})".format(
            i+1, n_folds, acc_train2[-1], acc_valid2[-1], len(x_train), len(x_test)))
        
        ###### Model2 end ########
        
        ###### Combining models #########
        #Averaging weights of both models and updating network1
        for p, layer in enumerate(network1):
            # Update weights
            for k, node in enumerate(layer):
                for j, weight in enumerate(node['weights']):
                    node['weights'][j] += network2[p][k]['weights'][j]
                    node['weights'][j] /= 2
                    
        model = NeuralNetwork(input_dim=d, output_dim=n_classes,
                              hidden_layers=hidden_layers, seed=seed_weights)
        
        model.update_nn(network1)
        
        # Make predictions for training and test data
        ypred_train = model.predict(X_train)
        ypred_valid = model.predict(X_valid)

        # Compute training/test accuracy score from predicted values
        acc_train.append(100*np.sum(Y_train==ypred_train)/len(Y_train))
        acc_valid.append(100*np.sum(y_valid==ypred_valid)/len(y_valid))

        # Print cross-validation result
        print(" ModelF : Fold {}/{}: acc_train = {:.2f}%, acc_valid = {:.2f}% (n_train = {}, n_valid = {})".format(
            i+1, n_folds, acc_train[-1], acc_valid[-1], len(X_train), len(X_valid)))
        print()

    # ===================================
    # Print results
    # ===================================
    print("  -> acc_train_avg = {:.2f}%, acc_valid_avg = {:.2f}%".format(
        sum(acc_train)/float(len(acc_train)), sum(acc_valid)/float(len(acc_valid))))

# Driver
if __name__ == "__main__":
    main()