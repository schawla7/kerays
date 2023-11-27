from joblib import parallel_backend # added line.
from ray.util.joblib import register_ray # added line.
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from src.models.DecisionTree import DistributedDecisionTree
from src.models.LogisticRegression import DistributedLogisticRegression
from src.models.RandomForest import DistributedRandomForest
from src.models.SVM import DistributedSVM
from src.models.NN import DistributedNeuralNetwork

class TextClassifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def select_model(self, X_train,y_train,X_test,y_test):
        if self.model_name == 'LogisticRegression':
            model = DistributedLogisticRegression(X_train,y_train,X_test,y_test)
        elif self.model_name == 'RandomForest':
            model = DistributedRandomForest(X_train,y_train,X_test,y_test)
        elif self.model_name == 'DecisionTreeClassifier':
            model = DistributedDecisionTree(X_train,y_train,X_test,y_test)
        elif self.model_name == "SVM":
            model = DistributedSVM(X_train,y_train,X_test,y_test)
        elif self.model_name == "NN":
            model = DistributedNeuralNetwork(X_train,y_train,X_test,y_test)
        else:
            raise ValueError("Invalid model name provided in the configuration file.")
        model.train_and_test()