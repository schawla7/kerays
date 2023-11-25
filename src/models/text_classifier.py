from joblib import parallel_backend # added line.
from ray.util.joblib import register_ray # added line.
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from src.models.DecisionTree import DistributedDecisionTree
from src.models.LogisticRegression import DistributedLogisticRegression

class TextClassifier:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def select_model(self, X_train,y_train,X_test,y_test):
        if self.model_name == 'LogisticRegression':
            model = DistributedLogisticRegression(X_train,y_train,X_test,y_test)
        elif self.model_name == 'RandomForest':
            self.model = RandomForestClassifier()
        elif self.model_name == 'DecisionTreeClassifier':
            model = DistributedDecisionTree(X_train,y_train,X_test,y_test)
        elif self.model_name == "SVM":
            self.model = SVC()
        elif self.model_name == "NN":
            self.model = MLPClassifier()
        else:
            raise ValueError("Invalid model name provided in the configuration file.")
        model.train_and_test()