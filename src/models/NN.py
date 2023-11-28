from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from joblib import parallel_backend
from ray.util.joblib import register_ray
from sklearn.model_selection import RandomizedSearchCV

class DistributedNeuralNetwork:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = MLPClassifier()
        self.trained = False
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, params=None):
        # if params is None:
        #     params = {
        #         'hidden_layer_sizes': [(100,), (50, 50), (100, 50, 100)],
        #         'activation': ['relu', 'tanh', 'logistic'],
        #         'solver': ['adam', 'sgd'],
        #         'alpha': [0.0001, 0.001, 0.01]
        #     }
        #
        # randomized_search = RandomizedSearchCV(
        #     self.model, params, cv=5, scoring='accuracy', n_iter=10, random_state=42
        # )
        self.model.fit(X_train, y_train)
        # self.model = randomized_search.best_estimator_
        self.trained = True

    def test(self, X_test):
        if not self.trained:
            print("Model has not been trained yet.")
            return None
        return self.model.predict(X_test)

    def train_and_test(self):
        if self.model is None:
            raise ValueError("Model not selected. Please select a model first.")

        register_ray()

        with parallel_backend('ray'):
            self.train(self.X_train, self.y_train)
            predictions = self.test(self.X_test)

            accuracy = self.calculate_metrics(self.y_test, predictions)

            print(f"Accuracy: {accuracy}")

            data = {'acc':accuracy,'pred':predictions,'true':self.y_test}
            return data

    def calculate_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
