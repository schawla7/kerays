from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import parallel_backend
from ray.util.joblib import register_ray
from sklearn.model_selection import RandomizedSearchCV

class DistributedRandomForest:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.model = RandomForestClassifier()
        self.trained = False
        self.X_train, self.y_train, self.X_test, self.y_test = X_train, y_train, X_test, y_test

    def train(self, X_train, y_train, params=None):
        if params is None:
            params = {
                'n_estimators': [100, 300, 500],
                'max_depth': [None, 3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 6, 8],
                'criterion': ['gini', 'entropy']
            }

        randomized_search = RandomizedSearchCV(
            self.model, params, cv=5, scoring='accuracy', n_iter=10, random_state=42
        )
        randomized_search.fit(X_train, y_train)
        self.model = randomized_search.best_estimator_
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
