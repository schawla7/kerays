import ray
from ray.data import read_csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from ray.data.preprocessors import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from joblib import parallel_backend # added line.
from ray.util.joblib import register_ray # added line.


ray.init(num_cpus=4)

def load_processed_iris():
    iris_dset = read_csv("/home/schawla7/kerays/data/iris.csv")

    # Scale features
    minmax_scaler = MinMaxScaler(columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])
    minmax_scaler.fit(iris_dset)
    scaled_features = minmax_scaler.transform(iris_dset)
    return scaled_features

# Load preprocessed data
iris_dset = load_processed_iris()

# Split dataset into train and test
train_df, test_df = iris_dset.train_test_split(test_size=0.25)

train_df = train_df.to_pandas()
test_df = test_df.to_pandas()

# Obtain X and Y splits
X_train, y_train = train_df.drop('Species', axis=1), train_df['Species']
X_test, y_test = test_df.drop('Species', axis=1), test_df['Species']

register_ray() # added line.

model = SVC(kernel='rbf')

with parallel_backend('ray'): # added line.
    model.fit(X_train,y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the metrics
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print('Confusion Matrix:')
print(conf_matrix)



"""

from ray.train.torch import TorchTrainer
import torch.optim as optim
from ray.train import ScalingConfig
import torch
import torch.nn as nn

# Convert to torch Datasets
train_ds = train_df.to_torch(
    label_column="Species",
    feature_columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    feature_column_dtypes=torch.float32,
    label_column_dtype=torch.float32
)
test_ds = test_df.to_torch(
    label_column="Species",
    feature_columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"],
    feature_column_dtypes=torch.float32,
    label_column_dtype=torch.float32
)

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32)


def train_func(config):
    # Model, Loss, Optimizer
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 1)  # Output has 3 classes
    )
    # [1] Prepare model.
    model = ray.train.torch.prepare_model(model)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    # Training
    for epoch in range(10):
        for features, labels in train_loader:
            outputs = model(features)
            print(outputs.shape)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # [3] Report metrics and checkpoint.
        ray.train.report({"loss": loss.item()})

# [4] Configure scaling and resource requirements.
scaling_config = ScalingConfig(num_workers=2, use_gpu=False)

# [5] Launch distributed training job.
trainer = TorchTrainer(train_func, scaling_config=scaling_config)
result = trainer.fit()

print(result.metrics)    # The metrics reported during training.

"""