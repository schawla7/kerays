import os
import argparse

"""
python3 bagger.py --problemType Text --dataset titanic --target Survived --model NN
"""

parser = argparse.ArgumentParser(description="Script for handling classification or regression problems")

# Adding arguments
parser.add_argument("-p", "--problemType", choices=["Text", "Image"], help="Type of problem: Classification or Regression", required=True)
parser.add_argument("-t", "--targetVariable", type=str, help="Name of the target variable", required=True)
parser.add_argument("-m", "--model", type=str, help="Name of the target variable", required=True)

# Adding mutually exclusive group for optional dataset arguments
dataset_group = parser.add_mutually_exclusive_group(required=True)
dataset_group.add_argument("--kaggle", type=str, help="Use Kaggle dataset")
dataset_group.add_argument("--pytorchDataset", type=str, help="Use PyTorch dataset")
dataset_group.add_argument("--dataset", type=str, help="Specify another dataset name")

# Parsing arguments
args = parser.parse_args()

# Accessing arguments
problem_type = args.problemType
target_variable = args.targetVariable
model_name = args.model

# Determine which dataset argument was used
selected_dataset = None
if args.kaggle:
    selected_dataset = args.kaggle
elif args.pytorchDataset:
    selected_dataset = args.pytorchDataset
else:
    selected_dataset = args.dataset

if problem_type=="Text":
    os.system(f"ray job submit --no-wait --working-dir kerays -- python3 bagger.py --problemType {problem_type} --dataset {selected_dataset} --target {target_variable} --model NN")
    os.system(f"ray job submit --no-wait --working-dir kerays -- python3 bagger.py --problemType {problem_type} --dataset {selected_dataset} --target {target_variable} --model LogisticRegression")
    os.system(f"ray job submit --no-wait --working-dir kerays -- python3 bagger.py --problemType {problem_type} --dataset {selected_dataset} --target {target_variable} --model SVM")
    os.system(f"ray job submit --no-wait --working-dir kerays -- python3 bagger.py --problemType {problem_type} --dataset {selected_dataset} --target {target_variable} --model RandomForest")
    os.system("python3 kerays/result.py")