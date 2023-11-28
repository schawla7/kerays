import argparse
import json

from src.models.text_classifier import TextClassifier
from src.preprocessing.data_preprocessor import DataPreprocessor


def call_preprocessor(selected_dataset, target_variable):
    # Create an instance of the DataPreprocessor class
    data_processor = DataPreprocessor(target_variable)

    # Call the preprocess_data method by providing the file path
    X_train, y_train, X_test, y_test = data_processor.preprocess_data(f"/home/schawla7/kerays/data/{selected_dataset}.csv")
    return X_train, y_train, X_test, y_test

def trainer(X_train, y_train, X_test, y_test,model):
    model_selector = TextClassifier(model)
    result  = model_selector.select_model(X_train,y_train,X_test,y_test)
    return result

def main():
    # Creating argument parser
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

    # Printing out the received arguments
    print(f"Problem Type: {problem_type}")
    print(f"Target Variable: {target_variable}")
    print(f"Selected Dataset: {selected_dataset}")
    X_train, y_train, X_test, y_test = call_preprocessor(selected_dataset, target_variable)
    result = trainer(X_train, y_train, X_test, y_test, model_name)
    return result

def dump_results(results):
    # File path where you want to save the JSON
    file_path = "/home/schawla7/kerays/output/result.json"
    results["pred"] = results["pred"].tolist()
    results["true"] = results["true"].tolist()

    # Writing dictionary to JSON file
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file)

if __name__ == "__main__":
    result = main()
    dump_results(result)
