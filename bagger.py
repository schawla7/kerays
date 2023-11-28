import ray
import time
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import os
import json
import argparse

#replace hello_world with your training function
@ray.remote(num_cpus=4)
def bagger(pt,ds,t,m):
    os.system("cd /home/schawla7/kerays")
    os.system(f" python3 main.py --problemType {pt} --dataset {ds} --target {t} --model {m}")
    with open('/home/schawla7/kerays/output/result.json','r') as file:
        content = json.load(file)
        return content

ray.init()

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
# print(f"Problem Type: {problem_type}")
print(f"Target Variable: {target_variable}")
print(f"Selected Dataset: {selected_dataset}")

# Create a placement group.
pg = placement_group([{"CPU": 4}], strategy="STRICT_PACK")
r1 = ray.get(bagger.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote(problem_type,selected_dataset,target_variable,model_name))
# Writing modified dictionary back to the JSON file
with open(f"/home/schawla7/kerays/agg_outputs/result-{model_name}.json", 'w') as json_file:
    json.dump(r1, json_file)