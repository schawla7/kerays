import ray
import time
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import os
import json
import argparse

@ray.remote(num_cpus=4)
def hello_world(modelname):
    os.system(f"python3 image_script.py -m {modelname}")
    with open(f'/home/sl203/result/output-{modelname}.json','r') as file:
        content = json.load(file)
        return content

if __name__ == "__main__":
    # Creating argument parser
    parser = argparse.ArgumentParser(description="Script for handling image classification")

    parser.add_argument("-m", "--targetModel", type=str, help="Name of the model to be used [Resnet, Alexnet]", required=True)
    
    # Parsing arguments
    args = parser.parse_args()

    target_model = args.targetModel
    
    ray.init()

    # Create a placement group.
    pg = placement_group([{"CPU": 4}], strategy="STRICT_PACK")

    # Get a list of all remote function handles
    #remote_handles = [hello_world.remote() for _ in range(int(ray.cluster_resources()["CPU"]))]
    result = ray.get(hello_world.options(scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg)).remote(target_model))
    print(result)
    
    with open(f'kerays/result/output-{target_model}.json', 'w') as json_file:
        json.dump(result, json_file)
        
    #time.sleep(1)