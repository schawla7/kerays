import os
import json
from collections import Counter
from sklearn.metrics import accuracy_score

# Directory containing JSON files
directory = '/home/sl203/kerays/result'


# Function to read JSON files, aggregate 'pred' lists, and compare with 'true'
def process_and_compare(directory):
    # Dictionary to store aggregated values
    aggregated_values = {}
    individual_predictions = []
    individual_truth = []
    individual_acc = []
    true_values = None

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Check if the file starts with 'result-' and ends with '.json'
        if filename.startswith('output-') and filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Read JSON content from the file
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Extract 'pred' list from the dictionary
                pred_list = data.get('pred', [])
                true_list = data.get('true',[])
                acc = data.get('acc')
                individual_predictions.append(pred_list)
                individual_truth.append(true_list)
                individual_acc.append(acc)


                # Aggregate 'pred' lists
                for index, value in enumerate(pred_list):
                    if index not in aggregated_values:
                        aggregated_values[index] = []
                    aggregated_values[index].append(value)

            # Delete the file after processing
            os.remove(file_path)

    # Find the majority value at each index across all 'pred' lists
    for index, values in aggregated_values.items():
        counter = Counter(values)
        majority_value = counter.most_common(1)[0][0]
        aggregated_values[index] = majority_value

    return aggregated_values, individual_predictions, true_values, individual_acc, individual_truth


# Call the function and get the aggregated values, individual predictions, and true values
aggregated_preds, individual_preds, true_values,individual_acc,individual_truth = process_and_compare(directory)

# Print 'true' values
print("True Values:")
print(true_values)

# Print aggregated values
print("\nAggregated Predictions:")
print(aggregated_preds)
print("\nIndividual Predictions:")
# Print individual predictions
c = 0
for filename, preds in enumerate(individual_preds):
    print(f"File {filename + 1}:")
    print("Y Pred: \n")
    print(preds)
    print()
    print("Accuracy: \n")
    print(individual_acc[c])
    print()
    print("Y Test: \n")
    print(individual_truth[c])
    print()
    c += 1

# Calculate accuracy score using sklearn
accuracy = accuracy_score(true_values, [aggregated_preds[i] for i in range(len(true_values))])

# Print accuracy score
print(f"\nAccuracy Score: {accuracy}")