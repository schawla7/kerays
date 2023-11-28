import os
import json
from collections import Counter
from sklearn.metrics import accuracy_score

# Directory containing JSON files
directory = '/home/schawla7/kerays/agg_outputs'


# Function to read JSON files, aggregate 'pred' lists, and compare with 'true'
def process_and_compare(directory):
    aggregated_values = {}
    individual_predictions = []
    true_values = []

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Check if the file starts with 'result-' and ends with '.json'
        if filename.startswith('result-') and filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Read JSON content from the file
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Extract 'pred' list from the dictionary
                pred_list = data.get('pred', [])
                individual_predictions.append(pred_list)

                # Extract 'true' values from each file
                true_val = data.get('true', [])
                true_values.append(true_val)

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

    return aggregated_values, individual_predictions, true_values


# Call the function and get the aggregated values, individual predictions, and true values
aggregated_preds, individual_preds, true_values = process_and_compare(directory)

# Print 'true' values from all dictionaries
print("True Values from all Dictionaries:")
for idx, true_val in enumerate(true_values):
    print(f"Dictionary {idx + 1}: {true_val}")

# Calculate and print accuracies for each dictionary against its true values
for idx, pred_values in enumerate(individual_preds):
    accuracy = accuracy_score(true_values[idx], pred_values)
    print(f"\nAccuracy for Dictionary {idx + 1}: {accuracy}")

# Choose a true value to compare with aggregated predictions
chosen_true = true_values[0]  # Change the index to choose a different true value for comparison

# Calculate accuracy of aggregated predictions against the chosen true value
aggregated_accuracy = accuracy_score(chosen_true, [aggregated_preds[i] for i in range(len(chosen_true))])
print(f"\nAggregated Prediction Accuracy against Chosen True Value: {aggregated_accuracy}")
