# imports for distance dataset
import torch
import numpy as np
from collections import defaultdict
import os
from datetime import datetime
import shutil
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from distance_model import DistanceEstimation, load_data, loss_function

# Function to write to both files
def write_to_files(text, timestamped_file, latest_file):
    timestamped_file.write(text + '\n')
    latest_file.write(text + '\n')

# Loads in the models
distanceEstimation = DistanceEstimation()
distanceEstimation.load_models_for_training()
distanceEstimation.laneModel.eval()
distanceEstimation.objectModel.eval()

# Loads the training data and split it into lanes and objects
data, targets = load_data("../DistanceTestingData/")

# Create output directory if it doesn't exist
output_dir = "testingResults"
os.makedirs(output_dir, exist_ok=True)

# Generate timestamp for filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamped_filename = os.path.join(output_dir, f"testResult_{timestamp}.txt")
latest_filename = f"latestResults.txt"

# Saves the checkpoint used as well
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
lane_timestamped_checkpoint = os.path.join(checkpoint_dir, f"lane_checkpoint_{timestamp}.pth")
object_timestamped_checkpoint = os.path.join(checkpoint_dir, f"object_checkpoint_{timestamp}.pth")
shutil.copy2(distanceEstimation.laneCheckpointPath, lane_timestamped_checkpoint)
shutil.copy2(distanceEstimation.objectCheckpointPath, object_timestamped_checkpoint)

# Counters for tracking filtered data
total_samples = 0
criterion = loss_function

# Initialize lists to store predictions and actual values for analysis
predictions = []
actual_values = []
absolute_errors = []
relative_errors = []
class_numbers = []

# Initialize dictionaries to store class-based statistics
class_predictions = defaultdict(list)
class_actual = defaultdict(list)
class_errors = defaultdict(list)
class_rel_errors = defaultdict(list)

    # Open both files for writing
with open(timestamped_filename, 'w') as ts_file, open(latest_filename, 'w') as latest_file:
    
    write_to_files("Evaluating model predictions vs expected outputs:", ts_file, latest_file)
    write_to_files("=" * 60, ts_file, latest_file)

    # Count samples by model type
    lane_count = 0
    object_count = 0

    # Cycle through the data and make predictions
    with torch.no_grad():  # Disable gradient computation for faster inference
        for i in range(len(data)):
            # Get input features and target for this sample
            input_features = data[i]  # Shape: (5,)
            expected_output = targets[i]  # Scalar
            class_num = int(input_features[0])  # Extract class number
            
            # Estimate the distance
            predicted_output = distanceEstimation.estimate_distance(class_num, data[i][1], data[i][2], data[i][3], data[i][4])
            if class_num == 0:
                lane_count += 1
            else:
                object_count += 1
            
            # Store for analysis
            predictions.append(predicted_output)
            actual_values.append(expected_output)
            class_numbers.append(class_num)
            
            # Calculate errors
            abs_error = abs(predicted_output - expected_output)
            rel_error = (abs_error / expected_output) * 100 if expected_output != 0 else float('inf')
            
            absolute_errors.append(abs_error)
            relative_errors.append(rel_error)
            
            # Store for class-based analysis
            class_predictions[class_num].append(predicted_output)
            class_actual[class_num].append(expected_output)
            class_errors[class_num].append(abs_error)
            class_rel_errors[class_num].append(rel_error)

    # Convert to numpy arrays for easier analysis
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    absolute_errors = np.array(absolute_errors)
    relative_errors = np.array(relative_errors)
    class_numbers = np.array(class_numbers)

    # Filter out infinite values for relative error calculations
    finite_relative_errors = relative_errors[np.isfinite(relative_errors)]

    # Calculate rounded predictions accuracy
    rounded_predictions = np.round(predictions)
    rounded_actual = np.round(actual_values)
    correct_rounded = np.sum(rounded_predictions == rounded_actual)
    total_samples = len(predictions)
    rounded_accuracy = (correct_rounded / total_samples) * 100

    # Calculate within 1 unit accuracy
    within_1_unit = np.sum(absolute_errors <= 1)
    within_1_unit_accuracy = (within_1_unit / total_samples) * 100

    write_to_files("\n" + "=" * 60, ts_file, latest_file)
    write_to_files("EVALUATION SUMMARY:", ts_file, latest_file)
    write_to_files("=" * 60, ts_file, latest_file)

    # Overall statistics
    write_to_files(f"Total samples analyzed: {len(data)}", ts_file, latest_file)
    write_to_files(f"Mean Absolute Error (MAE): {np.mean(absolute_errors):.4f}", ts_file, latest_file)
    write_to_files(f"Root Mean Square Error (RMSE): {np.sqrt(np.mean(absolute_errors**2)):.4f}", ts_file, latest_file)
    if len(finite_relative_errors) > 0:
        write_to_files(f"Mean Relative Error: {np.mean(finite_relative_errors):.2f}%", ts_file, latest_file)
    else:
        write_to_files("Mean Relative Error: N/A (all targets are 0)", ts_file, latest_file)

    # Accuracy percentages - ADDED WITHIN 1 UNIT
    write_to_files(f"Percentage of predictions within 1 unit of actual: {within_1_unit_accuracy:.1f}%", ts_file, latest_file)
    if len(finite_relative_errors) > 0:
        write_to_files(f"Percentage of predictions within 10% of actual: {np.sum(finite_relative_errors <= 10) / len(finite_relative_errors) * 100:.1f}%", ts_file, latest_file)
        write_to_files(f"Percentage of predictions within 20% of actual: {np.sum(finite_relative_errors <= 20) / len(finite_relative_errors) * 100:.1f}%", ts_file, latest_file)
    else:
        write_to_files(f"Percentage of predictions within 5 units: {np.sum(absolute_errors <= 5) / len(absolute_errors) * 100:.1f}%", ts_file, latest_file)
        write_to_files(f"Percentage of predictions within 10 units: {np.sum(absolute_errors <= 10) / len(absolute_errors) * 100:.1f}%", ts_file, latest_file)

    # Rounded accuracy statistics - NOW BELOW THE PERCENTAGE ACCURACIES
    write_to_files(f"Correct predictions (rounded to whole number): {correct_rounded} out of {total_samples}", ts_file, latest_file)
    write_to_files(f"Rounded accuracy: {rounded_accuracy:.2f}%", ts_file, latest_file)

    # Prediction statistics
    write_to_files(f"\nPrediction Range: [{np.min(predictions):.2f}, {np.max(predictions):.2f}]", ts_file, latest_file)
    write_to_files(f"Expected Range: [{np.min(actual_values):.2f}, {np.max(actual_values):.2f}]", ts_file, latest_file)
    write_to_files(f"Mean Prediction: {np.mean(predictions):.2f}", ts_file, latest_file)
    write_to_files(f"Mean Expected: {np.mean(actual_values):.2f}", ts_file, latest_file)

    # Error statistics
    write_to_files(f"\nWorst Absolute Error: {np.max(absolute_errors):.4f}", ts_file, latest_file)
    write_to_files(f"Best Absolute Error: {np.min(absolute_errors):.4f}", ts_file, latest_file)
    write_to_files(f"Std Dev of Errors: {np.std(absolute_errors):.4f}", ts_file, latest_file)

    # Find worst predictions
    worst_indices = np.argsort(absolute_errors)[-5:]  # 5 worst predictions
    write_to_files(f"\n5 WORST PREDICTIONS:", ts_file, latest_file)
    for idx in reversed(worst_indices):
        class_num = int(class_numbers[idx])
        write_to_files(f"Sample {idx}: Expected={actual_values[idx]:.2f}, Predicted={predictions[idx]:.2f}, Error={absolute_errors[idx]:.2f}, Class={class_num}", ts_file, latest_file)

    # Calculate loss using the same criterion as training
    total_loss = 0.0
    with torch.no_grad():
        for i in range(0, len(data), 32):  # Process in batches of 32
            batch_end = min(i + 32, len(data))
            batch_size = batch_end - i
            batch_loss = 0.0
            
            for j in range(i, batch_end):
                input_features = data[j]
                target = targets[j]
                class_num = int(input_features[0])
                
                if class_num == 0 :
                    spatial_features = input_features[1:5]
                    input_tensor = torch.tensor(spatial_features, dtype=torch.float32).unsqueeze(0)
                    prediction = distanceEstimation.laneModel(input_tensor).squeeze()
                else:
                    input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(0)
                    prediction = distanceEstimation.objectModel(input_tensor).squeeze()
                
                sample_loss = criterion(prediction, torch.tensor(target))
                batch_loss += sample_loss.item()
            
            total_loss += batch_loss

    average_loss = total_loss / len(data)
    write_to_files(f"\nAverage Loss (MSE): {average_loss:.6f}", ts_file, latest_file)

    # Optional: Create a simple visualization of predictions vs actual
    write_to_files(f"\nSAMPLE COMPARISON (20 random samples):", ts_file, latest_file)
    write_to_files("Expected | Predicted | Error | Rounded Match | Class", ts_file, latest_file)
    write_to_files("-" * 65, ts_file, latest_file)
    
    # Generate random indices for 20 samples
    random_indices = np.random.choice(len(data), size=min(20, len(data)), replace=False)
    for i in random_indices:
        rounded_match = "✓" if round(predictions[i]) == round(actual_values[i]) else "✗"
        write_to_files(f"{actual_values[i]:8.2f} | {predictions[i]:9.2f} | {absolute_errors[i]:5.2f} | {rounded_match:^13} | {class_numbers[i]:3.0f}", ts_file, latest_file)

    # Count predictions that are exactly 0
    zero_predictions = np.sum(predictions == 0)
    write_to_files(f"\nNumber of predictions that are exactly 0: {zero_predictions} out of {len(predictions)} ({zero_predictions/len(predictions)*100:.1f}%)", ts_file, latest_file)
    
    # CLASS-BASED ANALYSIS
    write_to_files("\n" + "=" * 80, ts_file, latest_file)
    write_to_files("CLASS-BASED ANALYSIS:", ts_file, latest_file)
    write_to_files("=" * 80, ts_file, latest_file)

    # Get all unique classes and sort them
    unique_classes = sorted(class_predictions.keys())
    
    write_to_files(f"Found {len(unique_classes)} different classes: {unique_classes}", ts_file, latest_file)

    class_summary = []
    
    # Calculate all class statistics first for the summary table
    for class_num in unique_classes:
        class_pred = np.array(class_predictions[class_num])
        class_act = np.array(class_actual[class_num])
        class_err = np.array(class_errors[class_num])
        class_rel_err = np.array(class_rel_errors[class_num])
        
        # Calculate rounded accuracy for this class
        class_rounded_pred = np.round(class_pred)
        class_rounded_act = np.round(class_act)
        class_correct_rounded = np.sum(class_rounded_pred == class_rounded_act)
        class_total = len(class_pred)
        class_rounded_accuracy = (class_correct_rounded / class_total) * 100 if class_total > 0 else 0
        
        # Calculate within 1 unit accuracy for this class
        class_within_1_unit = np.sum(class_err <= 1)
        class_within_1_unit_accuracy = (class_within_1_unit / class_total) * 100 if class_total > 0 else 0
        
        # Filter finite relative errors for this class
        finite_class_rel_err = class_rel_err[np.isfinite(class_rel_err)]
        
        num_samples = len(class_pred)
        mae = np.mean(class_err)
        rmse = np.sqrt(np.mean(class_err**2))
        mean_rel_err = np.mean(finite_class_rel_err) if len(finite_class_rel_err) > 0 else np.nan
        
        # Calculate within 10% and 20% accuracy
        if len(finite_class_rel_err) > 0:
            within_10pct = np.sum(finite_class_rel_err <= 10) / len(finite_class_rel_err) * 100
            within_20pct = np.sum(finite_class_rel_err <= 20) / len(finite_class_rel_err) * 100
        else:
            # Use absolute error thresholds if no finite relative errors
            within_10pct = np.sum(class_err <= 5) / len(class_err) * 100  # Within 5 units
            within_20pct = np.sum(class_err <= 10) / len(class_err) * 100  # Within 10 units
        
        # Determine which model was used for this class
        class_indices = np.where(class_numbers == class_num)[0]
        
        # Store for summary table
        class_summary.append({
            'class': class_num,
            'samples': num_samples,
            'mae': mae,
            'rmse': rmse,
            'mean_rel_err': mean_rel_err,
            'mean_pred': np.mean(class_pred),
            'mean_actual': np.mean(class_act),
            'correct_rounded': class_correct_rounded,
            'rounded_accuracy': class_rounded_accuracy,
            'within_1_unit': class_within_1_unit,
            'within_1_unit_accuracy': class_within_1_unit_accuracy,
            'within_10pct': within_10pct,
            'within_20pct': within_20pct
        })

    # Summary table - UPDATED TO INCLUDE WITHIN 1 UNIT COLUMN
    write_to_files("\n" + "=" * 150, ts_file, latest_file)
    write_to_files("CLASS PERFORMANCE SUMMARY TABLE:", ts_file, latest_file)
    write_to_files("=" * 150, ts_file, latest_file)
    write_to_files(f"{'Class':<6} {'Samples':<8} {'MAE':<8} {'RMSE':<8} {'RelErr%':<8} {'Within1U%':<9} {'Within10%':<9} {'Within20%':<9} {'RoundAcc%':<9} {'MeanPred':<9} {'MeanActual':<10}", ts_file, latest_file)
    write_to_files("-" * 150, ts_file, latest_file)
    
    for summary in class_summary:
        rel_err_str = f"{summary['mean_rel_err']:.2f}" if not np.isnan(summary['mean_rel_err']) else "N/A"
        write_to_files(f"{summary['class']:<6} {summary['samples']:<8} {summary['mae']:<8.3f} {summary['rmse']:<8.3f} {rel_err_str:<8} {summary['within_1_unit_accuracy']:<9.1f} {summary['within_10pct']:<9.1f} {summary['within_20pct']:<9.1f} {summary['rounded_accuracy']:<9.1f} {summary['mean_pred']:<9.2f} {summary['mean_actual']:<10.2f}", ts_file, latest_file)

    # Find best and worst performing classes (by MAE and rounded accuracy)
    valid_summaries = [s for s in class_summary if not np.isnan(s['mae'])]
    if valid_summaries:
        best_class_mae = min(valid_summaries, key=lambda x: x['mae'])
        worst_class_mae = max(valid_summaries, key=lambda x: x['mae'])
        best_class_rounded = max(valid_summaries, key=lambda x: x['rounded_accuracy'])
        worst_class_rounded = min(valid_summaries, key=lambda x: x['rounded_accuracy'])
        best_class_within_1_unit = max(valid_summaries, key=lambda x: x['within_1_unit_accuracy'])
        worst_class_within_1_unit = min(valid_summaries, key=lambda x: x['within_1_unit_accuracy'])
        
        write_to_files(f"\nBest performing class (lowest MAE): {best_class_mae['class']} (MAE: {best_class_mae['mae']:.4f})", ts_file, latest_file)
        write_to_files(f"Worst performing class (highest MAE): {worst_class_mae['class']} (MAE: {worst_class_mae['mae']:.4f})", ts_file, latest_file)
        write_to_files(f"Best rounded accuracy class: {best_class_rounded['class']} ({best_class_rounded['rounded_accuracy']:.1f}%)", ts_file, latest_file)
        write_to_files(f"Worst rounded accuracy class: {worst_class_rounded['class']} ({worst_class_rounded['rounded_accuracy']:.1f}%)", ts_file, latest_file)
        write_to_files(f"Best within 1 unit accuracy class: {best_class_within_1_unit['class']} ({best_class_within_1_unit['within_1_unit_accuracy']:.1f}%)", ts_file, latest_file)
        write_to_files(f"Worst within 1 unit accuracy class: {worst_class_within_1_unit['class']} ({worst_class_within_1_unit['within_1_unit_accuracy']:.1f}%)", ts_file, latest_file)

    write_to_files("\nDetailed statistics by class:", ts_file, latest_file)
    write_to_files("-" * 80, ts_file, latest_file)
    
    for class_num in unique_classes:
        class_pred = np.array(class_predictions[class_num])
        class_act = np.array(class_actual[class_num])
        class_err = np.array(class_errors[class_num])
        class_rel_err = np.array(class_rel_errors[class_num])
        
        # Get the corresponding summary data
        class_summary_data = next(s for s in class_summary if s['class'] == class_num)
        
        write_to_files(f"\nCLASS {class_num}:", ts_file, latest_file)
        write_to_files(f"  Samples: {class_summary_data['samples']}", ts_file, latest_file)
        write_to_files(f"  Mean Absolute Error: {class_summary_data['mae']:.4f}", ts_file, latest_file)
        write_to_files(f"  Root Mean Square Error: {class_summary_data['rmse']:.4f}", ts_file, latest_file)
        if not np.isnan(class_summary_data['mean_rel_err']):
            write_to_files(f"  Mean Relative Error: {class_summary_data['mean_rel_err']:.2f}%", ts_file, latest_file)
        else:
            write_to_files(f"  Mean Relative Error: N/A (all targets are 0)", ts_file, latest_file)
        write_to_files(f"  Within 1 unit accuracy: {class_summary_data['within_1_unit_accuracy']:.1f}%", ts_file, latest_file)
        write_to_files(f"  Within 10% accuracy: {class_summary_data['within_10pct']:.1f}%", ts_file, latest_file)
        write_to_files(f"  Within 20% accuracy: {class_summary_data['within_20pct']:.1f}%", ts_file, latest_file)
        write_to_files(f"  Correct rounded predictions: {class_summary_data['correct_rounded']} out of {class_summary_data['samples']}", ts_file, latest_file)
        write_to_files(f"  Rounded accuracy: {class_summary_data['rounded_accuracy']:.2f}%", ts_file, latest_file)
        write_to_files(f"  Mean Prediction: {class_summary_data['mean_pred']:.2f}", ts_file, latest_file)
        write_to_files(f"  Mean Actual: {class_summary_data['mean_actual']:.2f}", ts_file, latest_file)
        write_to_files(f"  Prediction Range: [{np.min(class_pred):.2f}, {np.max(class_pred):.2f}]", ts_file, latest_file)
        write_to_files(f"  Actual Range: [{np.min(class_act):.2f}, {np.max(class_act):.2f}]", ts_file, latest_file)
        write_to_files(f"  Worst Error: {np.max(class_err):.4f}", ts_file, latest_file)
        write_to_files(f"  Best Error: {np.min(class_err):.4f}", ts_file, latest_file)

    # Show some examples from each class
    write_to_files("\n" + "=" * 80, ts_file, latest_file)
    write_to_files("SAMPLE PREDICTIONS BY CLASS (3 random from each class):", ts_file, latest_file)
    write_to_files("=" * 80, ts_file, latest_file)
    
    for class_num in unique_classes:
        class_indices = np.where(class_numbers == class_num)[0]
        
        write_to_files(f"\nClass {class_num} samples:", ts_file, latest_file)
        write_to_files("Expected | Predicted | Error | Rounded Match", ts_file, latest_file)
        write_to_files("-" * 45, ts_file, latest_file)
        
        # Randomly select 3 samples (or all if less than 3)
        num_samples = min(3, len(class_indices))
        if num_samples > 0:
            random_class_indices = np.random.choice(class_indices, size=num_samples, replace=False)
            for idx in random_class_indices:
                rounded_match = "✓" if round(predictions[idx]) == round(actual_values[idx]) else "✗"
                write_to_files(f"{actual_values[idx]:8.2f} | {predictions[idx]:9.2f} | {absolute_errors[idx]:5.2f} | {rounded_match:^13}", ts_file, latest_file)
        else:
            write_to_files("No samples available for this class", ts_file, latest_file)
    
    # Add timestamp information at the end
    write_to_files(f"\n" + "=" * 60, ts_file, latest_file)
    write_to_files(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ts_file, latest_file)
    write_to_files(f"Results saved to: {timestamped_filename}", ts_file, latest_file)

print(f"Testing completed! Results saved to:")
print(f"  - {timestamped_filename}")
print(f"  - {latest_filename}")