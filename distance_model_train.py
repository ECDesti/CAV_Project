"""

Script to be used to train the models

"""

# The required imports
import glob
from distance_model import DistanceEstimation, load_data, loss_function
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Loads in the models
distanceEstimation = DistanceEstimation()
distanceEstimation.load_models_for_training()
distanceEstimation.laneModel.train()
distanceEstimation.objectModel.train()

# Initialises required variables
startEpoch = 0
if distanceEstimation.objectCheckpoint is not None:
    startEpoch = distanceEstimation.objectCheckpoint['epoch']
objectLowestLoss = float('inf')
if distanceEstimation.objectCheckpoint is not None:
    objectLowestLoss = distanceEstimation.objectCheckpoint['loss']
laneLowestLoss = float('inf')
if distanceEstimation.laneCheckpoint is not None:
    laneLowestLoss = distanceEstimation.laneCheckpoint['loss']
targetEpochs = 1000
lossFunction = loss_function

# Loads the training data and split it into lanes and objects
data, targets = load_data()

# Convert to tensors once before training
data_tensor = torch.tensor(data, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32)

# Separate lane and object data
lane_mask = data_tensor[:, 0] == 0
lane_dataset = TensorDataset(data_tensor[lane_mask][:, 1:5], targets_tensor[lane_mask])
lane_loader = DataLoader(lane_dataset, batch_size=32, shuffle=True)

object_mask = data_tensor[:, 0] != 0
object_dataset = TensorDataset(data_tensor[object_mask], targets_tensor[object_mask])
object_loader = DataLoader(object_dataset, batch_size=32, shuffle=True)

print(f"Lanes: {len(lane_dataset)}, Objects: {len(object_dataset)}")

# Starts the training
print("Starting Training")
try:
    for epoch in range(startEpoch, targetEpochs):
        laneTotalLoss = 0
        objectTotalLoss = 0
        
        # Train lane model
        for batch_data, batch_targets in lane_loader:
            distanceEstimation.laneOptimiser.zero_grad()
            output = distanceEstimation.laneModel(batch_data).squeeze()
            loss = lossFunction(output, batch_targets)
            loss.backward()
            distanceEstimation.laneOptimiser.step()
            laneTotalLoss += loss.item()
        
        # Train object model
        for batch_data, batch_targets in object_loader:
            distanceEstimation.objectOptimiser.zero_grad()
            output = distanceEstimation.objectModel(batch_data).squeeze()
            loss = lossFunction(output, batch_targets)
            loss.backward()
            distanceEstimation.objectOptimiser.step()
            objectTotalLoss += loss.item()
            
        print("=" * 50)
        print(f"Epoch {epoch + 1}")

        laneStatement = f"LANES: Total Loss: {laneTotalLoss:.4f}, Average Loss: {(laneTotalLoss / len(lane_dataset)):.4f}"
        if laneTotalLoss < laneLowestLoss:
            laneLowestLoss = laneTotalLoss
            laneStatement += " (NEW BEST!)"
            
            # Creates and saves a new checkpoint
            checkpoint = {
                'epoch' : epoch,
                'model_state_dict': distanceEstimation.laneModel.state_dict(),
                'optimiser_state_dict': distanceEstimation.laneOptimiser.state_dict(),
                'loss': laneTotalLoss
            }
            torch.save(checkpoint, distanceEstimation.laneCheckpointPath)

        objectStatement = f"OBJECTS: Total Loss: {objectTotalLoss:.4f}, Average Loss: {(objectTotalLoss / len(object_dataset)):.4f}"
        if objectTotalLoss < objectLowestLoss:
            objectLowestLoss = objectTotalLoss
            objectStatement += " (NEW BEST!)"
            
            # Creates and saves a new checkpoint
            checkpoint = {
                'epoch' : epoch,
                'model_state_dict': distanceEstimation.objectModel.state_dict(),
                'optimiser_state_dict': distanceEstimation.objectOptimiser.state_dict(),
                'loss': objectTotalLoss
            }
            torch.save(checkpoint, distanceEstimation.objectCheckpointPath)
        
        print(laneStatement)
        print(objectStatement)

except KeyboardInterrupt:
    print(f"\nTraining interrupted by user")
# except Exception as e:
#     print(f"\nTraining stopped due to error: {e}")