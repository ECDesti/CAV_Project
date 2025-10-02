"""

This script contains the definitions for:
1. The ObjectDistanceModel class
2. The LaneDistanceModel class
3. The DistanceEstimation class

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import glob

class ObjectDistanceModel(nn.Module):
    def __init__(self, num_classes=25, spatial_input_dim=4, class_embed_dim=8, hidden_dim=16, output_dim=1):

        # Initialises the parent class which is nn.Module
        super(ObjectDistanceModel, self).__init__()
        
        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
        
        # Class to gate weights
        self.gate_generator = nn.Sequential(
            nn.Linear(class_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # Gate values between 0 and 1
        )
        
        # Spatial feature processing
        self.spatial_processor = nn.Sequential(
            nn.Linear(spatial_input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, input):
        class_num = input[:, 0].long() # converts to torch.long data type
        spatial_features = input[:, 1:5]  # x_center, y_center, width, height
        
        # Generate gates from class information
        class_embed = self.class_embedding(class_num)
        gates = self.gate_generator(class_embed)
        
        # Process spatial features
        spatial_processed = self.spatial_processor(spatial_features)
        
        # Apply class-dependent gating
        gated_features = spatial_processed * gates
        
        # Final processing
        output = self.final_layers(gated_features)
        return F.softplus(output)



class LaneDistanceModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=1):
        
        # Initialises the parent class which is nn.Module
        super(LaneDistanceModel, self).__init__()
        
        # Larger feedforward network with more layers and wider hidden dimensions
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, input):
        # Runs the input through the model
        output = self.layers(input)
        # Ensures output is positive
        return F.softplus(output)

class DistanceEstimation:
    def __init__(self):
        # Object model variables
        self.objectCheckpointPath = "object_distance_model_checkpoint.pth"
        self.objectModel = None
        self.objectOptimiser = None
        self.objectCheckpoint = None

        # Lane model variables
        self.laneCheckpointPath = "lane_distance_model_checkpoint.pth"
        self.laneModel = None
        self.laneOptimiser = None
        self.laneCheckpoint = None
    
    def load_models(self, object_checkpoint_path = "object_distance_model_checkpoint.pth", lane_checkpoint_path = "lane_distance_model_checkpoint.pth"):
        self.objectCheckpointPath = object_checkpoint_path
        self.laneCheckpointPath = lane_checkpoint_path
        
        if os.path.exists(object_checkpoint_path) and os.path.exists(lane_checkpoint_path):
            # Loads the object detection checkpoint
            self.objectCheckpoint = torch.load(object_checkpoint_path)
            self.objectModel = ObjectDistanceModel()
            self.objectModel.load_state_dict(self.objectCheckpoint['model_state_dict'])
            self.objectOptimiser = torch.optim.Adam(self.objectModel.parameters())
            self.objectOptimiser.load_state_dict(self.objectCheckpoint['optimiser_state_dict'])
            print("Successfully loaded Object Distance Estimation model")
            
            # Loads the lane detection checkpoint
            self.laneCheckpoint = torch.load(lane_checkpoint_path)
            self.laneModel = LaneDistanceModel()
            self.laneModel.load_state_dict(self.laneCheckpoint['model_state_dict'])
            self.laneOptimiser = torch.optim.Adam(self.laneModel.parameters())
            self.laneOptimiser.load_state_dict(self.laneCheckpoint['optimiser_state_dict'])
            print("Successfully loaded Lane Distance Estimation model")
        else:
            print(f"Object Checkpoint Path of {object_checkpoint_path} found: {os.path.exists(object_checkpoint_path)}")
            print(f"Lane Checkpoint Path of {lane_checkpoint_path} found: {os.path.exists(lane_checkpoint_path)}")
            print("FAILED TO LOAD MODELS")    
    
    def initialise_models(self, object_checkpoint_path = "object_distance_model_checkpoint.pth", lane_checkpoint_path = "lane_distance_model_checkpoint.pth"):
        self.objectCheckpointPath = object_checkpoint_path
        self.laneCheckpointPath = lane_checkpoint_path

        # Initialises the Object Distance Estimation Model
        self.objectModel = ObjectDistanceModel()
        self.objectOptimiser = torch.optim.Adam(self.objectModel.parameters())
        print("Initialised Object Distance Estimation Model")

        # Initialises the Lane Distance Estimation Model
        self.laneModel = LaneDistanceModel()
        self.laneOptimiser = torch.optim.Adam(self.laneModel.parameters())
        print("Initialised Lane Distance Estimation Model")
    
    def load_models_for_training(self, object_checkpoint_path = "object_distance_model_checkpoint.pth", lane_checkpoint_path = "lane_distance_model_checkpoint.pth"):
        self.objectCheckpointPath = object_checkpoint_path
        self.laneCheckpointPath = lane_checkpoint_path
        
        # Loads models if they exist, otherwise initialise them
        if os.path.exists(object_checkpoint_path) and os.path.exists(lane_checkpoint_path):
            self.load_models()
        else:
            self.initialise_models()

    def estimate_distance(self, class_num, x_center, y_center, width, height):
        if (self.laneModel is None or self.objectModel is None):
            print("ERROR: models aren't loaded")
            return None
        
        input_tensor = torch.tensor([class_num, x_center, y_center, width, height]).unsqueeze(0)

        with torch.no_grad():
            estimate_tensor = self.laneModel(input_tensor) if class_num == 0 else self.objectModel(input_tensor)
            estimate = estimate_tensor.item()
            return estimate * 100

# Function for loading the training data 
def load_data(dir = "../DistanceData/"):
    print("Started loading data")
    # Initialises the lists to return 
    data = []
    targets = []

    # Initialises variables for error checking
    empty_lines = 0
    incorrect_formatted_lines = 0
    type_errors = 0

    # Loads the text files using glob
    txt_files = glob.glob(f"{dir}img*.txt")

    # Iterates through the training data 
    for filename in txt_files:
        with open(filename, 'r') as file:
            # Iterates through each line of the file
            for line in file:
                # removes whitespace and skips over empty lines
                line = line.strip()
                if not line: 
                    empty_lines += 1
                    continue

                # Checks the format of the lane
                parts = line.split()
                if len(parts) != 6:
                    incorrect_formatted_lines += 1
                    continue

                # Checks each part is a float
                try:
                    cn, xc, yc, w, h, d = map(float, parts)
                except ValueError as e:
                    type_errors += 1
                    continue
                
                # appends the box to the lists
                data.append([cn, xc, yc, w, h])
                targets.append(d / 100)
    
    # prints feedback to the user
    print(f"Empty lines: {empty_lines}")
    print(f"Incorrectly Formatted Lines: {incorrect_formatted_lines}")
    print(f"Type Errors: {type_errors}")
    print("Finished Loading Data")
    
    # returns the list as numpy arrays 
    data = np.array(data, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)
    return data, targets

# The loss function 
def loss_function(predictions, targets):
    mse_loss = F.mse_loss(predictions, targets)
    return mse_loss