import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class AttentionBasedFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(AttentionBasedFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Assuming the observation space is a Box with shape (num_nodes, num_features)
        num_nodes = observation_space.shape[0]
        num_features = observation_space.shape[1]
        
        # Define the attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=num_features, num_heads=4, batch_first=True)
        
        # Final linear layer to produce the feature vector
        self.linear = nn.Linear(num_features, features_dim)

    def forward(self, observations):
        # observations shape: (batch_size, num_nodes, num_features)
        
        # No need to permute since batch_first=True
        x = observations
        
        # Apply attention mechanism
        attn_output, _ = self.attention(x, x, x)
        
        # Take the mean across nodes to get a single feature vector per batch
        attn_output = attn_output.mean(dim=1)  # shape: (batch_size, num_features)

        # Layer normalization
        attn_output = F.layer_norm(attn_output, attn_output.size()[1:])
        
        # Pass through the final linear layer
        features = F.relu(self.linear(attn_output))  # shape: (batch_size, features_dim)
        
        return features
    

class LinearFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(LinearFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Assuming the observation space is a Box with shape (num_nodes, num_features)
        num_nodes = observation_space.shape[0]
        num_features = observation_space.shape[1]
        
        # Final linear layer to produce the feature vector
        self.linear1 = nn.Linear(num_nodes * num_features, 1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear3 = nn.Linear(1024, features_dim)

    def forward(self, observations):
        # observations shape: (batch_size, num_nodes, num_features)
        
        # Flatten the observations
        x = observations.view(observations.size(0), -1)  # shape: (batch_size, num_nodes * num_features)
        
        # Pass through the linear layers
        features = F.relu(self.linear1(x))  # shape: (batch_size, 1024)
        # Layer normalization 
        features = F.layer_norm(features, features.size()[1:])
        features = F.relu(self.linear2(features))  # shape: (batch_size, 1024)
        features = F.layer_norm(features, features.size()[1:])
        features = F.relu(self.linear3(features))  # shape: (batch_size, features_dim)
        features = F.layer_norm(features, features.size()[1:])
        
        return features
    

class ConvolutionalFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(ConvolutionalFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Assuming the observation space is a Box with shape (num_nodes, num_features)
        num_nodes = observation_space.shape[0]
        num_features = observation_space.shape[1]
        
        # Define 1D convolutional layers to obtain output of shape (batch_size, 32, num_nodes)
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=num_features, padding=num_features//2 -1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=num_features, padding=num_features//2)
        
        # Compute the output size after the convolutional layers to determine the input size for the final linear layer
        dummy_input = torch.zeros(1, num_features, num_nodes)  # shape: (1, num_features, num_nodes)
        dummy_output = self.conv2(F.relu(self.conv1(dummy_input)))  # shape: (1, 32, num_nodes)
        conv_output_size = dummy_output.view(1, -1).size(1)  # shape: (1, 32 * num_nodes)

        # Final linear layer to produce the feature vector
        self.linear = nn.Linear(conv_output_size, features_dim)

    def forward(self, observations):
        # observations shape: (batch_size, num_nodes, num_features)
        
        # Permute to (batch_size, num_features, num_nodes) for Conv1d
        x = observations.permute(0, 2, 1)  # shape: (batch_size, num_features, num_nodes)
        
        # Apply convolutional layers
        x = F.relu(self.conv1(x))  # shape: (batch_size, 32, num_nodes)
        x = F.relu(self.conv2(x))  # shape: (batch_size, 32, num_nodes)

        # Flatten the output
        x = x.view(x.size(0), -1)  # shape: (batch_size, 32 * num_nodes)
        
        # Layer normalization
        x = F.layer_norm(x, x.size()[1:])

        # Pass through the final linear layer
        features = F.relu(self.linear(x))  # shape: (batch_size, features_dim)
        
        return features
    

class EntropyFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(EntropyFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Assuming the observation space is a Box with shape (num_nodes, num_features)
        num_nodes = observation_space.shape[0]
        num_features = observation_space.shape[1]
        
        # Define 2 Linear Layers
        self.linear1 = nn.Linear(num_nodes, 512)
        self.linear2 = nn.Linear(512, features_dim)

    def forward(self, observations):
        # observations shape: (batch_size, num_nodes, num_features)

        # Obtain pdf from cdf
        pdf = observations[:, :, 1:] - observations[:, :, :-1]  # shape: (batch_size, num_nodes, num_features-1)
        # Calculate entropy
        entropy = -torch.sum(pdf * torch.log(pdf + 1e-8), dim=2)  # shape: (batch_size, num_nodes)

        # Clamp the entropy values to avoid extreme values
        entropy = torch.clamp(entropy, min=0.0, max=10.0)
        
        # Pass through the linear layers
        features = F.relu(self.linear1(entropy))  # shape: (batch_size, 512)
        features = F.relu(self.linear2(features))  # shape: (batch_size, features_dim)
        
        return features


class LinearLoadFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(LinearLoadFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Assuming the observation space is a Box with shape (num_nodes, num_features)
        num_nodes = observation_space.shape[0]
        num_features = observation_space.shape[1]

        self.num_nodes = num_nodes
        self.num_slot_per_frame = 200
        
        # Define 2 Linear Layers
        self.linear1 = nn.Linear(num_features, 256)
        self.linear2 = nn.Linear(256, features_dim)

    def forward(self, observations):
        # Obtain expected load from the error distribution
        expected_silent_nodes = observations.sum(dim=1) # num_bins
        x = (self.num_nodes - expected_silent_nodes) / self.num_slot_per_frame  # shape: (batch_size, num_features)

        # Pass through the linear layers
        features = F.leaky_relu(self.linear1(x))  # shape: (batch_size, 256)
        features = F.leaky_relu(self.linear2(features))  # shape: (batch_size, features_dim)

        return features