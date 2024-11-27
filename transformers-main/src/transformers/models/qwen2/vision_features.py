import torch
from transformers import CLIPProcessor, CLIPVisionModel
from PIL import Image
import numpy as np
import sys
import os
sys.path.append(os.path.abspath("src\\transformers\\models\\qwen2"))
from vlmfeedforward import *

class CLIPFeatureExtractor:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialize CLIP feature extractor.
        
        Args:
            model_name (str): Name of the CLIP model to use
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPVisionModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
    def extract_features(self, image_path, output_type="pooled"):
        """
        Extract features from an image using CLIP.
        
        Args:
            image_path (str): Path to the input image
            output_type (str): Type of features to extract:
                - 'pooled': Global image features (CLS token)
                - 'patches': Features for each patch
                - 'hidden': Last hidden states
                
        Returns:
            np.ndarray: Extracted features
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(pixel_values, output_hidden_states=True)
            
            
            if output_type == "pooled":
                # Get pooled features (CLS token) [batch_size, hidden_size]
                features = outputs.pooler_output
            elif output_type == "patches":
                # Get patch features without CLS token [batch_size, num_patches, hidden_size]
                features = outputs.last_hidden_state[:, 1:, :]
            elif output_type == "hidden":
                # Get all hidden states [num_layers, batch_size, sequence_length, hidden_size]
                features = torch.stack(outputs.hidden_states)
            else:
                raise ValueError(f"Unknown output_type: {output_type}")
            
            # Convert to numpy and normalize if pooled or patch features
            features = features.cpu().numpy()
            if output_type in ["pooled", "patches"]:
                features = features / np.linalg.norm(features, axis=-1, keepdims=True)
            
            return features.squeeze()

    def extract_grid_features(self, image_path, grid_size=(7, 7)):
        """
        Extract features from a grid of image patches.
        
        Args:
            image_path (str): Path to the input image
            grid_size (tuple): Size of the grid (height, width)
            
        Returns:
            np.ndarray: Grid of features with shape [height, width, feature_dim]
        """
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        patch_width = width // grid_size[1]
        patch_height = height // grid_size[0]
        
        grid_features = np.zeros((grid_size[0], grid_size[1], self.model.config.hidden_size))
        
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Extract patch
                left = j * patch_width
                top = i * patch_height
                right = left + patch_width
                bottom = top + patch_height
                patch = image.crop((left, top, right, bottom))
                
                # Process patch
                inputs = self.processor(images=patch, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    outputs = self.model(pixel_values)
                    features = outputs.pooler_output.cpu().numpy()
                    features = features / np.linalg.norm(features)
                    grid_features[i, j] = features.squeeze()
        
        return grid_features

# Example usage
if __name__ == "__main__":
    # Initialize feature extractor
    extractor = CLIPFeatureExtractor()
    
    # Example image path
    image_path = "src/transformers/models/qwen2/example.jpg"
    
    # Extract different types of features
    #pooled_features = extractor.extract_features(image_path, output_type="pooled")
    #print("Global features shape:", pooled_features.shape)
    
   # patch_features = extractor.extract_features(image_path, output_type="patches")
    #print("Patch features shape:", patch_features.shape)
    
    hidden_states = extractor.extract_features(image_path, output_type="hidden")
    print("Hidden states shape:", hidden_states.shape)
    print("Type of hidden state", type(hidden_states))
    # Extract grid features
    #grid_features = extractor.extract_grid_features(image_path, grid_size=(7, 7))
    #print("Grid features shape:", grid_features.shape)

    p=QWENVLM(768, 1024)(torch.tensor(hidden_states))
