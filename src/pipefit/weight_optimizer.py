# """
# Module for optimizing weights for pipeline suitability analysis using various methods.
# Includes gradient-based optimization and neural network approaches using PyTorch.
# """
# import os
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, TensorDataset
# import matplotlib.pyplot as plt
# from typing import Dict, List, Tuple, Optional, Union, Callable
# import json

# from .data_loader import DataLoader as PipelineDataLoader
# from .cost_map import CostMapGenerator
# from .path_finder import LeastCostPathFinder
# from .path_evaluator import PathEvaluator


# class SimpleGradientOptimizer:
#     """
#     Optimizes weights for factors using gradient-based methods.
#     This is a simpler approach that does not require neural networks.
#     """
    
#     def __init__(self, datasets: List[str], factors: List[str], 
#                  initial_weights: Optional[Dict[str, float]] = None,
#                  learning_rate: float = 0.01,
#                  regularization: float = 0.01):
#         """
#         Initialize the optimizer.
        
#         Args:
#             datasets: List of paths to dataset folders
#             factors: List of factor names to optimize
#             initial_weights: Optional initial weights (if None, equal weights are used)
#             learning_rate: Learning rate for gradient descent
#             regularization: L2 regularization parameter
#         """
#         self.datasets = datasets
#         self.factors = factors
#         self.learning_rate = learning_rate
#         self.regularization = regularization
        
#         # Initialize weights
#         if initial_weights is None:
#             # Start with equal weights
#             weight_value = 1.0 / len(factors)
#             self.weights = {factor: weight_value for factor in factors}
#         else:
#             self.weights = initial_weights.copy()
            
#         # Normalize weights to sum to 1
#         self._normalize_weights()
        
#         # Load datasets
#         self.dataset_loaders = []
#         self.cost_generators = []
        
#         for dataset_path in datasets:
#             loader = PipelineDataLoader(dataset_path)
#             loader.load_data()
#             self.dataset_loaders.append(loader)
            
#             cost_generator = CostMapGenerator(loader.factors)
#             self.cost_generators.append(cost_generator)
            
#     def _normalize_weights(self) -> None:
#         """Normalize weights to sum to 1."""
#         total = sum(self.weights.values())
#         for factor in self.weights:
#             self.weights[factor] /= total
            
#     def evaluate_weights(self, weights: Dict[str, float]) -> Tuple[float, List[Dict]]:
#         """
#         Evaluate a set of weights across all datasets.
        
#         Args:
#             weights: Dictionary of factor weights
            
#         Returns:
#             Tuple of (average_score, list_of_metrics_per_dataset)
#         """
#         all_metrics = []
#         total_score = 0.0
        
#         for i, (loader, cost_generator) in enumerate(zip(self.dataset_loaders, self.cost_generators)):
#             # Generate cost map
#             cost_map = cost_generator.generate_cost_map(weights)
            
#             # Get metadata
#             metadata = loader.get_common_metadata()
            
#             # Use reference path start and end points for validation
#             ref_path = loader.reference_path
#             ref_geom = ref_path.geometry.iloc[0]
            
#             # Extract start and end coordinates
#             start_coords = (ref_geom.coords[0][0], ref_geom.coords[0][1])
#             end_coords = (ref_geom.coords[-1][0], ref_geom.coords[-1][1])
            
#             # Convert to indices
#             start_indices = loader.coordinates_to_indices(start_coords[0], start_coords[1])
#             end_indices = loader.coordinates_to_indices(end_coords[0], end_coords[1])
            
#             # Find path
#             path_finder = LeastCostPathFinder(cost_map)
#             path_indices, path_cost = path_finder.find_path(start_indices, end_indices)
            
#             # Evaluate path
#             evaluator = PathEvaluator(
#                 predicted_path=path_indices,
#                 reference_path=loader.reference_path,
#                 transform=metadata['transform'],
#                 crs=metadata['crs']
#             )
            
#             metrics = evaluator.evaluate_all_metrics()
#             all_metrics.append(metrics)
            
#             # Use combined score for optimization
#             total_score += metrics['combined_score']
            
#         # Calculate average score
#         average_score = total_score / len(self.datasets)
        
#         # Add regularization term (L2 regularization)
#         if self.regularization > 0:
#             weight_values = np.array(list(weights.values()))
#             reg_term = self.regularization * np.sum(weight_values**2)
#             average_score += reg_term
            
#         return average_score, all_metrics
    
#     def optimize(self, num_iterations: int = 100, 
#                  early_stopping_patience: int = 10,
#                  verbose: bool = True) -> Tuple[Dict[str, float], List[float]]:
#         """
#         Optimize weights using gradient descent.
        
#         Args:
#             num_iterations: Number of optimization iterations
#             early_stopping_patience: Stop if no improvement after this many iterations
#             verbose: Whether to print progress information
            
#         Returns:
#             Tuple of (best_weights, scores_history)
#         """
#         scores_history = []
#         best_score = float('inf')
#         best_weights = self.weights.copy()
#         patience_counter = 0
        
#         for iteration in range(num_iterations):
#             current_score, metrics = self.evaluate_weights(self.weights)
#             scores_history.append(current_score)
            
#             if verbose and (iteration % 5 == 0 or iteration == num_iterations - 1):
#                 print(f"Iteration {iteration}, Score: {current_score:.6f}")
#                 print(f"Current weights: {self.weights}")
                
#             # Check for improvement
#             if current_score < best_score:
#                 best_score = current_score
#                 best_weights = self.weights.copy()
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
                
#             # Early stopping
#             if patience_counter >= early_stopping_patience:
#                 if verbose:
#                     print(f"Early stopping at iteration {iteration}")
#                 break
                
#             # Calculate gradients for each weight
#             gradients = {}
#             epsilon = 1e-4  # Small value for numerical gradient
            
#             for factor in self.factors:
#                 # Increase weight slightly
#                 temp_weights = self.weights.copy()
#                 temp_weights[factor] += epsilon
#                 self._normalize_dict(temp_weights)
                
#                 forward_score, _ = self.evaluate_weights(temp_weights)
                
#                 # Decrease weight slightly
#                 temp_weights = self.weights.copy()
#                 temp_weights[factor] -= epsilon
#                 self._normalize_dict(temp_weights)
                
#                 backward_score, _ = self.evaluate_weights(temp_weights)
                
#                 # Calculate numerical gradient
#                 gradient = (forward_score - backward_score) / (2 * epsilon)
#                 gradients[factor] = gradient
                
#             # Update weights
#             for factor in self.factors:
#                 self.weights[factor] -= self.learning_rate * gradients[factor]
                
#             # Ensure non-negative weights
#             for factor in self.factors:
#                 self.weights[factor] = max(0.01, self.weights[factor])
                
#             # Normalize weights to sum to 1
#             self._normalize_weights()
            
#         # Restore best weights
#         self.weights = best_weights
        
#         if verbose:
#             print(f"\nOptimization complete. Best score: {best_score:.6f}")
#             print(f"Best weights: {best_weights}")
            
#         return best_weights, scores_history
    
#     def _normalize_dict(self, d: Dict[str, float]) -> None:
#         """Helper to normalize a dictionary to sum to 1."""
#         total = sum(d.values())
#         for key in d:
#             d[key] /= total
            
#     def plot_optimization_history(self, scores_history: List[float]) -> None:
#         """
#         Plot the optimization history.
        
#         Args:
#             scores_history: List of scores from optimization
#         """
#         plt.figure(figsize=(10, 6))
#         plt.plot(scores_history)
#         plt.title('Optimization History')
#         plt.xlabel('Iteration')
#         plt.ylabel('Cost (Lower is Better)')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()
        
#     def save_weights(self, output_path: str) -> None:
#         """
#         Save optimized weights to a JSON file.
        
#         Args:
#             output_path: Path to save the weights
#         """
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
#         with open(output_path, 'w') as f:
#             json.dump(self.weights, f, indent=4)
            
#         print(f"Weights saved to {output_path}")
        
#     def load_weights(self, input_path: str) -> None:
#         """
#         Load weights from a JSON file.
        
#         Args:
#             input_path: Path to load the weights from
#         """
#         with open(input_path, 'r') as f:
#             self.weights = json.load(f)
            
#         print(f"Weights loaded from {input_path}")


# class WeightPredictorModel(nn.Module):
#     """PyTorch model for predicting weights from dataset features."""
    
#     def __init__(self, input_size: int, output_size: int):
#         """
#         Initialize the model.
        
#         Args:
#             input_size: Size of the input features
#             output_size: Size of the output (number of weights to predict)
#         """
#         super(WeightPredictorModel, self).__init__()
        
#         # Feature extraction layers
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(input_size, 64),
#             nn.ReLU(),
#             nn.BatchNorm1d(64),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.BatchNorm1d(32)
#         )
        
#         # Weight prediction layer (outputs sum to 1 via softmax)
#         self.weight_predictor = nn.Linear(32, output_size)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         """Forward pass through the network."""
#         x = self.feature_extractor(x)
#         x = self.weight_predictor(x)
#         x = self.softmax(x)
#         return x


# class NeuralWeightPredictor:
#     """
#     Neural network-based model using PyTorch that predicts optimal weights for pipeline routing.
#     This model can learn from multiple datasets to predict weights based on 
#     dataset characteristics.
#     """
    
#     def __init__(self, input_shape: Tuple[int, ...]):
#         """
#         Initialize the neural weight predictor.
        
#         Args:
#             input_shape: Shape of the input features (e.g., dataset characteristics)
#         """
#         self.input_shape = input_shape
#         self.factor_names = []  # Will be set later
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         self.model = None  # Will be initialized in _build_model
        
#     def _build_model(self, output_size: int) -> None:
#         """
#         Build the neural network model.
        
#         Args:
#             output_size: Number of factors to predict weights for
#         """
#         self.model = WeightPredictorModel(self.input_shape[0], output_size)
#         self.model.to(self.device)
        
#     def train(self, X_train, y_train, epochs=50, batch_size=16, validation_split=0.2, 
#               learning_rate=0.001):
#         """
#         Train the neural network model.
        
#         Args:
#             X_train: Training features (dataset characteristics)
#             y_train: Training labels (optimal weights)
#             epochs: Number of training epochs
#             batch_size: Batch size
#             validation_split: Fraction of data to use for validation
#             learning_rate: Learning rate for optimizer
            
#         Returns:
#             Training history (dict with 'loss' and 'val_loss' keys)
#         """
#         # Convert to PyTorch tensors
#         X_tensor = torch.tensor(X_train, dtype=torch.float32)
#         y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
#         # Set factor names if not already set
#         if not self.factor_names:
#             self.factor_names = [f"factor_{i}" for i in range(y_train.shape[1])]
            
#         # Build model if not already built
#         if self.model is None:
#             self._build_model(y_train.shape[1])
            
#         # Create dataset
#         dataset = TensorDataset(X_tensor, y_tensor)
        
#         # Split into training and validation
#         val_size = int(validation_split * len(dataset))
#         train_size = len(dataset) - val_size
#         train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
#         # Create data loaders
#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
#         # Define loss function and optimizer
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
#         # Training history
#         history = {'loss': [], 'val_loss': []}
        
#         # Training loop
#         for epoch in range(epochs):
#             # Training phase
#             self.model.train()
#             train_loss = 0.0
#             for inputs, targets in train_loader:
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
                
#                 # Zero the gradients
#                 optimizer.zero_grad()
                
#                 # Forward pass
#                 outputs = self.model(inputs)
                
#                 # Calculate loss
#                 loss = criterion(outputs, targets)
                
#                 # Backward pass and optimize
#                 loss.backward()
#                 optimizer.step()
                
#                 train_loss += loss.item() * inputs.size(0)
                
#             train_loss /= len(train_loader.dataset)
#             history['loss'].append(train_loss)
            
#             # Validation phase
#             self.model.eval()
#             val_loss = 0.0
#             with torch.no_grad():
#                 for inputs, targets in val_loader:
#                     inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
#                     # Forward pass
#                     outputs = self.model(inputs)
                    
#                     # Calculate loss
#                     loss = criterion(outputs, targets)
                    
#                     val_loss += loss.item() * inputs.size(0)
                    
#             val_loss /= len(val_loader.dataset)
#             history['val_loss'].append(val_loss)
            
#             # Print progress
#             print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")
            
#         return history
    
#     def predict_weights(self, dataset_features):
#         """
#         Predict optimal weights for a dataset based on its features.
        
#         Args:
#             dataset_features: Features describing the dataset
            
#         Returns:
#             Dictionary of predicted weights
#         """
#         # Ensure model is built
#         if self.model is None:
#             raise ValueError("Model must be trained before making predictions.")
            
#         # Ensure proper input shape
#         if len(dataset_features.shape) == 1:
#             dataset_features = np.expand_dims(dataset_features, axis=0)
            
#         # Convert to PyTorch tensor
#         features_tensor = torch.tensor(dataset_features, dtype=torch.float32).to(self.device)
        
#         # Set model to evaluation mode
#         self.model.eval()
        
#         # Make prediction
#         with torch.no_grad():
#             predicted_weights = self.model(features_tensor)
            
#         # Convert to numpy array
#         predicted_weights = predicted_weights.cpu().numpy()[0]
        
#         # Convert to dictionary
#         weights_dict = {}
#         for i, factor in enumerate(self.factor_names):
#             weights_dict[factor] = float(predicted_weights[i])
            
#         return weights_dict
    
#     def save_model(self, path):
#         """
#         Save the model to a file.
        
#         Args:
#             path: Path to save the model
#         """
#         if self.model is None:
#             raise ValueError("Model must be trained before saving.")
            
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(path), exist_ok=True)
        
#         # Save model state dict
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'factor_names': self.factor_names,
#             'input_shape': self.input_shape
#         }, path)
        
#         print(f"Model saved to {path}")
        
#     def load_model(self, path):
#         """
#         Load the model from a file.
        
#         Args:
#             path: Path to load the model from
#         """
#         # Load checkpoint
#         checkpoint = torch.load(path, map_location=self.device)
        
#         # Get factor names and input shape
#         self.factor_names = checkpoint['factor_names']
#         self.input_shape = checkpoint['input_shape']
        
#         # Build model
#         self._build_model(len(self.factor_names))
        
#         # Load state dict
#         self.model.load_state_dict(checkpoint['model_state_dict'])
        
#         # Set model to evaluation mode
#         self.model.eval()
        
#         print(f"Model loaded from {path}")


# def extract_dataset_features(data_loader: PipelineDataLoader) -> np.ndarray:
#     """
#     Extract features from a dataset that can be used to predict optimal weights.
    
#     Args:
#         data_loader: DataLoader instance with loaded data
        
#     Returns:
#         Array of features
#     """
#     features = []
    
#     # Extract features from raster factors
#     for factor_name, factor_data in data_loader.factors.items():
#         # Calculate basic statistics for each factor
#         features.extend([
#             np.mean(factor_data),
#             np.std(factor_data),
#             np.min(factor_data),
#             np.max(factor_data)
#         ])
        
#     # Extract features from reference path
#     if data_loader.reference_path is not None:
#         ref_geom = data_loader.reference_path.geometry.iloc[0]
#         features.extend([
#             ref_geom.length,  # Path length
#             ref_geom.bounds[2] - ref_geom.bounds[0],  # X extent
#             ref_geom.bounds[3] - ref_geom.bounds[1]   # Y extent
#         ])
    
#     return np.array(features)