"""
Module for generating cost maps by combining raster factors with weights.
"""
import numpy as np
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt


class CostMapGenerator:
    """
    Generates cost maps by combining raster factors with weights.
    """
    
    def __init__(self, factors: Dict[str, np.ndarray]):
        """
        Initialize with a dictionary of raster factors.
        
        Args:
            factors: Dictionary mapping factor names to numpy arrays
        """
        self.factors = factors
        
    def generate_cost_map(self, weights: Dict[str, float], 
                          default_weight: Optional[float] = None,
                          weight_mapping: Optional[Dict[str, str]] = None) -> np.ndarray:
        """
        Generate a cost map by combining factors with weights.
        
        Args:
            weights: Dictionary mapping factor names to weights
            default_weight: Optional default weight for factors without specified weights
            weight_mapping: Optional mapping from weight keys to factor names
                           e.g., {'elevation': 'elev', 'protected_areas': 'protected'}
            
        Returns:
            2D numpy array representing the cost map
        """
        # Apply weight mapping if provided
        if weight_mapping:
            mapped_weights = {}
            for weight_name, weight_value in weights.items():
                if weight_name in weight_mapping:
                    mapped_name = weight_mapping[weight_name]
                    mapped_weights[mapped_name] = weight_value
                else:
                    mapped_weights[weight_name] = weight_value
            weights = mapped_weights
        
        # Identify factors without weights
        missing_weights = set(self.factors.keys()) - set(weights.keys())
        
        # Apply default weight if provided
        if default_weight is not None and missing_weights:
            for factor_name in missing_weights:
                weights[factor_name] = default_weight
                
        # Check for weights without corresponding factors
        unknown_weights = set(weights.keys()) - set(self.factors.keys())
        if unknown_weights:
            print(f"Warning: Weights provided for unknown factors: {unknown_weights}. They will be ignored.")
            
        # Initialize cost map with zeros
        first_factor = next(iter(self.factors.values()))
        shape = first_factor.shape
        cost_map = np.zeros(shape, dtype=np.float32)
        
        # Combine factors with weights
        total_weight = sum(weight for factor, weight in weights.items() if factor in self.factors)
        
        if total_weight == 0:
            raise ValueError("Total weight is zero. Cannot create cost map.")
        
        for factor_name, weight in weights.items():
            if factor_name in self.factors:
                # Normalize weight
                normalized_weight = weight / total_weight
                
                # Add weighted factor to cost map
                cost_map += self.factors[factor_name] * normalized_weight
        
        return cost_map
    
    def visualize_cost_map(self, cost_map: np.ndarray, title: str = "Cost Map", 
                           cmap: str = "viridis", figsize: tuple = (10, 8)) -> None:
        """
        Visualize the cost map.
        
        Args:
            cost_map: 2D numpy array representing the cost map
            title: Title for the visualization
            cmap: Matplotlib colormap to use
            figsize: Figure size (width, height) in inches
        """
        plt.figure(figsize=figsize)
        plt.imshow(cost_map, cmap=cmap)
        plt.colorbar(label="Cost (0=Suitable, 1=Not Suitable)")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()
        
    def visualize_factors(self, factors_to_show: Optional[List[str]] = None, 
                          figsize: tuple = (15, 10)) -> None:
        """
        Visualize selected or all factors.
        
        Args:
            factors_to_show: Optional list of factor names to show (if None, shows all)
            figsize: Figure size (width, height) in inches
        """
        # Determine which factors to show
        if factors_to_show is None:
            factors_to_show = list(self.factors.keys())
        else:
            # Check that all specified factors exist
            unknown_factors = set(factors_to_show) - set(self.factors.keys())
            if unknown_factors:
                print(f"Warning: Unknown factors: {unknown_factors}. They will be skipped.")
            
            # Filter to only include known factors
            factors_to_show = [f for f in factors_to_show if f in self.factors]
            
        if not factors_to_show:
            print("No factors to show.")
            return
            
        num_factors = len(factors_to_show)
        cols = min(num_factors, 3)
        rows = (num_factors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if num_factors == 1:
            axes = np.array([axes])
        
        axes = axes.flatten()
        
        # Create a single colorbar for all plots
        vmin = min(np.min(self.factors[factor_name]) for factor_name in factors_to_show)
        vmax = max(np.max(self.factors[factor_name]) for factor_name in factors_to_show)
        
        for i, factor_name in enumerate(factors_to_show):
            ax = axes[i]
            im = ax.imshow(self.factors[factor_name], cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(factor_name)
            
        # Add a single colorbar for all subplots
        fig.colorbar(im, ax=axes.tolist(), shrink=0.6)
            
        # Hide unused subplots
        for i in range(num_factors, len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()