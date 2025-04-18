"""
Module for finding least cost paths on cost maps.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.graph import route_through_array
from typing import Tuple, List, Dict, Optional, Union, Literal
import geopandas as gpd
from shapely.geometry import LineString, Point
import pyastar2d


class LeastCostPathFinder:
    """
    Finds the least cost path between two points on a cost map.
    Supports multiple path-finding algorithms.
    """
    
    def __init__(self, cost_map: np.ndarray):
        """
        Initialize with a cost map.
        
        Args:
            cost_map: 2D numpy array representing the cost map
        """
        self.cost_map = cost_map
        
    def find_path(self, start_point: Tuple[int, int] = None, 
                  end_point: Tuple[int, int] = None, 
                  method: str = 'astar', 
                  fully_connected: bool = True) -> Tuple[np.ndarray, float]:
        """
        Find the least cost path between start and end points.
        
        Args:
            start_point: Tuple of (row, col) indices for the start point. 
                        If None, uses lower left corner.
            end_point: Tuple of (row, col) indices for the end point.
                      If None, uses upper right corner.
            method: Path-finding method to use ('astar' or 'dijkstra')
            fully_connected: Whether to use 8-connectivity (True) or 4-connectivity (False)
            
        Returns:
            Tuple of (path_indices, path_cost) where path_indices is an array of (row, col) indices
            and path_cost is the total cost of the path
        """
        # Set default start and end points if not provided
        if start_point is None:
            start_point = (self.cost_map.shape[0] - 1, 0)  # Lower left
        
        if end_point is None:
            end_point = (0, self.cost_map.shape[1] - 1)  # Upper right
            
        # Choose path-finding method
        if method.lower() == 'astar':
            return self._find_path_astar(self.cost_map, start_point, end_point, fully_connected)
        elif method.lower() == 'dijkstra':
            return self._find_path_dijkstra(self.cost_map, start_point, end_point, fully_connected)
        else:
            raise ValueError(f"Unknown path-finding method: {method}. Use 'astar' or 'dijkstra'.")
    
    def _find_path_astar(self, cost_map: np.ndarray, 
                         start_point: Tuple[int, int], 
                         end_point: Tuple[int, int],
                         fully_connected: bool) -> Tuple[np.ndarray, float]:
        """
        Find path using pyastar2d's A* algorithm.
        
        Args:
            cost_map: Cost map with positive values
            start_point: Start point indices (row, col)
            end_point: End point indices (row, col)
            fully_connected: Whether to use 8-connectivity (True) or 4-connectivity (False)
            
        Returns:
            Tuple of (path_indices, path_cost)
        """
        # Bounds checking for start and end points
        height, width = cost_map.shape
        if (start_point[0] < 0 or start_point[0] >= height or 
            start_point[1] < 0 or start_point[1] >= width):
            raise ValueError(f"Start point {start_point} is out of bounds for cost map of shape {cost_map.shape}")
            
        if (end_point[0] < 0 or end_point[0] >= height or 
            end_point[1] < 0 or end_point[1] >= width):
            raise ValueError(f"End point {end_point} is out of bounds for cost map of shape {cost_map.shape}")
        
        # Make a copy of the cost map to avoid modifying the original
        cost_map_astar = cost_map.copy()
        
        # pyastar2d requires the minimum cost to be at least 1.0
        min_val = np.min(cost_map_astar)
        if min_val < 1.0:
            # Scale the cost map so that min value is exactly 1.0
            cost_map_astar = cost_map_astar - min_val + 1.0
        
        # Convert to float32 for pyastar2d
        cost_map_astar = cost_map_astar.astype(np.float32)
        
        # Run pyastar2d
        path_indices = pyastar2d.astar_path(
            cost_map_astar,     # Must be float32 with min value of at least 1.0
            start_point,       # (row, col)
            end_point,         # (row, col)
            allow_diagonal=fully_connected
        )
        
        if path_indices is None:
            raise ValueError(f"No path found from {start_point} to {end_point}")
        
        # Calculate the path cost using the original cost map
        path_cost = 0.0
        for i in range(len(path_indices) - 1):
            r1, c1 = path_indices[i]
            r2, c2 = path_indices[i + 1]
            # For diagonal moves, use Euclidean distance
            if r1 != r2 and c1 != c2:
                path_cost += np.sqrt(2) * self.cost_map[r2, c2]
            else:
                path_cost += self.cost_map[r2, c2]
                
        return path_indices, path_cost
    
    def _find_path_dijkstra(self, cost_map: np.ndarray, 
                           start_point: Tuple[int, int], 
                           end_point: Tuple[int, int],
                           fully_connected: bool) -> Tuple[np.ndarray, float]:
        """
        Find path using Dijkstra's algorithm (via skimage's route_through_array).
        
        Args:
            cost_map: Cost map with positive values
            start_point: Start point indices (row, col)
            end_point: End point indices (row, col)
            fully_connected: Whether to use 8-connectivity (True) or 4-connectivity (False)
            
        Returns:
            Tuple of (path_indices, path_cost)
        """
        # Find the least cost path using route_through_array (Dijkstra's algorithm)
        indices, cost = route_through_array(
            cost_map, 
            start=start_point, 
            end=end_point,
            fully_connected=fully_connected
        )
        
        # Convert list of indices to numpy array
        path_indices = np.array(indices)
        
        return path_indices, cost
    
    def path_indices_to_coordinates(self, path_indices: np.ndarray, transform) -> np.ndarray:
        """
        Convert path indices to geographic coordinates using the transform.
        
        Args:
            path_indices: Array of (row, col) indices
            transform: Geotransform to convert indices to coordinates
            
        Returns:
            Array of (x, y) coordinates
        """
        if transform is None:
            raise ValueError("Transform is required to convert indices to coordinates")
            
        coords = []
        for index in path_indices:
            row, col = index
            x, y = transform * (col, row)
            coords.append((x, y))
            
        return np.array(coords)
    
    def path_to_linestring(self, path_indices: np.ndarray, transform) -> LineString:
        """
        Convert path indices to a Shapely LineString.
        
        Args:
            path_indices: Array of (row, col) indices
            transform: Geotransform to convert indices to coordinates
            
        Returns:
            Shapely LineString geometry
        """
        coords = self.path_indices_to_coordinates(path_indices, transform)
        return LineString(coords)
    
    def path_to_geodataframe(self, path_indices: np.ndarray, cost: float, transform, crs) -> gpd.GeoDataFrame:
        """
        Convert path indices to a GeoDataFrame.
        
        Args:
            path_indices: Array of (row, col) indices
            cost: Total cost of the path
            transform: Geotransform to convert indices to coordinates
            crs: Coordinate reference system for the output GeoDataFrame
            
        Returns:
            GeoDataFrame containing the path as a LineString
        """
        if crs is None:
            raise ValueError("CRS is required to create a GeoDataFrame")
            
        linestring = self.path_to_linestring(path_indices, transform)
        
        gdf = gpd.GeoDataFrame(
            {'cost': [cost], 'geometry': [linestring]},
            crs=crs
        )
        
        return gdf
    
    def visualize_path(self, path_indices: np.ndarray, start_point: Tuple[int, int] = None, 
                       end_point: Tuple[int, int] = None, figsize: tuple = (10, 8)) -> None:
        """
        Visualize the path on the cost map.
        
        Args:
            path_indices: Array of (row, col) indices
            start_point: Optional tuple of (row, col) indices for the start point
            end_point: Optional tuple of (row, col) indices for the end point
            figsize: Figure size (width, height) in inches
        """
        plt.figure(figsize=figsize)
        
        # Display the cost map
        plt.imshow(self.cost_map, cmap='viridis')
        plt.colorbar(label='Cost')
        
        # Plot the path
        path_rows, path_cols = path_indices[:, 0], path_indices[:, 1]
        plt.plot(path_cols, path_rows, 'r-', linewidth=2, label='Least Cost Path')
        
        # Plot start and end points if provided
        if start_point is not None:
            plt.plot(start_point[1], start_point[0], 'go', markersize=10, label='Start')
            
        if end_point is not None:
            plt.plot(end_point[1], end_point[0], 'bo', markersize=10, label='End')
            
        plt.title('Least Cost Path')
        plt.legend()
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.show()