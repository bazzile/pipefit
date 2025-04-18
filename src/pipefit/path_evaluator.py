"""
Module for evaluating similarity between predicted and reference paths.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, Union
from scipy.spatial import distance


class PathEvaluator:
    """
    Evaluates the similarity between a predicted path and a reference path.
    Provides multiple metrics for comparison.
    """
    
    def __init__(self, predicted_path: Union[np.ndarray, LineString, gpd.GeoDataFrame], 
                 reference_path: Union[LineString, gpd.GeoDataFrame, str], 
                 transform=None, crs=None):
        """
        Initialize with predicted and reference paths.
        
        Args:
            predicted_path: Predicted path as array of indices, LineString, or GeoDataFrame
            reference_path: Reference path as LineString, GeoDataFrame, or path to shapefile
            transform: Optional geotransform (required if predicted_path is array of indices)
            crs: Optional coordinate reference system
        """
        self.transform = transform
        self.crs = crs
        
        # Convert predicted path to LineString if needed
        if isinstance(predicted_path, np.ndarray):
            if transform is None:
                raise ValueError("Transform is required if predicted_path is an array of indices")
            self.predicted_linestring = self._indices_to_linestring(predicted_path, transform)
        elif isinstance(predicted_path, LineString):
            self.predicted_linestring = predicted_path
        elif isinstance(predicted_path, gpd.GeoDataFrame):
            if len(predicted_path) != 1:
                raise ValueError("Predicted path GeoDataFrame must contain exactly one geometry")
            geometry = predicted_path.geometry.iloc[0]
            if isinstance(geometry, LineString):
                self.predicted_linestring = geometry
            else:
                raise ValueError("Predicted path geometry must be a LineString")
        else:
            raise ValueError("Predicted path must be array of indices, LineString, or GeoDataFrame")
            
        # Convert reference path to LineString if needed
        if isinstance(reference_path, str):
            # Load reference path from file
            from geopandas import read_file
            reference_gdf = read_file(reference_path)
            if 'geometry' not in reference_gdf.columns:
                raise ValueError("Reference path file must contain a geometry column")
            
            # Process geometry as in the existing code
            if len(reference_gdf) == 1:
                geometry = reference_gdf.geometry.iloc[0]
                if isinstance(geometry, LineString):
                    self.reference_linestring = geometry
                elif isinstance(geometry, MultiLineString):
                    # Convert MultiLineString to LineString by merging
                    coords = []
                    for line in geometry.geoms:
                        coords.extend(list(line.coords))
                    self.reference_linestring = LineString(coords)
                else:
                    raise ValueError("Reference path geometry must be a LineString or MultiLineString")
            else:
                # If multiple features, merge them
                lines = []
                for geom in reference_gdf.geometry:
                    if isinstance(geom, LineString):
                        lines.append(geom)
                    elif isinstance(geom, MultiLineString):
                        for line in geom.geoms:
                            lines.append(line)
                
                # Merge all lines into one LineString
                coords = []
                for line in lines:
                    coords.extend(list(line.coords))
                self.reference_linestring = LineString(coords)
        elif isinstance(reference_path, LineString):
            self.reference_linestring = reference_path
        elif isinstance(reference_path, gpd.GeoDataFrame):
            # Process GeoDataFrame as in the existing code
            if 'geometry' not in reference_path.columns:
                raise ValueError("Reference path GeoDataFrame must contain a geometry column")
            
            # Handle single or multiple geometries
            if len(reference_path) == 1:
                geometry = reference_path.geometry.iloc[0]
                if isinstance(geometry, LineString):
                    self.reference_linestring = geometry
                elif isinstance(geometry, MultiLineString):
                    # Convert MultiLineString to LineString by merging
                    coords = []
                    for line in geometry.geoms:
                        coords.extend(list(line.coords))
                    self.reference_linestring = LineString(coords)
                else:
                    raise ValueError("Reference path geometry must be a LineString or MultiLineString")
            else:
                # If multiple features, merge them
                lines = []
                for geom in reference_path.geometry:
                    if isinstance(geom, LineString):
                        lines.append(geom)
                    elif isinstance(geom, MultiLineString):
                        for line in geom.geoms:
                            lines.append(line)
                
                # Merge all lines into one LineString
                coords = []
                for line in lines:
                    coords.extend(list(line.coords))
                self.reference_linestring = LineString(coords)
        else:
            raise ValueError("Reference path must be LineString, GeoDataFrame, or path to shapefile")
        
    def _indices_to_linestring(self, indices: np.ndarray, transform) -> LineString:
        """
        Convert array of (row, col) indices to a LineString using the transform.
        
        Args:
            indices: Array of (row, col) indices
            transform: Geotransform to convert indices to coordinates
            
        Returns:
            Shapely LineString
        """
        coords = []
        for index in indices:
            row, col = index
            x, y = transform * (col, row)
            coords.append((x, y))
            
        return LineString(coords)
    
    def calculate_hausdorff_distance(self) -> float:
        """
        Calculate the Hausdorff distance between predicted and reference paths.
        This measures the greatest minimum distance between the two paths.
        
        Returns:
            Hausdorff distance value
        """
        # Extract coordinates from LineStrings
        pred_coords = np.array(self.predicted_linestring.coords)
        ref_coords = np.array(self.reference_linestring.coords)
        
        # Calculate directed Hausdorff distances
        forward = distance.directed_hausdorff(pred_coords, ref_coords)[0]
        backward = distance.directed_hausdorff(ref_coords, pred_coords)[0]
        
        # Hausdorff distance is the maximum of the two directed distances
        return max(forward, backward)
    
    def calculate_average_minimum_distance(self) -> float:
        """
        Calculate the average minimum distance between points on the two paths.
        This measures the average distance from points on one path to the closest point on the other path.
        
        Returns:
            Average minimum distance
        """
        # Extract coordinates from LineStrings
        pred_coords = np.array(self.predicted_linestring.coords)
        ref_coords = np.array(self.reference_linestring.coords)
        
        # Calculate minimum distances from predicted to reference
        min_distances_pred = []
        for point in pred_coords:
            distances = np.sqrt(np.sum((ref_coords - point)**2, axis=1))
            min_distances_pred.append(np.min(distances))
            
        # Calculate minimum distances from reference to predicted
        min_distances_ref = []
        for point in ref_coords:
            distances = np.sqrt(np.sum((pred_coords - point)**2, axis=1))
            min_distances_ref.append(np.min(distances))
            
        # Average of all minimum distances
        all_min_distances = min_distances_pred + min_distances_ref
        return np.mean(all_min_distances)
    
    def calculate_buffer_overlap(self, buffer_distance: float) -> float:
        """
        Calculate the overlap between buffered paths as a Jaccard similarity.
        
        Args:
            buffer_distance: Distance to buffer each path
            
        Returns:
            Jaccard similarity index (intersection area / union area)
        """
        pred_buffer = self.predicted_linestring.buffer(buffer_distance)
        ref_buffer = self.reference_linestring.buffer(buffer_distance)
        
        intersection_area = pred_buffer.intersection(ref_buffer).area
        union_area = pred_buffer.union(ref_buffer).area
        
        # Return Jaccard similarity (0 = no overlap, 1 = perfect overlap)
        return intersection_area / union_area
    
    def calculate_path_length_ratio(self) -> float:
        """
        Calculate the ratio of predicted path length to reference path length.
        
        Returns:
            Path length ratio (predicted/reference)
        """
        pred_length = self.predicted_linestring.length
        ref_length = self.reference_linestring.length
        
        return pred_length / ref_length
    
    def evaluate_all_metrics(self, buffer_distance: float = 100) -> Dict[str, float]:
        """
        Calculate all path similarity metrics.
        
        Args:
            buffer_distance: Distance to buffer each path for overlap calculation
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'hausdorff_distance': self.calculate_hausdorff_distance(),
            'average_min_distance': self.calculate_average_minimum_distance(),
            'buffer_overlap': self.calculate_buffer_overlap(buffer_distance),
            'path_length_ratio': self.calculate_path_length_ratio()
        }
        
        # Create a combined score (lower is better)
        # Normalize metrics to 0-1 range and invert buffer_overlap (higher is better)
        normalized_hausdorff = metrics['hausdorff_distance'] / (metrics['hausdorff_distance'] + 1000)
        normalized_avg_dist = metrics['average_min_distance'] / (metrics['average_min_distance'] + 1000)
        normalized_overlap = 1 - metrics['buffer_overlap']
        
        # Length ratio penalizes both too short and too long paths
        length_penalty = abs(metrics['path_length_ratio'] - 1)
        normalized_length = length_penalty / (length_penalty + 1)
        
        # Combined score: weighted average of normalized metrics
        metrics['combined_score'] = (
            0.3 * normalized_hausdorff + 
            0.3 * normalized_avg_dist + 
            0.2 * normalized_overlap + 
            0.2 * normalized_length
        )
        
        return metrics
    
    def visualize_paths(self, title: str = "Path Comparison", 
                        figsize: tuple = (10, 8),
                        show_metrics: bool = True,
                        buffer_distance: float = None) -> None:
        """
        Visualize both paths for comparison.
        
        Args:
            title: Title for the visualization
            figsize: Figure size (width, height) in inches
            show_metrics: Whether to display metrics on the plot
            buffer_distance: Optional distance to show path buffers
        """
        # Create GeoDataFrames for visualization
        pred_gdf = gpd.GeoDataFrame(geometry=[self.predicted_linestring], crs=self.crs)
        ref_gdf = gpd.GeoDataFrame(geometry=[self.reference_linestring], crs=self.crs)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot buffers if requested
        if buffer_distance is not None:
            pred_buffer = self.predicted_linestring.buffer(buffer_distance)
            ref_buffer = self.reference_linestring.buffer(buffer_distance)
            
            # Create GeoDataFrames for buffers
            pred_buffer_gdf = gpd.GeoDataFrame(geometry=[pred_buffer], crs=self.crs)
            ref_buffer_gdf = gpd.GeoDataFrame(geometry=[ref_buffer], crs=self.crs)
            
            # Plot with transparency
            pred_buffer_gdf.plot(ax=ax, color='blue', alpha=0.2)
            ref_buffer_gdf.plot(ax=ax, color='red', alpha=0.2)
            
            # Calculate and plot intersection
            intersection = pred_buffer.intersection(ref_buffer)
            gpd.GeoDataFrame(geometry=[intersection], crs=self.crs).plot(
                ax=ax, color='purple', alpha=0.4)
        
        # Plot paths
        pred_gdf.plot(ax=ax, color='blue', linewidth=2, label='Predicted Path')
        ref_gdf.plot(ax=ax, color='red', linewidth=2, label='Reference Path')
        
        # Add start and end points
        pred_start = Point(self.predicted_linestring.coords[0])
        pred_end = Point(self.predicted_linestring.coords[-1])
        ref_start = Point(self.reference_linestring.coords[0])
        ref_end = Point(self.reference_linestring.coords[-1])
        
        gpd.GeoDataFrame(geometry=[pred_start], crs=self.crs).plot(
            ax=ax, color='green', marker='o', markersize=60, label='Predicted Start')
        gpd.GeoDataFrame(geometry=[pred_end], crs=self.crs).plot(
            ax=ax, color='cyan', marker='o', markersize=60, label='Predicted End')
        gpd.GeoDataFrame(geometry=[ref_start], crs=self.crs).plot(
            ax=ax, color='yellow', marker='o', markersize=60, label='Reference Start')
        gpd.GeoDataFrame(geometry=[ref_end], crs=self.crs).plot(
            ax=ax, color='magenta', marker='o', markersize=60, label='Reference End')
        
        # Add metrics to plot if requested
        if show_metrics:
            metrics = self.evaluate_all_metrics(buffer_distance=buffer_distance or 100)
            metrics_text = "\n".join([
                f"Hausdorff Distance: {metrics['hausdorff_distance']:.2f}",
                f"Avg Min Distance: {metrics['average_min_distance']:.2f}",
                f"Buffer Overlap: {metrics['buffer_overlap']:.2f}",
                f"Path Length Ratio: {metrics['path_length_ratio']:.2f}",
                f"Combined Score: {metrics['combined_score']:.2f} (lower is better)"
            ])
            
            plt.figtext(0.02, 0.02, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title(title)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()