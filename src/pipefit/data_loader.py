"""
Module for loading and handling geospatial datasets for pipeline routing analysis.
"""
import os
import rasterio
import geopandas as gpd
import numpy as np
from glob import glob
from typing import Dict, List, Tuple, Optional


class DataLoader:
    """
    Loads and manages geospatial datasets for pipeline suitability analysis.
    Handles raster factors.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize the data loader with the path to a dataset folder.
        
        Args:
            dataset_path: Path to the dataset folder containing raster factors
        """
        self.dataset_path = dataset_path
        self.factors = {}  # Dictionary to store raster factors
        self.factor_metadata = {}  # Metadata for each raster factor
        self.factor_mapping = {}  # Mapping between original factor names and standardized names
        
    def load_data(self, factor_mapping: Optional[Dict[str, str]] = None) -> None:
        """
        Load all raster factors from the dataset folder.
        
        Args:
            factor_mapping: Optional mapping from file names to standardized factor names
                           e.g., {'elev.tif': 'elevation', 'protected.tif': 'protected_areas'}
        """
        self.factor_mapping = factor_mapping or {}
        self._load_raster_factors()
        
    def _load_raster_factors(self) -> None:
        """
        Load all raster factors from the factors subfolder.
        Each factor should be a GeoTIFF with values normalized between 0 and 1.
        """
        factors_path = os.path.join(self.dataset_path, 'factors')
        if not os.path.exists(factors_path):
            raise FileNotFoundError(f"Factors folder not found at {factors_path}")
        
        # Find all raster files in the factors folder
        raster_files = glob(os.path.join(factors_path, '*.tif'))
        
        for raster_file in raster_files:
            # Get the base filename without extension
            base_filename = os.path.basename(raster_file)
            
            # Apply factor mapping if provided, otherwise use the base filename without extension
            if base_filename in self.factor_mapping:
                factor_name = self.factor_mapping[base_filename]
            else:
                factor_name = os.path.splitext(base_filename)[0]
                
            with rasterio.open(raster_file) as src:
                # Store the raster data
                self.factors[factor_name] = src.read(1)
                
                # Store metadata for later use
                self.factor_metadata[factor_name] = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'width': src.width,
                    'height': src.height,
                    'bounds': src.bounds
                }
        
        print(f"Loaded {len(self.factors)} raster factors: {list(self.factors.keys())}")
    
    def get_common_metadata(self) -> Dict:
        """
        Returns common metadata (transform, CRS, dimensions) from the first factor.
        Assumes all factors have the same spatial properties.
        
        Returns:
            Dict with common metadata
        """
        if not self.factor_metadata:
            raise ValueError("No factor metadata available. Load data first.")
            
        # Use the first factor's metadata as reference
        first_factor = next(iter(self.factor_metadata.values()))
        return {
            'transform': first_factor['transform'],
            'crs': first_factor['crs'],
            'width': first_factor['width'],
            'height': first_factor['height'],
            'bounds': first_factor['bounds']
        }
    
    def save_raster(self, data: np.ndarray, output_path: str, metadata: Optional[Dict] = None) -> None:
        """
        Save a numpy array as a GeoTIFF raster file.
        
        Args:
            data: 2D numpy array to save
            output_path: Path to save the raster
            metadata: Optional metadata dictionary (if None, uses common metadata)
        """
        if metadata is None:
            metadata = self.get_common_metadata()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=metadata['crs'],
            transform=metadata['transform'],
            compress='deflate'
        ) as dst:
            dst.write(data, 1)
            
        print(f"Saved raster to {output_path}")
    
    def coordinates_to_indices(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert geographic coordinates to raster indices.
        
        Args:
            x: X-coordinate (longitude)
            y: Y-coordinate (latitude)
            
        Returns:
            Tuple of (row, col) indices
        """
        metadata = self.get_common_metadata()
        transform = metadata['transform']
        
        col, row = ~transform * (x, y)
        return int(row), int(col)
    
    def indices_to_coordinates(self, row: int, col: int) -> Tuple[float, float]:
        """
        Convert raster indices to geographic coordinates.
        
        Args:
            row: Row index
            col: Column index
            
        Returns:
            Tuple of (x, y) coordinates
        """
        metadata = self.get_common_metadata()
        transform = metadata['transform']
        
        x, y = transform * (col, row)
        return x, y


def load_reference_path(path: str) -> gpd.GeoDataFrame:
    """
    Load a reference path from a GeoJSON file.
    
    Args:
        path: Path to the GeoJSON file or directory containing a GeoJSON file
        
    Returns:
        GeoDataFrame containing the reference path
    """
    # If a directory is provided, look for GeoJSON files
    if os.path.isdir(path):
        geojson_pattern = os.path.join(path, '*.geojson')
        # Also search for .json files that might be GeoJSON format
        json_pattern = os.path.join(path, '*.json')
        
        geojson_list = glob(geojson_pattern) + glob(json_pattern)
        
        if not geojson_list:
            raise FileNotFoundError(f"No GeoJSON file found in {path}")
        
        # Use the first GeoJSON file found
        path = geojson_list[0]
    
    # Load the GeoJSON file
    gdf = gpd.read_file(path, driver='GeoJSON')
    
    print(f"Loaded reference path from {path}")
    return gdf