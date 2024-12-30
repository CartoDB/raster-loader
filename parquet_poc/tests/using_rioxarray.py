import rioxarray
import pyarrow as pa
import numpy as np

import pyarrow.parquet as pq

# Read the GeoTIFF file
geotiff_path = '/home/cayetano/Downloads/raster/classification_germany_cog.tif'
raster = rioxarray.open_rasterio(geotiff_path)

# Define the block size (window size)
block_size = (256, 256)  # Adjust the block size as needed

# Get the dimensions of the raster
height, width = raster.shape[1], raster.shape[2]

# Iterate over the raster in blocks
for i in range(0, height, block_size[0]):
    for j in range(0, width, block_size[1]):
        # Define the window
        window = raster[:, i:i+block_size[0], j:j+block_size[1]]
        
        # Convert the window to a numpy array
        window_array = window.values
        
        # Create a PyArrow table from the numpy array
        table = pa.Table.from_arrays([pa.array(window_array.flatten())], names=['values'])
        
        # Define the output Parquet file path
        parquet_path = f'output_block_{i}_{j}.parquet'
        
        # Write the table to a Parquet file
        pq.write_table(table, parquet_path)