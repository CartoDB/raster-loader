import concurrent.futures
import gc
import json
import math
import numpy as np
import os
import pandas as pd
import pyproj
import rasterio
import re
import rio_cogeo
import shutil
import sys
import traceback
import uuid
import zlib

import quadbin

from datetime import datetime, timezone
from quadbin_vectorized import (
    points_to_cells, tiles_to_cells, cells_to_tiles, cells_to_children
)
from google.cloud import bigquery, storage
from google.cloud.storage import transfer_manager

from raster_loader.io.common import (
    get_nodata_value, rasterio_metadata, is_valid_raster_dataset
)
from raster_loader import __version__


DEFAULT_COG_BLOCK_SIZE = 256

should_swap = {"=": sys.byteorder != "little", "<": False, ">": True, "|": False}


def array_to_bytes(arr):
    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes
    return arr_bytes


# function from datawarehouse.py
def band_rename_function(band_name: str):
    return band_name


def get_raster_overviews(file_path):
    with rasterio.open(file_path) as raster_src:
        return raster_src.overviews(1)


def get_resolution_and_block_sizes(
    raster_dataset: rasterio.io.DatasetReader, raster_info: dict
):
    # assuming all windows have the same dimensions
    a_window = next(raster_dataset.block_windows())
    block_width = a_window[1].width
    block_height = a_window[1].height
    resolution = int(
        raster_info["GEO"]["MaxZoom"]
        - math.log(
            block_width
            / DEFAULT_COG_BLOCK_SIZE
            * block_height
            / DEFAULT_COG_BLOCK_SIZE,
            4,
        )
    )
    return block_width, block_height, resolution


def get_raster_dataset_info(file_path, overview_level):
    raster_info = rio_cogeo.cog_info(file_path).model_dump()

    with rasterio.open(file_path, overview_level=overview_level) as raster_src:
        print(f"Processing raster with overview level {overview_level} - {raster_src.shape}")
        raster_crs = raster_src.crs.to_string()
        raster_to_4326_transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )
        _, _, base_zoom_level = get_resolution_and_block_sizes(raster_src, raster_info)

        resolution = base_zoom_level - (overview_level + 1) if overview_level is not None else base_zoom_level
        if overview_level is not None:
            overview_blocks, windows, overview_transform = prepare_overview_windows(file_path, overview_level)
        else:
            windows = [win for _, win in list(raster_src.block_windows())]
            overview_blocks, overview_transform = None, None

        print(f"Total windows: {len(windows)}")

    return {
        "transformer": raster_to_4326_transformer,
        "resolution": resolution,
        "windows": windows,
        "overview_blocks": overview_blocks,
        "overview_transform": overview_transform,
    }


def process_raster_to_parquet(file_path, chunk_id, bands_info, data_folder,
                              overview_level=None, max_workers=None):
    r_info = get_raster_dataset_info(file_path, overview_level)
    windows = r_info["windows"]
    print(f'Processing raster with resolution {r_info["resolution"]}')
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers or os.cpu_count()) as executor:
        futures = []
        for idx, chunk in enumerate(range(0, len(windows), chunk_size)):
            max_chunk_size = min(chunk + chunk_size, len(windows))
            print(f"Processing chunk {idx} [{chunk} to {max_chunk_size} of {len(windows)} rows]")
            futures.append(executor.submit(
                raster_to_parquet,
                file_path, idx, bands_info, data_folder, r_info["transformer"],
                r_info["resolution"], windows[chunk:max_chunk_size], overview_level,
                overview_blocks=r_info["overview_blocks"][chunk:max_chunk_size] if overview_level is not None else None,
                overview_transform=r_info["overview_transform"]
            ))

    total_rows_to_upload = 0
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            total_rows_to_upload += result
            print(f'future result: {result} rows')
        except Exception as err:
            print(f"An error occurred: {err} - {future}")
            traceback.print_exc()
        # finally:
        #     # Clean up memory
        #     del future
        #     gc.collect()

    return total_rows_to_upload


def raster_to_parquet(file_path, chunk_id, bands_info, data_folder, transformer,
                      resolution, windows, overview_level=None, overview_blocks=None,
                      overview_transform=None):
    with rasterio.open(file_path, overview_level=overview_level) as raster_src:
        # All windows from all bands have the same shape
        bl_width, bl_height = raster_src.block_shapes[0]
        src_width, src_height = raster_src.width, raster_src.height
        bl_width = min(bl_width, src_width)
        bl_height = min(bl_height, src_height)
        print(f"Block shape: {bl_width, bl_height} - Source shape: {src_width, src_height}")

        no_data_value = get_nodata_value(raster_src)

        if overview_level is None:
            transform = raster_src.transform
            windows_array = np.array([(win.row_off, win.col_off) for win in windows], dtype='int64')

            row_offs = windows_array[:, 0]
            col_offs = windows_array[:, 1]

            x, y = transformer.transform(
                *(transform * (col_offs + bl_width * 0.5, row_offs + bl_height * 0.5))
            )

            blocks = points_to_cells(x, y, resolution)

        else:
            blocks, transform = overview_blocks, overview_transform

        print(f"Total windows processed in chunk: {len(windows)}")
        windows_bounds = [rasterio.windows.bounds(win, transform) for win in windows]
        print(windows_bounds[0], '..............')
        window_for_array = rasterio.windows.union(windows)

        print(f"Window for array: {window_for_array}")

        window_for_array_bounds = rasterio.windows.bounds(window_for_array, transform)
        print(f"Window bounds: {window_for_array_bounds}")
        window_transform = rasterio.transform.from_bounds(
            *window_for_array_bounds, window_for_array.width, window_for_array.height
        )
        print(window_transform)

        raster_df = pd.DataFrame({
            "block": blocks,
            "win_bounds": windows_bounds
        })
        # print(f'Raster Dataframe memory usage: {raster_df.memory_usage().sum() / 1024 / 1024} MB')

        raster_df["wins_array"] = raster_df.apply(
            lambda row: rasterio.windows.from_bounds(*row["win_bounds"], window_transform).round(), axis=1
        )
        raster_df["wa_row_off"] = raster_df["wins_array"].apply(lambda win: win.row_off)
        raster_df["wa_col_off"] = raster_df["wins_array"].apply(lambda win: win.col_off)
        # print('NaN rows', raster_df[raster_df.isna()].count())
        raster_df.dropna(inplace=True)
        # print(f'Raster Dataframe memory usage: {raster_df.memory_usage().sum() / 1024 / 1024} MB')

        # if overview_level == 5:
        #     pd.set_option('display.max_columns', None)
        #     pd.set_option('max_colwidth', None)
        #     print(raster_df.head())

        # raster_df.drop(columns=["wins_array", "win_bounds"], inplace=True)
        columns_to_drop = ["wins_array", "win_bounds", "wa_row_off", "wa_col_off"]
        # print(f'Raster Dataframe memory usage: {raster_df.memory_usage().sum() / 1024 / 1024} MB')
        for band, band_name in bands_info:
            print(f"Processing band {band_name}")
            raster_arr = raster_src.read(band, window=window_for_array, boundless=True)
            print(f"Shape: {raster_arr.shape}, DType: {raster_arr.dtype}, Size: {raster_arr.nbytes / 1024 / 1024} MB")
            # print(raster_arr.__array_interface__)

            if raster_arr.size == 0:
                del raster_arr
                gc.collect()
                continue
            raster_df[band_name] = raster_df.apply(
                lambda row: raster_arr[
                    row["wa_row_off"]: row["wa_row_off"] + bl_height,
                    row["wa_col_off"]: row["wa_col_off"] + bl_width
                ].flatten(), axis=1
            )
            band_name_empty = f"{band_name}_empty"
            columns_to_drop.append(band_name_empty)
            raster_df[band_name_empty] = raster_df.apply(
                lambda row: np.all(row[band_name] == no_data_value)
                or row[band_name].size == 0, axis=1
            )
            # print(f'Raster Dataframe memory usage: {raster_df.memory_usage().sum() / 1024 / 1024} MB')
            # raster_df[band_name] = raster_df.apply(lambda row: zlib.compress(array_to_bytes(row[band_name])(), level=9, wbits=31), axis=1)
            raster_df[band_name] = raster_df.apply(lambda row: array_to_bytes(row[band_name])(), axis=1)
            # print(f'Raster Dataframe memory usage: {raster_df.memory_usage().sum() / 1024 / 1024} MB')
            del raster_arr
            gc.collect()
        raster_df = raster_df[~raster_df.apply(
            lambda row: all(row[f"{band_name}_empty"] for _, band_name in bands_info),
            axis=1
        )]
        # print(f'Raster Dataframe memory usage: {raster_df.memory_usage().sum() / 1024 / 1024} MB')

        if raster_df.shape[0] > 0:
            raster_df.drop(columns=columns_to_drop, inplace=True)
            raster_df.reset_index(drop=True, inplace=True)

            out_file_name = f"raster_{f'ov{overview_level}' if overview_level is not None else ''}ch{chunk_id}.parquet"
            out_file_path = os.path.join(data_folder, out_file_name)
            print(f'Writing parquet file: {out_file_path}')
            raster_df.to_parquet(
                out_file_path, index=None, row_group_size=1000, engine="pyarrow", compression="snappy"
            )
        # print(f'Raster Dataframe memory usage: {raster_df.memory_usage().sum() / 1024 / 1024} MB')
        print(raster_df.shape)
        rows_to_upload = raster_df.shape[0]
        del raster_df
        gc.collect()
        return rows_to_upload


def prepare_overview_windows(file_path, overview_index):
    raster_info = rio_cogeo.cog_info(file_path).model_dump()
    with rasterio.open(file_path) as raster_dataset:
        is_valid_raster_dataset(raster_dataset)

        block_width, block_height, resolution = get_resolution_and_block_sizes(
            raster_dataset, raster_info
        )
        raster_crs = raster_dataset.crs.to_string()
        raster_to_4326_transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )
        pixels_to_raster_transform = raster_dataset.transform

        overview_factors = raster_dataset.overviews(1)
        (block_width, block_height) = raster_dataset.block_shapes[0]

        min_base_tile_lng, min_base_tile_lat = raster_to_4326_transformer.transform(
            *(pixels_to_raster_transform * (block_width * 0.5, block_height * 0.5))
        )
        max_base_tile_lng, max_base_tile_lat = raster_to_4326_transformer.transform(
            *(
                pixels_to_raster_transform
                * (
                    raster_dataset.width - block_width * 0.5,
                    raster_dataset.height - block_height * 0.5,
                )
            )
        )
        # quadbin cell at base resolution
        min_base_tile = quadbin.point_to_cell(
            min_base_tile_lng, min_base_tile_lat, resolution
        )
        min_base_x, min_base_y, _z = quadbin.cell_to_tile(min_base_tile)

        # quadbin cell at overview resolution (quadbin_tile -> quadbin_cell)
        min_tile = quadbin.point_to_cell(
            min_base_tile_lng, min_base_tile_lat, resolution - overview_index - 1
        )
        max_tile = quadbin.point_to_cell(
            max_base_tile_lng, max_base_tile_lat, resolution - overview_index - 1
        )
        min_x, min_y, min_z = quadbin.cell_to_tile(min_tile)
        max_x, max_y, _z = quadbin.cell_to_tile(max_tile)

        factor = overview_factors[overview_index]

        tile_xs, tile_ys = np.meshgrid(
            np.arange(min_x, max_x + 1),
            np.arange(min_y, max_y + 1)
        )
        tile_xs = tile_xs.flatten()
        tile_ys = tile_ys.flatten()

        cells = tiles_to_cells(tile_xs, tile_ys, min_z)
        children = cells_to_children(cells, resolution)
        children_tiles = cells_to_tiles(children).reshape(children.shape[0], 3, children.shape[1])

        min_child_xs = np.min(children_tiles, axis=2)[:, 0]
        min_child_ys = np.min(children_tiles, axis=2)[:, 1]

        factor = overview_factors[overview_index]

        col_offs = (block_width * (min_child_xs - min_base_x)) // factor
        row_offs = (block_height * (min_child_ys - min_base_y)) // factor
        widths = np.full_like(col_offs, block_width)
        heights = np.full_like(row_offs, block_height)

        windows = [
            rasterio.windows.Window(int(col_off), int(row_off), int(width), int(height))
            for col_off, row_off, width, height in zip(col_offs, row_offs, widths, heights)
        ]

        # Affine transform from first window (upper, left)
        ov_transform = rasterio.transform.Affine(
            raster_dataset.transform[0] * factor,
            0.0,
            raster_dataset.transform[2] + (windows[0].col_off * raster_dataset.transform[0]),
            0.0,
            raster_dataset.transform[4] * factor,
            raster_dataset.transform[5] - (windows[0].row_off * raster_dataset.transform[0])
        )

        print(f'Overview {overview_index} Transform: {ov_transform}')
        return cells, windows, ov_transform


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def upload_parquet_from_gcs(bucket_name, blob_name, table_id, bands_info):
    client = bigquery.Client()
    uri = f"gs://{bucket_name}/{blob_name}"
    band_columns = [
        bigquery.SchemaField(band_name, bigquery.enums.SqlTypeNames.BYTES)
        for _, band_name in bands_info
    ]
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        schema=[
            bigquery.SchemaField("block", bigquery.enums.SqlTypeNames.INT64)
        ] + band_columns,
        clustering_fields=["block"],
    )
    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()  # Wait for the job to complete
    print(f"Loaded {blob_name} into {table_id}")


def upload_parquets_to_bigquery(data_folder, bucket_name, bucket_folder, output_table, bands_info):
    parquet_files = [
        file_name for file_name in os.listdir(data_folder)
        if file_name.endswith(".parquet")
    ]

    storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024
    storage.blob._MAX_MULTIPART_SIZE = 5 * 1024 * 1024

    print(f"Uploading parquet files to GCS bucket {bucket_name}")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_results = transfer_manager.upload_many_from_filenames(
        bucket,
        parquet_files,
        source_directory=data_folder,
        blob_name_prefix=bucket_folder,
        deadline=None,
        max_workers=os.cpu_count()
    )

    for file_name, result in zip(parquet_files, gcs_results):
        if isinstance(result, Exception):
            print("Failed to upload {} due to exception: {}".format(file_name, result))
        else:
            print("Uploaded {} to GCS bucket [{}].".format(file_name, bucket_name))

    print(f"Moving parquet files from GCS to BigQuery table {output_table}")
    upload_parquet_from_gcs(bucket_name, os.path.join(bucket_folder, '*.parquet'), output_table, bands_info)


def add_metadata_to_bigquery_table(output_table, metadata):
    client = bigquery.Client()
    table = client.get_table(output_table)

    # Add new column "metadata" to the table
    new_schema = table.schema[:]
    new_schema.append(bigquery.SchemaField("metadata", "STRING"))
    table.schema = new_schema
    table.labels = {
        "raster_loader": re.sub(r"[^a-z0-9_-]", "_", __version__.lower())
    }
    table = client.update_table(table, ["schema", "labels"])

    # Insert metadata into the new column
    rows_to_insert = [
        {"block": 0, "metadata": json.dumps(metadata)}
    ]
    errors = client.insert_rows_json(output_table, rows_to_insert)
    if errors:
        print(f"Errors occurred while inserting metadata: {errors}")
    else:
        print("Metadata inserted successfully.")


def prepare_outputs(table_id_sufix):
    time_now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    upload_id = f"{time_now}_{uuid.uuid4().hex.replace('-', '')}"

    output_data_folder = f"/tmp/raster_parquet_data/{upload_id}/"
    if os.path.exists(output_data_folder):
        shutil.rmtree(output_data_folder)
    os.makedirs(output_data_folder)

    output_table = f"{table_id_sufix}_{upload_id}"
    output_bucket_folder = f'raster_parquet/{upload_id}/'

    return output_data_folder, output_table, output_bucket_folder


def process_raster_data(file_path, chunk_size, bands_info, data_folder, max_workers=None):
    """Process raster data including overviews and base resolution into parquet files.
    
    Args:
        file_path (str): Path to the input raster file
        chunk_size (int): Size of chunks to process
        bands_info (list): List of tuples containing band information (band, band_name)
        data_folder (str): Output folder for parquet files
        max_workers (int, optional): Maximum number of worker processes
    """
    overviews = get_raster_overviews(file_path)
    print(f'Processing overviews: {overviews}')
    
    # Process overviews except the minimum zoom level
    for ov_idx, overview in enumerate(overviews[:-1]):  # Skip last overview
        print(f"\nProcessing overview [{ov_idx}] - level {overview}")
        process_raster_to_parquet(
            file_path, 
            chunk_size, 
            bands_info, 
            data_folder, 
            overview_level=ov_idx
        )

    # Process base resolution
    print("\nProcessing base resolution")
    process_raster_to_parquet(
        file_path, 
        chunk_size, 
        bands_info, 
        data_folder, 
        max_workers=max_workers
    )


def main(file_path, table_id_sufix, chunk_size, band, band_name, 
         bucket_name, max_workers=None):
    """Main function to process raster data and upload to BigQuery.
    
    Args:
        file_path (str): Path to the input raster file
        table_id_sufix (str): Suffix for the BigQuery table ID
        chunk_size (int): Size of chunks to process
        band (list): List of band numbers
        band_name (list): List of band names
        bucket_name (str): GCS bucket name
        max_workers (int, optional): Maximum number of worker processes
    """
    data_folder, output_table, bucket_folder = prepare_outputs(table_id_sufix)
    bands_info = list(zip(band, band_name))
    print(f'Bands info: {bands_info}')
    
    # Get metadata using default band rename function
    metadata = rasterio_metadata(file_path, bands_info, lambda band_name: band_name)
    
    # Process raster data
    process_raster_data(file_path, chunk_size, bands_info, data_folder, max_workers)
    
    # Upload to BigQuery
    upload_parquets_to_bigquery(data_folder, bucket_name, bucket_folder, output_table, bands_info)
    add_metadata_to_bigquery_table(output_table, metadata)


if __name__ == "__main__":
    # # Test case 1: medium size raster, 1 band (Byte). gs://carto-ps-raster-data-examples/geotiff/blended_output_cog.tif
    # chunk_size = 1000
    # max_workers = None
    # file_path = "/home/cayetano/Downloads/raster/classification_germany_cog.tif"
    # band, band_name = ([1], ["band_1"])

    # # Test case 2: big raster (3.5GB), 3 bands (Byte). gs://carto-ps-raster-data-examples/geotiff/blended_output_cog.tif
    # chunk_size = 1000
    # max_workers = None
    # file_path = "/home/cayetano/Downloads/raster/blended_output_cog.tif"
    # band, band_name = ([1, 2, 3], ["band_1", "band_2", "band_3"])

    # # Test case 3: big raster (8.5GB), 2 bands (Float32). gs://carto-ps-raster-data-examples/geotiff/output_5band_cog.tif
    # chunk_size = 1000
    # max_workers = 8
    # file_path = "/home/cayetano/Downloads/raster/output_5band_cog.tif"
    # band, band_name = ([1, 2], ["band_1", "band_2"])

    # # Test case 5 small raster, 1 band (Byte). gs://carto-ps-raster-data-examples/geotiff/corelogic_wind/20211201_forensic_wind_banded_cog.tif
    chunk_size = 10000
    max_workers = None
    file_path = "/home/cayetano/Downloads/raster/corelogic/202112geotiffs/cog/20211201_forensic_wind_banded_cog.tif"
    band, band_name = ([1], ["band_1"])

    # # Test case 6: big sparse raster, 1 band (Byte). 30m resolution, entire world. gs://carto-ps-raster-data-examples/geotiff/discreteloss_2023_COG.tif
    # chunk_size = 1000
    # max_workers = 8
    # file_path = "/home/cayetano/Downloads/raster/discreteloss_2023_COG.tif"
    # band, band_name = ([1], ["band_1"])

    # # Test case 7: big raster, 1 band (3.3GB, Byte). gs://carto-ps-raster-data-examples/geotiff/usda_usda_us_aggriculture_cog.tif
    # chunk_size = 5000
    # max_workers = None
    # file_path = "/home/cayetano/Downloads/raster/usda_usda_us_aggriculture_cog.tif"
    # band, band_name = ([1], ["band_1"])
    
    # # Test case 7: big raster, 1 band (3.3GB, Byte). gs://carto-ps-raster-data-examples/geotiff/Public_flood_flood_prone_areas_global_cog.tif
    # chunk_size = 1000
    # max_workers = None
    # file_path = "/home/cayetano/Downloads/raster/Public_flood_flood_prone_areas_global_cog.tif"
    # band, band_name = ([1], ["band_1"])

    bucket_name = "cayetanobv-data"
    table_id_sufix = "cartobq.cayetanobv_raster.raster_parquet_test"

    main(file_path, table_id_sufix, chunk_size, band, band_name, bucket_name, max_workers)
