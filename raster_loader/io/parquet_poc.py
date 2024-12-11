import concurrent.futures
import gc
import numpy as np
import os
import pandas as pd
import pyproj
import rasterio
import rio_cogeo
import math
import shutil
import sys
import traceback
import uuid
import json
import re

from datetime import datetime, timezone
from quadbin_vectorized import points_to_cells, cell_to_tile
from google.cloud import bigquery, storage
from google.cloud.storage import transfer_manager

from raster_loader.io.common import get_nodata_value, rasterio_metadata, is_valid_raster_dataset
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
        windows = list(raster_src.block_windows())
        print(f"Total windows: {len(windows)}")

    return raster_to_4326_transformer, resolution, windows


def process_raster_to_parquet(file_path, chunk_size, bands_info, data_folder, overview_level=None):
    transformer, resolution, windows = get_raster_dataset_info(file_path, overview_level)
    print('||||||||||||||||||||||||||')
    print(f"Processing raster with resolution {resolution}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for idx, chunk in enumerate(range(0, len(windows), chunk_size)):
            max_chunk_size = min(chunk + chunk_size, len(windows))
            print(f"Processing chunk {idx} [{chunk} to {max_chunk_size} of {len(windows)} rows]")
            futures.append(executor.submit(
                raster_to_parquet,
                file_path, idx, bands_info, data_folder, transformer,
                resolution, overview_level, windows[chunk:max_chunk_size]
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
    
    return total_rows_to_upload


def raster_to_parquet(file_path, chunk_id, bands_info, data_folder, transformer,
                      resolution, overview_level, windows):
    with rasterio.open(file_path, overview_level=overview_level) as raster_src:
        print(f"Total windows processed in chunk: {len(windows)}")
        windows_array = np.array([(win.row_off, win.col_off) for _, win in windows], dtype='int64')
        windows_bounds = [rasterio.windows.bounds(win, raster_src.transform) for _, win in windows]

        row_offs = windows_array[:, 0]
        col_offs = windows_array[:, 1]

        # All windows from all bands have the same shape
        bl_width, bl_height = raster_src.block_shapes[0]
        src_width, src_height = raster_src.width, raster_src.height
        print(f"Block shape: {bl_width, bl_height} - Source shape: {src_width, src_height}")
        bl_width = min(bl_width, src_width)
        bl_height = min(bl_height, src_height)

        x, y = transformer.transform(
            *(raster_src.transform * (col_offs + bl_width * 0.5, row_offs + bl_height * 0.5))
        )
        print(np.column_stack((x, y)))

        blocks = points_to_cells(x, y, resolution)

        raster_df = pd.DataFrame({
            "block": blocks,
            "win_bounds": windows_bounds
        })

        no_data_value = get_nodata_value(raster_src)

        row_offs_max = row_offs.max()
        col_offs_max = col_offs.max()
        row_offs_min = row_offs.min()
        col_offs_min = col_offs.min()
        # window_for_array = rasterio.windows.Window.from_slices((row_offs_min, row_offs_max + bl_height), (col_offs_min, col_offs_max + bl_width))
        window_for_array = rasterio.windows.union([win for _, win in windows])
        # print(rasterio.windows.bounds(window_for_array, raster_transform))
        print('####', (row_offs_min, row_offs_max), (col_offs_min, col_offs_max))
        print(window_for_array)
        print(raster_df.shape)
        print(rasterio.windows.bounds(windows[0][1], raster_src.transform), windows[0])
        window_for_array_bounds = rasterio.windows.bounds(window_for_array, raster_src.transform)
        print('------------------', window_for_array_bounds)
        window_transform = rasterio.transform.from_bounds(*window_for_array_bounds, window_for_array.width, window_for_array.height)
        print(window_transform)
        raster_df["wins_array"] = raster_df.apply(
            lambda row: rasterio.windows.from_bounds(*row["win_bounds"], window_transform).round(), axis=1
        )
        raster_df["wa_row_off"] = raster_df["wins_array"].apply(lambda win: win.row_off)
        raster_df["wa_col_off"] = raster_df["wins_array"].apply(lambda win: win.col_off)
        print('NaN rows', raster_df[raster_df.isna()].count())
        raster_df.dropna(inplace=True)

        columns_to_drop = ["win_bounds", "wins_array", "wa_row_off", "wa_col_off"]
        for band, band_name in bands_info:
            print(f"Processing band {band_name}")
            raster_arr = raster_src.read(band, window=window_for_array, boundless=True)
            # print(raster_arr.__array_interface__)
            # print(raster_src.profile)
            print('@@@', raster_arr.shape, raster_arr.dtype, raster_arr.nbytes/1024/1024)
            if raster_arr.size == 0:
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
            raster_df[band_name] = raster_df.apply(lambda row: array_to_bytes(row[band_name])(), axis=1)
            del raster_arr
            gc.collect()
        raster_df = raster_df[~raster_df.apply(
            lambda row: all(row[f"{band_name}_empty"] for _, band_name in bands_info),
            axis=1
        )]

        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(raster_df[['block', 'wins_array', 'wa_row_off', 'wa_col_off']].tail())

        raster_df.drop(columns=columns_to_drop, inplace=True)
        raster_df.reset_index(drop=True, inplace=True)

        out_file_name = f"raster_{f'ov{overview_level}' if overview_level is not None else ''}ch{chunk_id}.parquet"
        out_file_path = os.path.join(data_folder, out_file_name)
        print(f'Writing parquet file: {out_file_path}')
        raster_df.to_parquet(
            out_file_path, index=None, row_group_size=1000, engine="pyarrow", compression="snappy"
        )

        # print(raster_df.head())
        print(raster_df.shape)
        rows_to_upload = raster_df.shape[0]
        del raster_df
        gc.collect()
        return rows_to_upload


# def prepare_overview_windows(file_path: str):
#     raster_info = rio_cogeo.cog_info(file_path).dict()
#     with rasterio.open(file_path) as raster_dataset:
#         block_width, block_height, resolution = get_resolution_and_block_sizes(
#             raster_dataset, raster_info
#         )
#         raster_crs = raster_dataset.crs.to_string()

#         raster_to_4326_transformer = pyproj.Transformer.from_crs(
#             raster_crs, "EPSG:4326", always_xy=True
#         )
#         pixels_to_raster_transform = raster_dataset.transform
#         is_valid_raster_dataset(raster_dataset)

#         overview_factors = raster_dataset.overviews(1)
#         (block_width, block_height) = raster_dataset.block_shapes[0]

#         overview_data = []

#         for overview_index in range(0, len(overview_factors)):
#             min_base_tile_lng, min_base_tile_lat = raster_to_4326_transformer.transform(
#                 *(pixels_to_raster_transform * (block_width * 0.5, block_height * 0.5))
#             )
#             max_base_tile_lng, max_base_tile_lat = raster_to_4326_transformer.transform(
#                 *(
#                     pixels_to_raster_transform
#                     * (
#                         raster_dataset.width - block_width * 0.5,
#                         raster_dataset.height - block_height * 0.5,
#                     )
#                 )
#             )
#             min_base_tile = points_to_cells(
#                 np.array([min_base_tile_lng]), np.array([min_base_tile_lat]), resolution
#             )[0]
#             min_base_x, min_base_y, _z = cell_to_tile(min_base_tile)

#             min_tile = points_to_cells(
#                 np.array([min_base_tile_lng]), np.array([min_base_tile_lat]), resolution - overview_index - 1
#             )[0]
#             max_tile = points_to_cells(
#                 np.array([max_base_tile_lng]), np.array([max_base_tile_lat]), resolution - overview_index - 1
#             )[0]
#             min_x, min_y, min_z = cell_to_tile(min_tile)
#             max_x, max_y, _z = cell_to_tile(max_tile)

#             tile_xs, tile_ys = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
#             tile_xs = tile_xs.flatten()
#             tile_ys = tile_ys.flatten()

#             children = points_to_cells(tile_xs, tile_ys, resolution)
#             children_tiles = np.column_stack(cell_to_tile(children))
#             child_xs = children_tiles[:, 0]
#             child_ys = children_tiles[:, 1]
#             min_child_x, max_child_x = child_xs.min(), child_xs.max()
#             min_child_y, max_child_y = child_ys.min(), child_ys.max()
#             factor = overview_factors[overview_index]

#             tile_windows = [
#                 rasterio.windows.Window(
#                     col_off=block_width * (min_child_x - min_base_x),
#                     row_off=block_height * (min_child_y - min_base_y),
#                     width=(max_child_x - min_child_x + 1) * block_width,
#                     height=(max_child_y - min_child_y + 1) * block_height,
#                 )
#                 for min_child_x, min_child_y, max_child_x, max_child_y in zip(
#                     child_xs, child_ys, child_xs + factor, child_ys + factor
#                 )
#             ]

#             overview_data.extend([
#                 {"block": quadbin.tile_to_cell((tile_x, tile_y, min_z)), "window": tile_window}
#                 for tile_x, tile_y, tile_window in zip(tile_xs, tile_ys, tile_windows)
#             ])

#         return pd.DataFrame(overview_data)


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

    print(f"Uploading parquet files to BigQuery table {output_table}")
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


if __name__ == "__main__":
    chunk_size = 1000
    # file_path = "/home/cayetano/Downloads/raster/classification_germany_cog.tif"
    # file_path = "/home/cayetano/Downloads/raster/corelogic/202112geotiffs/cog/20211201_forensic_wind_banded_cog.tif"
    # band, band_name = ([1], ["band_1"])
    # file_path = "/home/cayetano/Downloads/raster/output_5band_cog.tif"
    # band, band_name = ([1, 2], ["band_1", "band_2"])
    file_path = "/home/cayetano/Downloads/raster/blended_output_cog.tif"
    band, band_name = ([1, 2, 3], ["band_1", "band_2", "band_3"])

    time_now = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    upload_id = f"{time_now}_{uuid.uuid4().hex.replace('-', '')}"
    data_folder = f"/tmp/raster_parquet_data/{upload_id}/"
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    os.makedirs(data_folder)

    bands_info = list(zip(band, band_name))

    metadata = rasterio_metadata(file_path, bands_info, band_rename_function)

    # overviews = get_raster_overviews(file_path)
    # print(overviews)
    # for ov_idx, overview in enumerate(overviews):
    #     print(f"Processing overview [{ov_idx}] - level {overview}")
    #     process_raster_to_parquet(file_path, chunk_size, bands_info, data_folder, overview_level=ov_idx)

    process_raster_to_parquet(file_path, chunk_size, bands_info, data_folder)

    bucket_name = "cayetanobv-data"
    table_id_sufix = "cartobq.cayetanobv_raster.raster_parquet_test"
    output_table = f"{table_id_sufix}_{upload_id}"
    bucket_folder = f'raster_parquet/{upload_id}/'
    upload_parquets_to_bigquery(data_folder, bucket_name, bucket_folder, output_table, bands_info)

    add_metadata_to_bigquery_table(output_table, metadata)