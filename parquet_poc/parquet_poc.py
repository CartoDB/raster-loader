import concurrent.futures
import gc
import math
import numpy as np
import os
import pandas as pd
import pyproj
import rasterio
import rio_cogeo
import shutil
import sys
import traceback
import uuid

from datetime import datetime, timezone
from quadbin_vectorized import points_to_cells
from google.cloud import bigquery, storage
from google.cloud.storage import transfer_manager


DEFAULT_COG_BLOCK_SIZE = 256
DEFAULT_TYPES_NODATA_VALUES = {
    "int8": -128,
    "int16": -32768,
    "int32": -2147483648,
    "int64": -9223372036854775808,
    "uint8": 0,
    "uint16": 65535,
    "uint32": 4294967295,
    "uint64": 18446744073709551615,
    "float16": np.nan,
    "float32": np.nan,
    "float64": np.nan,
}

should_swap = {"=": sys.byteorder != "little", "<": False, ">": True, "|": False}


def array_to_bytes(arr):
    if should_swap[arr.dtype.byteorder]:
        arr_bytes = np.ascontiguousarray(arr.byteswap()).tobytes
    else:
        arr_bytes = np.ascontiguousarray(arr).tobytes
    return arr_bytes


def get_default_nodata_value(dtype: str) -> float:
    if dtype in DEFAULT_TYPES_NODATA_VALUES:
        return DEFAULT_TYPES_NODATA_VALUES[dtype]
    else:
        raise ValueError(f"Unsupported data type: {dtype}")


def band_original_nodata_value(
    raster_dataset: rasterio.io.DatasetReader, band: int
) -> float:
    nodata_value = raster_dataset.nodata
    if nodata_value is None:
        nodata_value = raster_dataset.get_nodatavals()[band - 1]
    return nodata_value


def band_nodata_value(raster_dataset: rasterio.io.DatasetReader, band: int) -> float:
    nodata_value = band_original_nodata_value(raster_dataset, band)
    if nodata_value is None:
        nodata_value = get_default_nodata_value(raster_dataset.dtypes[band - 1])
    return nodata_value


def get_nodata_value(raster_dataset: rasterio.io.DatasetReader) -> float:
    value = band_nodata_value(raster_dataset, 1)
    # So far we only support one nodata value for all bands
    if raster_dataset.nodata is None:
        for band in raster_dataset.indexes:
            band_value = band_nodata_value(raster_dataset, band)
            # Note (np.nan != np.nan) == True
            both_nan = np.isnan(band_value) and np.isnan(value)
            if (not both_nan) and (band_value != value):
                raise ValueError("Invalid no data value")
    return value


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


def get_raster_dataset_info(file_path):
    with rasterio.open(file_path) as raster_src:
        raster_info = rio_cogeo.cog_info(file_path).model_dump()
        raster_crs = raster_src.crs.to_string()
        raster_to_4326_transformer = pyproj.Transformer.from_crs(
            raster_crs, "EPSG:4326", always_xy=True
        )
        pixels_to_raster_transform = raster_src.transform
        _, _, resolution = get_resolution_and_block_sizes(raster_src, raster_info)

        windows = list(raster_src.block_windows())

    return raster_to_4326_transformer, pixels_to_raster_transform, resolution, windows


def raster_to_parquet(file_path, chunk_id, bands_info, data_folder, transformer,
                      raster_transform, resolution, windows):
    with rasterio.open(file_path) as raster_src:
        windows_array = np.array([(win.row_off, win.col_off) for _, win in windows], dtype='int64')
        windows_bounds = [rasterio.windows.bounds(win, raster_transform) for _, win in windows]

        row_offs = windows_array[:, 0]
        col_offs = windows_array[:, 1]

        # All windows from all bands have the same shape
        bl_width, bl_height = raster_src.block_shapes[0]

        x, y = transformer.transform(
            *(raster_transform * (col_offs + bl_width * 0.5, row_offs + bl_height * 0.5))
        )

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
        window_for_array = rasterio.windows.Window.from_slices((row_offs_min, row_offs_max + bl_height), (col_offs_min, col_offs_max + bl_width))
        # window_for_array2 = rasterio.windows.union([win for _, win in windows])
        print('####', (row_offs_min, row_offs_max), (col_offs_min, col_offs_max))
        print(window_for_array)
        # print(window_for_array2)
        print(raster_df.shape)
        print(rasterio.windows.bounds(windows[0][1], raster_transform), windows[0])
        window_for_array_bounds = rasterio.windows.bounds(window_for_array, raster_transform)
        print(window_for_array_bounds)
        window_transform = rasterio.transform.from_bounds(*window_for_array_bounds, window_for_array.width, window_for_array.height)
        print(window_transform)
        raster_df["wins_array"] = raster_df.apply(lambda row: rasterio.windows.from_bounds(*row["win_bounds"], window_transform), axis=1)
        raster_df["wa_row_off"] = raster_df["wins_array"].apply(lambda win: win.row_off)
        raster_df["wa_col_off"] = raster_df["wins_array"].apply(lambda win: win.col_off)
        print(raster_df.head())
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
                    int(row["wa_row_off"]): int(row["wa_row_off"] + bl_height),
                    int(row["wa_col_off"]): int(row["wa_col_off"] + bl_width)
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

        raster_df.drop(columns=columns_to_drop, inplace=True)
        raster_df.reset_index(drop=True, inplace=True)

        out_file_name = os.path.join(data_folder, f"raster_{chunk_id}.parquet")
        print(f'Writing parquet file: {out_file_name}')
        raster_df.to_parquet(
            out_file_name, index=None, row_group_size=1000, engine="pyarrow", compression="snappy"
        )

        # print(raster_df.head())
        print(raster_df.shape)
        rows_to_upload = raster_df.shape[0]
        del raster_df
        gc.collect()
        return rows_to_upload


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def upload_parquet_from_gcs(bucket_name, blob_name, table_id):
    client = bigquery.Client()
    uri = f"gs://{bucket_name}/{blob_name}"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        clustering_fields=["block"],
    )
    load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
    load_job.result()  # Wait for the job to complete
    print(f"Loaded {blob_name} into {table_id}")


def upload_parquets_to_bigquery(data_folder, bucket_name, table_id, upload_id):
    parquet_files = [
        file_name for file_name in os.listdir(data_folder)
        if file_name.endswith(".parquet")
    ]
    bucket_folder = f'raster_parquet/{upload_id}/'

    print(f"Uploading parquet files to GCS bucket {bucket_name}")
    storage.blob._DEFAULT_CHUNKSIZE = 10 * 1024 * 1024
    storage.blob._MAX_MULTIPART_SIZE = 10 * 1024 * 1024
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

    output_table = f"{table_id}_{upload_id}"
    print(f"Uploading parquet files to BigQuery table {output_table}")
    upload_parquet_from_gcs(bucket_name, os.path.join(bucket_folder, '*.parquet'), output_table)


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

    transformer, raster_transform, resolution, windows = get_raster_dataset_info(file_path)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for idx, chunk in enumerate(range(0, len(windows), chunk_size)):
            max_chunk_size = min(chunk + chunk_size, len(windows))
            print(f"Processing chunk {idx} [{chunk} to {max_chunk_size} of {len(windows)} rows]")
            futures.append(executor.submit(
                raster_to_parquet,
                file_path, idx, bands_info, data_folder, transformer,
                raster_transform, resolution, windows[chunk:max_chunk_size]
            ))

    total_rows_to_upload = 0
    for future in concurrent.futures.as_completed(futures):
        try:
            result = future.result()
            total_rows_to_upload += result
            print(f'future result: {result} rows')
        except Exception as e:
            print(f"An error occurred: {e} - {future}")
            traceback.print_exc()
    
    print(f"Total rows to upload: {total_rows_to_upload}")

    bucket_name = "cayetanobv-data"
    table_id_sufix = "cartobq.cayetanobv_raster.raster_parquet_test"
    upload_parquets_to_bigquery(data_folder, bucket_name, table_id_sufix, upload_id)
