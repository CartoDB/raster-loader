from google.cloud import bigquery
import pandas as pd
import pyproj
import rasterio
from tqdm import tqdm

from . import errors, utils


class RasterLoader:
    """Upload a raster file to a data warehouse.

    Args:
        file_path (str): The path to the raster file.
        dst_crs (int, default = 4326): The EPSG code of the destination CRS.
        dst_columns (list, default = ['lat_NW', 'lon_NW', 'lat_NE', 'lon_NE', \
            'lat_SE', 'lon_SE', 'lat_SW', 'lon_SW', 'block_height', \
            'block_width', 'land_usage_cat_int8']): The columns of the \
            destination table.
    """

    def __init__(
        self,
        file_path,
        dst_crs=4326,
        dst_columns=[
            "lat_NW",
            "lon_NW",
            "lat_NE",
            "lon_NE",
            "lat_SE",
            "lon_SE",
            "lat_SW",
            "lon_SW",
            "block_height",
            "block_width",
            "land_usage_cat_int8",
        ],
    ):
        self.file_path = file_path
        self.dst_crs = pyproj.CRS.from_epsg(dst_crs)
        self.dst_columns = dst_columns

    def _structure_data(self, transformer, dataset, window_chunk):
        """Structure the data from the raster file.

        Args:
            pyproj_transformer (pyproj.Transformer): A pyproj transformer object.
            dataset (rasterio.io.DatasetReader): A rasterio dataset object.
            window_chunk (list): A list of rasterio.windows.Window objects.

        Returns:
            pandas.DataFrame: A pandas DataFrame object.
        """
        data_df = pd.DataFrame(
            [
                (
                    *utils.calculate_coordinate(
                        transformer,
                        dataset.transform,
                        window.row_off + 0,
                        window.col_off + 0,
                    ),
                    *utils.calculate_coordinate(
                        transformer,
                        dataset.transform,
                        window.row_off + 0,
                        window.col_off + window.width,
                    ),
                    *utils.calculate_coordinate(
                        transformer,
                        dataset.transform,
                        window.row_off + window.height,
                        window.col_off + window.width,
                    ),
                    *utils.calculate_coordinate(
                        transformer,
                        dataset.transform,
                        window.row_off + window.height,
                        window.col_off + 0,
                    ),
                    window.height,
                    window.width,
                    dataset.read(1, window=window).tobytes(),
                )
                for window in window_chunk
            ],
            columns=self.dst_columns,
        )

        return data_df

    def _bigquery_client(self, project=None):  # pragma: no cover
        """Create a BigQuery client.

        Returns:
            google.cloud.bigquery.client.Client: A BigQuery client object.
        """
        return bigquery.Client(project=project)

    def to_bigquery(self, project, dataset, table, chunk_size=100):
        """Upload the raster file to BigQuery.

        Args:
            project (str): The name of the Google Cloud project.
            dataset (str): The name of the dataset.
            table (str): The name of the table.
            chunk_size (int, default = 100): The number of pixels to upload in each \
              chunk.
        """
        try:
            bigquery_client = self._bigquery_client(project)
        except Exception as e:
            raise errors.ClientError(e)

        with rasterio.open(self.file_path) as raster_dataset:
            src_crs = utils.get_dataset_crs(raster_dataset)
            transformer = pyproj.Transformer.from_crs(
                src_crs, self.dst_crs
            )  # compute lat and lon

            windows = [window for _, window in raster_dataset.block_windows()]
            pbar_len = len(range(0, len(windows), chunk_size))
            with tqdm(total=pbar_len) as pbar:
                for window_chunk in utils.chunks(
                    windows, chunk_size
                ):  # tune the number of elem in chunk depending on your RAM
                    data_df = self._structure_data(
                        transformer, raster_dataset, window_chunk
                    )

                    if data_df.size:
                        try:
                            bigquery_client.load_table_from_dataframe(
                                data_df, f"{project}.{dataset}.{table}"
                            )
                        except Exception as e:
                            raise errors.UploadError(e)

                    pbar.update(1)
