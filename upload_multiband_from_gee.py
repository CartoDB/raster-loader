from raster_loader import get_bands_number, rasterio_to_bigquery
import multiprocessing

INTERMEDIATE_BUCKET = """
https://console.cloud.google.com/storage/browser/carto-hackathon-2023-bqcarto;tab=objects?forceOnBucketsSortingFiltering=true&project=bqcarto/
"""

## USER INPUT
gee_image = 'LANDSAT/LC08/C01/T1_TOA/LC08_123032_20140515'
table_id = os.path.basename(gee_image)

table_prefix = f'cartodb-data-engineering-team.vdelacruz_carto.{table_id}'
project_id, dataset_id, table_id = table_prefix.split('.')

# OPTIONAL bands to include
bands = ['B4', 'B3', 'B2']

def gee_to_bigquery_wrapper(band):
    table_name = f'{table_id}_band_{band}'
    gee_to_bucket(
        gee_image,
        None,
        INTERMEDIATE_BUCKET,
        table_name,
    )

    bucket_ref = f'{INTERMEDIATE_BUCKET}table_name'

    rasterio_to_bigquery(
        file_path = bucket_ref,
        project_id = project_id,
        dataset_id = dataset_id,
        table_id = table_name,
        band=band,
        overwrite = True,
    )

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        pool.map(gee_to_bigquery_wrapper, bands)
    pool.close()
