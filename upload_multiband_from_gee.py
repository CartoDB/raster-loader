import os
from raster_loader import rasterio_to_bigquery, gee_to_bucket_wrapper
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
roi = '''{"type":"FeatureCollection","features":[{"type":"Feature","properties":{},"geometry":{"coordinates":[[[-12.765280594184702,46.21566957897852],[-11.956614277143075,39.17436777332358],[-3.235950496052567,38.09334028690674],[-1.2584258877862737,40.94195324624033],[-2.381163412646032,43.33057722262623],[-7.925228669539251,45.765725946727],[-12.765280594184702,46.21566957897852]]],"type":"Polygon"}}]}'''


def gee_to_bigquery_wrapper(band):
    table_name = f'{table_id}_band_{band}'
    gee_to_bucket_wrapper(
        gee_image,
        band,
        roi,
        INTERMEDIATE_BUCKET,
        table_name,
    )

    bucket_ref = f'{INTERMEDIATE_BUCKET}{table_name}'

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
