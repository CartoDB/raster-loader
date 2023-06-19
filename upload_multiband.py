from raster_loader import get_bands_number, rasterio_to_bigquery
import multiprocessing

# rasterio_to_bigquery(
#     file_path = 'gs://bq_vdelacruz/raster_test/fuji.tif',
#     project_id = 'cartodb-data-engineering-team',
#     dataset_id = 'vdelacruz_carto',
#     table_id = 'raster_fuji',
#     overwrite = True,
# )

# rasterio_to_bigquery(
#     file_path = '/Users/vdelacruz/Desktop/Repositories/raster-loader/raster_loader/tests/fixtures/fuji.tif',
#     project_id = 'cartodb-data-engineering-team',
#     dataset_id = 'vdelacruz_carto',
#     table_id = 'raster_fuji',
#     overwrite = True,
# )

# print_band_information('/Users/vdelacruz/Desktop/Repositories/raster-loader/HARV_RGB_Ortho.tif')

# rasterio_to_bigquery(
#     file_path = '/Users/vdelacruz/Desktop/Repositories/raster-loader/HARV_RGB_Ortho.tif',
#     project_id = 'cartodb-data-engineering-team',
#     dataset_id = 'vdelacruz_carto',
#     table_id = 'harv_rgb_ortho',
#     overwrite = True,
#     band=2,
# )


## USER INPUT
table_prefix = 'cartodb-data-engineering-team.vdelacruz_carto.harv_rgb_ortho'

# OPTIONAL bands to include
bands = [1, 2, 3]

project_id, dataset_id, table_id = table_prefix.split('.')

file_path = '/Users/vdelacruz/Desktop/Repositories/raster-loader/HARV_RGB_Ortho.tif'
bands_number = get_bands_number(file_path)

if True in [band < 1 or band > bands_number for band in bands]:
    raise 'Band number out of range'

# for band in bands:
#     rasterio_to_bigquery(
#         file_path = file_path,
#         project_id = project_id,
#         dataset_id = dataset_id,
#         table_id = f'{table_id}_band_{band}',
#         band=band,
#         overwrite = True,
#     )

def rasterio_to_bigquery_wrapper(band):
    rasterio_to_bigquery(
        file_path = file_path,
        project_id = project_id,
        dataset_id = dataset_id,
        table_id = f'{table_id}_band_{band}',
        band=band,
        overwrite = True,
    )

if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        pool.map(rasterio_to_bigquery_wrapper, bands)
    pool.close()
