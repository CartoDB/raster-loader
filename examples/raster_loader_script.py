# raster-loader

from raster_loader import RasterLoader

# Authentication
raster_loader = RasterLoader.from_oauth()
# raster_loader = RasterLoader.from_m2m("./carto_credentials.json")

# Get api base url
api_base_url = raster_loader.get_api_base_url()

# Get access token
access_token = raster_loader.get_access_token()

# CARTO Data Warehouse
carto_dw_project, carto_dw_token = raster_loader.get_carto_dw_credentials()
carto_dw_client = raster_loader.get_carto_dw_client()
