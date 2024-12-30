import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import numpy as np


def split_dataset(dataset_path, n):
    with rasterio.open(dataset_path) as src:
        width = src.width
        height = src.height
        tile_width = src.block_shapes[0][1]
        tile_height = src.block_shapes[0][0]

        for i in range(n):
            for j in range(n):
                window = Window(j * tile_width, i * tile_height, tile_width, tile_height)
                transform = src.window_transform(window)
                vrt_options = {
                    'resampling': rasterio.enums.Resampling.nearest,
                    'transform': transform,
                    'width': tile_width,
                    'height': tile_height
                }
                with WarpedVRT(src, **vrt_options) as vrt:
                    # data = vrt.read(window=window)
                    wins = list(vrt.block_windows())
                    print(wins[0], wins[-1])
                #     output_path = f"/tmp/raster/tile_{i}_{j}.vrt"
                #     with rasterio.open(output_path, 'w', driver='VRT', width=tile_width, height=tile_height, count=src.count, dtype=src.dtypes[0], crs=src.crs, transform=transform) as dst:
                #         dst.write(vrt.read(window=window))
                # vrt_path = f"/tmp/raster/tile_{i}_{j}.vrt"
#                     with open(vrt_path, 'w') as vrt_file:
#                         vrt_file.write(f"""<VRTDataset rasterXSize="{tile_width}" rasterYSize="{tile_height}">
#     <SRS>{src.crs.to_wkt()}</SRS>
#     <GeoTransform>{', '.join(map(str, transform.to_gdal()))}</GeoTransform>
#     <VRTRasterBand dataType="{src.dtypes[0]}" band="1">
#         <SimpleSource>
#             <SourceFilename relativeToVRT="0">{dataset_path}</SourceFilename>
#             <SourceBand>1</SourceBand>
#             <SourceProperties RasterXSize="{width}" RasterYSize="{height}" DataType="{src.dtypes[0]}" BlockXSize="{src.block_shapes[0][1]}" BlockYSize="{src.block_shapes[0][0]}"/>
#             <SrcRect xOff="{j * tile_width}" yOff="{i * tile_height}" xSize="{tile_width}" ySize="{tile_height}"/>
#             <DstRect xOff="0" yOff="0" xSize="{tile_width}" ySize="{tile_height}"/>
#         </SimpleSource>
#     </VRTRasterBand>
# </VRTDataset>""")


if __name__ == "__main__":
    dataset_path = "/home/cayetano/Downloads/raster/classification_germany_cog.tif"
    n = 4  # Number of splits along each dimension
    datasets = split_dataset(dataset_path, n)