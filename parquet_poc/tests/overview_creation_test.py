import rasterio
import rio_cogeo
import math
import os
import shutil


DEFAULT_COG_BLOCK_SIZE = 256


def get_resolution_and_block_sizes(raster_dataset, zoom_level):
    # assuming all windows have the same dimensions
    a_window = next(raster_dataset.block_windows())
    block_width = a_window[1].width
    block_height = a_window[1].height
    resolution = int(
        zoom_level
        - math.log(
            block_width / DEFAULT_COG_BLOCK_SIZE * block_height / DEFAULT_COG_BLOCK_SIZE,
            4,
        )
    )
    return block_width, block_height, resolution


def main():
    # file_path = "/home/cayetano/Downloads/raster/corelogic/202112geotiffs/cog/20211201_forensic_wind_banded_cog.tif"
    file_path = "/home/cayetano/Downloads/raster/classification_germany_cog.tif"
    file_cog_info = rio_cogeo.cog_info(file_path)
    base_zoom_level = file_cog_info["GEO"]["MaxZoom"]
    print(f"Base zoom level: {base_zoom_level}")

    output_folder = 'overviews_output'

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder)

    with rasterio.open(file_path, 'r') as raster_src:
        print(get_resolution_and_block_sizes(raster_src, base_zoom_level))
        print(raster_src.profile)
        overviews = raster_src.overviews(1)
        for idx, ov in enumerate(overviews):
            print('---------------------')
            with rasterio.open(file_path, 'r', overview_level=idx) as src_ov:
                # windows_for_array = list(src_ov.block_windows())
                # raster_transform = src_ov.transform
                # print(raster_transform)
                print(idx, ov, src_ov.shape, base_zoom_level - (idx + 1))
                # print(rasterio.windows.bounds(rasterio.windows.union([win for _, win in windows_for_array]), raster_transform))
                profile = src_ov.profile
                profile["driver"] = "COG"
                profile["tiling_scheme"] = "GoogleMapsCompatible"
                print(profile)
                with rasterio.MemoryFile() as memfile:
                    with memfile.open(**profile) as dst_dst:
                        # dst_dst.write(src_ov.read(1), 1)
                        # print(dst_dst.profile)
                        print(dst_dst.transform)
                        print(dst_dst.bounds)
                        # print(rasterio.windows.bounds(rasterio.windows.union([win for _, win in list(dst_dst.block_windows())]), dst_dst.transform))
                    print('1111111111')
                print('22222222222222')
                    # with memfile.open() as dst_dst_r:
                    #     # arr = dst_dst_r.read(1)
                    #     print(dst_dst_r.transform)
                    #     print(dst_dst_r.bounds)
                    #     print(dst_dst_r.shape)
                
                # output_file = os.path.join(output_folder, f"overview_{idx}.tif")
                # with rasterio.open(output_file, 'w', **profile) as dst_dst_f:
                #     print(f'Writing to {output_file}')
                #     dst_dst_f.write(src_ov.read(1), 1)
                #     print(rasterio.windows.bounds(rasterio.windows.union([win for _, win in list(dst_dst_f.block_windows())]), dst_dst_f.transform))
                #     # print(list(dst_dst_f.block_windows()))
                #     print(dst_dst_f.shape)


if __name__ == "__main__":
    main()
