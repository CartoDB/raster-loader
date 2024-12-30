--Compressed raster: cartobq.cayetanobv_raster.raster_parquet_test_20241219174716_644b23ce3f6c4f168b62af3849c01b31
--1. Uncompress the data
drop table if exists cartobq.cayetanobv_raster.raster_parquet_test_20241219174716_644b23ce3f6c4f168b62af3849c01b31_inflat;
create table cartobq.cayetanobv_raster.raster_parquet_test_20241219174716_644b23ce3f6c4f168b62af3849c01b31_inflat
cluster by block
as
select block, cast(cayetanobv.inflate_data(band_1) as bytes) as band_1, null as metadata
from cartobq.cayetanobv_raster.raster_parquet_test_20241219174716_644b23ce3f6c4f168b62af3849c01b31
where block !=0
union all
select block, null as band_1, metadata 
from cartobq.cayetanobv_raster.raster_parquet_test_20241219174716_644b23ce3f6c4f168b62af3849c01b31
where block = 0;

--2. Extract raster pixel values from blocks
DECLARE _metadata JSON;
SET _metadata = (
    SELECT PARSE_JSON(metadata, wide_number_mode=>'round')
    FROM `cartobq.cayetanobv_raster.raster_parquet_test_20241219174716_644b23ce3f6c4f168b62af3849c01b31_inflat`
    WHERE block=0
);

WITH __data AS (
    select block, band_1
    from cartobq.cayetanobv_raster.raster_parquet_test_20241219174716_644b23ce3f6c4f168b62af3849c01b31_inflat
    where block!=0
),

__blocks AS (
    SELECT dt.*, `carto-un.carto.QUADBIN_TOZXY`(block) AS __block_tile
    FROM __data dt
),
__offsets AS (
    SELECT
        __pixel_x + __pixel_y * int64(_metadata.block_width) AS __pixel_offset , __pixel_x, __pixel_y
    FROM
        UNNEST(GENERATE_ARRAY(0, int64(_metadata.block_width) - 1)) AS __pixel_y,
        UNNEST(GENERATE_ARRAY(0, int64(_metadata.block_width) - 1)) AS __pixel_x
),
__bands AS (
    SELECT 
        `carto-un.carto.__FSEEK_UINT8`(band_1, __pixel_offset) AS band_1
        , `carto-un.carto.QUADBIN_FROMZXY`(
            int64(_metadata.pixel_resolution),
            __pixel_x + __block_tile.x * int64(_metadata.block_width),
            __pixel_y + __block_tile.y * int64(_metadata.block_width)
        ) AS __pixel
        , __blocks.block AS block
    FROM __blocks, __offsets
),
-- __values AS (
--     SELECT IF(band_1 = float64(_metadata.nodata), NULL, band_1) AS band_1 , __pixel, block
--     FROM __bands
-- ),
__results AS (
    SELECT __pixel AS pixel, band_1, block
    FROM __bands
)
SELECT * FROM __results; 