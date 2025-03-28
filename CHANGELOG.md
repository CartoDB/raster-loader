# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->

## [0.11.0] 2025-03-20

- feat: add support for databricks (#169) (#feat: add support for databricks (#169)) ([38f6006](https://github.com/CartoDB/raster-loader/commit/38f6006415d642c13c0a38191f183b6af7fa5c8f) by Valentin de la Cruz Barquero).

## [0.10.5] 2025-03-11

### Fixed

- Fix band nodata NaN values (#170) ([a561949](https://github.com/CartoDB/raster-loader/commit/a56194978d531cbe0397088e142357f56b6a0f83) by Juan Ramón).
- fix: matching compressions should be checked when appending rasters (#165) ([0e30d11](https://github.com/CartoDB/raster-loader/commit/0e30d116ba922b6ebc8ab7895eb7371c80d351e6) by Valentin de la Cruz Barquero).

## [0.10.4] 2025-03-07

### Fixed

- Fix band nodata value (#167) ([2bf9bb5](https://github.com/CartoDB/raster-loader/commit/2bf9bb594a4e3eabcf5c7245cae8ad22f6cee156) by Juan Ramón).

## [0.10.3] 2025-01-17

<small>[Compare with latest](https://github.com/CartoDB/raster-loader/compare/v0.10.2...v0.10.3)</small>

### Fixed

- fix: most_common_approx duplicated counts definition ([1415d3d](https://github.com/CartoDB/raster-loader/commit/1415d3d2731bd0a1f35054846c598b7f3a839115) by cayetanobv).

### Added

- add new option: compression level (#162) ([fc45f68](https://github.com/CartoDB/raster-loader/commit/fc45f68415929b780d69311e51cef7e954ae87ac) by Cayetano Benavent).

## [0.10.2] 2025-01-14

<small>[Compare with latest](https://github.com/CartoDB/raster-loader/compare/v0.10.1...HEAD)</small>

### Fixed

- Fix: OverflowError error when casting approx sum to integer ([46cab53](https://github.com/CartoDB/raster-loader/commit/46cab53bbf71a86a7df784922956eb03f9dbb327) by Roberto Antolín).
- Fix: Compute approximate most common negative values ([f9f5ff5](https://github.com/CartoDB/raster-loader/commit/f9f5ff5010b1aea0d13afbea6d1869d4094fa7d7) by Roberto Antolín).

## [0.10.1] 2025-01-13

<small>[Compare with latest](https://github.com/CartoDB/raster-loader/compare/57d55999704fb003da2947db65d5617e27c5c104...HEAD)</small>

### Added

-   Snowflake key pair authentication support (#158)

<!-- insertion marker -->

## [0.10.0] 2025-01-10

<small>[Compare with latest](https://github.com/CartoDB/raster-loader/compare/v0.9.2...HEAD)</small>

### Added

-   add new option: compression (#160) ([c46dd51](https://github.com/CartoDB/raster-loader/commit/c46dd51bf53847e21de7550e5b826be1a6cda3eb) by Cayetano Benavent).

## [0.9.2] 2024-12-11

<small>[Compare with latest](https://github.com/CartoDB/raster-loader/compare/v0.9.1...HEAD)</small>

### Added

-   Add: Compute top values only for integer bands ([6c10cc0](https://github.com/CartoDB/raster-loader/commit/6c10cc025f5691f7841beee560437fb591bddfe9) by Roberto Antolín).

### Fixed

-   Fix: Tackle degenerate case of stdev computation ([b112c80](https://github.com/CartoDB/raster-loader/commit/b112c80be7d7c1adfd08f651b43dc591fd54a2ef) by Roberto Antolín).
-   Fix: Get count stats from shape of raster band ([c066a30](https://github.com/CartoDB/raster-loader/commit/c066a307ee116598c54ea4871d563f79deebad0b) by Roberto Antolín).
-   Fix: Raise error when 0 non-masked samples due to sparse rasters ([dfd89ae](https://github.com/CartoDB/raster-loader/commit/dfd89aef27726a3217843022769600315d8e5b6f) by Roberto Antolín).

### Changed

-   Change '--all_stats' flag to '--basic_stats' ([2cb89cc](https://github.com/CartoDB/raster-loader/pull/156/commits/2cb89cca30eb15189c876760c026074e262cc10f) by Roberto Antolín).

## [0.9.1] 2024-11-26

### Fixed

-   fix: changed default no data for byte data type ([06ad98f](https://github.com/CartoDB/raster-loader/commit/06ad98f3723c44ce847f475887cdca084c6ca571) by volaya).

## [0.9.0](https://github.com/CartoDB/raster-loader/releases/tag/v0.9.0) - 2024-11-04

<small>[Compare with first commit](https://github.com/CartoDB/raster-loader/compare/167c3d69359f9b3abb49a3c1c5aa6249f76c0992...v0.9.0)</small>

## [0.9.0](https://github.com/CartoDB/raster-loader/releases/tag/0.9.0) - 2024-11-04

### Added

-   Added exact stats (#153)

## [0.8.2] - 2024-10-07

### Bug Fixes

-   Fix casting in quantiles (#151)

## [0.8.1] - 2024-09-24

### Bug Fixes

-   Fix stats for unmasked rasters

## [0.8.0] - 2024-09-17

### Added

-   Add metadata for Builder Rasters (#147)

## [0.7.1] - 2024-07-05

### Bug Fixes

-   fix: support for Python 3.9 / NumPy 2.0 (#145)

## [0.7.0] - 2024-07-02

### Added

-   Support raster overviews (#140)

### Enhancements

-   increase chunk-size to 10000 (#142)

### Bug Fixes

-   fix: make the gdalwarp examples consistent (#143)

## [0.6.1] - 2024-04-02

### Enhancements

-   Add a argument to skip interactive question on upload failure (#138)

### Bug Fixes

-   fix: shapely.wkt import (#136)
-   fix: update pip commands to make it compatible with zsh (#137)

## [0.6.0] - 2024-03-25

### Enhancements

-   Add labels to BQ uploaded tables (#131)
-   Support input URLs and more connection credential types (#129)

### Bug Fixes

-   fixed using raster files with block size other than default value (#130)
-   fix: error when bigquery dependencies not installed (#133)

## [0.5.0] - 2024-01-05

### Enhancements

-   Add support for snowflake (#127)

## [0.4.0] - 2023-12-21

### Enhancements

-   Update raster-loader to generate new Raster and Metadata table format (#116)
-   Add pixel_resolution, rename block_resolution (#123)

### Bug Fixes

-   fix: metadata field pixel_resolution as an integer and not allow zooms over 26 (#124, #125)

## [0.3.3] - 2023-10-30

### Bug Fixes

-   Fixed issue in parsing long json decimals (#117)

## [0.3.2] - 2023-09-15

### Enhancements

-   Add append option to skip check (#114)

## [0.3.1] - 2023-04-21

### Enhancements

-   Store raster nodata value in table metadata (#111)
-   Add level to raster metadata (#110)

### Bug Fixes

-   Fixed issue in metadata when updating table (#112)

## [0.3.0] - 2023-03-07

### Enhancements

-   Create raster tables with geography and metadata (#105)

### Bug Fixes

-   Fixed band in field name (#102)
-   Dockerfile - avoid installing GDAL twice (#104)

## [0.2.0] - 2023-01-26

### Enhancements

-   Updated setup.cfg and readme (#70)
-   Bumped wheel from 0.37.1 to 0.38.1 (#63)
-   Added a basic docker-compose based dev environment (#80)
-   Use quadbin (#72)
-   Raise rasterio.errors.CRSError for invalid CRS and Add test error condition (#89)
-   Cluster "quadbin raster" table by quadbin (#95)
-   Changed the endianess to little endian to accomodate the front end (#97)

### Bug Fixes

-   Fixed swapped lon lat (#75)
-   Fixed performance regression bug (#77)
-   Added small speed hack (#79)

### Documentation

-   Added docs badge and readthedocs link to readme (#69)
-   Updated contributor guide (#91)
-   Updated docs for quadbin (#93)

## [0.1.0] - 2023-01-05

### Added

-   raster_loader module
    -   rasterio_to_bigquery function
    -   bigquery_to_records function
-   rasterio_to_bigquery.cli submodule
    -   upload command
    -   describe command
    -   info command
-   docs
-   tests
-   CI/CD with GitHub Actions
