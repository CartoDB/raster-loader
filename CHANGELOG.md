# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2023-10-30

### Bug Fixes
- Fixed issue in parsing long json decimals (#117)

## [0.3.2] - 2023-09-15

### Enhancements

- Add append option to skip check (#114)

## [0.3.1] - 2023-04-21

### Enhancements
- Store raster nodata value in table metadata (#111)
- Add level to raster metadata (#110)

### Bug Fixes
- Fixed issue in metadata when updating table (#112)

## [0.3.0] - 2023-03-07

### Enhancements
- Create raster tables with geography and metadata (#105)

### Bug Fixes
- Fixed band in field name (#102)
- Dockerfile - avoid installing GDAL twice (#104)

## [0.2.0] - 2023-01-26

### Enhancements
- Updated setup.cfg and readme (#70)
- Bumped wheel from 0.37.1 to 0.38.1 (#63)
- Added a basic docker-compose based dev environment (#80)
- Use quadbin (#72)
- Raise rasterio.errors.CRSError for invalid CRS and Add test error condition (#89)
- Cluster "quadbin raster" table by quadbin (#95)
- Changed the endianess to little endian to accomodate the front end (#97)

### Bug Fixes
- Fixed swapped lon lat (#75)
- Fixed performance regression bug (#77)
- Added small speed hack (#79)

### Documentation
- Added docs badge and readthedocs link to readme (#69)
- Updated contributor guide (#91)
- Updated docs for quadbin (#93)

## [0.1.0] - 2023-01-05

### Added

- raster_loader module
  - rasterio_to_bigquery function
  - bigquery_to_records function
- rasterio_to_bigquery.cli submodule
  - upload command
  - describe command
  - info command
- docs
- tests
- CI/CD with GitHub Actions
