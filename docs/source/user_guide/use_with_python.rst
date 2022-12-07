.. _python:

Using in Python projects
========================

After installing Raster Loader, you can import the package to your Python project. For
example:

.. code-block:: python

   from raster_loader import rasterio_to_bigquery

Currently, Raster Loader allows you to upload a local raster file to a BigQuery table
using the :func:`~raster_loader.rasterio_to_bigquery` function.

For example:

.. code-block:: python

    rasterio_to_bigquery(
        file_path = 'path/to/raster.tif',
        project_id = 'my-project',
        dataset_id = 'my_dataset',
        table_id = 'my_table',
    )

You can also access and inspect a raster file from a BigQuery table using the
:func:`~raster_loader.bigquery_to_records` function.

For example:

.. code-block:: python

    records_df = bigquery_to_records(
        project_id = 'my-project',
        dataset_id = 'my_dataset',
        table_id = 'my_table',
    )

See the :ref:`api_reference` for more details.
