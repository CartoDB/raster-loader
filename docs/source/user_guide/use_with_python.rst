.. _python:

Usage with Python projects
==========================

After installing Raster Loader, you can import the package into your Python project. For
example:

.. code-block:: python

   from raster_loader import rasterio_to_bigquery, bigquery_to_records

Uploading a raster file to BigQuery
-----------------------------------

Currently, Raster Loader allows you to upload a local raster file to an existing
BigQuery table using the :func:`~raster_loader.rasterio_to_bigquery` function.

.. note::

    Accessing BigQuery with Raster Loader requires the ``GOOGLE_APPLICATION_CREDENTIALS``
    environment variable to be set to the path of a JSON file containing your BigQuery
    credentials. See the `GCP documentation`_ for more information.

For example:

.. code-block:: python

    rasterio_to_bigquery(
        file_path = 'path/to/raster.tif',
        project_id = 'my-project',
        dataset_id = 'my_dataset',
        table_id = 'my_table',
    )

This function returns `True` if the upload was successful.

Inspecting a raster file on BigQuery
------------------------------------

You can also access and inspect a raster file located in a BigQuery table using the
:func:`~raster_loader.bigquery_to_records` function.

For example:

.. code-block:: python

    records_df = bigquery_to_records(
        project_id = 'my-project',
        dataset_id = 'my_dataset',
        table_id = 'my_table',
    )

This function returns a DataFrame with some samples from the raster table on BigQuery
(10 rows by default).

.. seealso::
    See the :ref:`api_reference` for more details.

.. _`GCP documentation`: https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key
