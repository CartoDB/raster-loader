.. _python:

Usage with Python projects
==========================

After installing Raster Loader, you can use it in your Python project.

First, import the corresponding connection class from the ``raster_loader`` package.

For BigQuery, use ``BigQueryConnection``:

.. code-block:: python

   from raster_loader import BigQueryConnection

For Snowflake, use ``SnowflakeConnection``:

.. code-block:: python

   from raster_loader import SnowflakeConnection

For Databricks, use ``DatabricksConnection``:

.. code-block:: python

   from raster_loader import DatabricksConnection

Then, create a connection object with the appropriate parameters.

For BigQuery:

.. code-block:: python

    connection = BigQueryConnection('my-project')

.. note::

    Accessing BigQuery with Raster Loader requires the ``GOOGLE_APPLICATION_CREDENTIALS``
    environment variable to be set to the path of a JSON file containing your BigQuery
    credentials. See the `GCP documentation`_ for more information.

For Snowflake:

.. code-block:: python

    connection = SnowflakeConnection('my-user', 'my-password', 'my-account', 'my-database', 'my-schema')

For Databricks:

.. code-block:: python

    connection = DatabricksConnection('my-server-hostname', 'my-token', 'my-cluster-id')

Uploading a raster file
-----------------------------------

To upload a raster file, use the ``upload_raster`` function


For example:

.. code-block:: python

    connection.upload_raster(
        file_path = 'path/to/raster.tif',
        fqn = 'database.schema.tablename',
    )

This function returns `True` if the upload was successful.

The input raster must be a ``GoogleMapsCompatible`` raster. You can make your raster compatible
by converting it with the following GDAL command:

.. code-block:: bash

   gdalwarp -of COG -co TILING_SCHEME=GoogleMapsCompatible -co COMPRESS=DEFLATE -co OVERVIEWS=IGNORE_EXISTINGNONE -co ADD_ALPHA=NO -co RESAMPLING=NEAREST -co BLOCKSIZE=512 <input_raster>.tif <output_raster>.tif

Inspecting a raster file
------------------------

You can also access and inspect a raster file located in a BigQuery or Snowflake table using the
:func:`get_records` function.

For example:

.. code-block:: python

    records = connection.get_records(
        fqn = 'database.schema.tablename',
    )

This function returns a DataFrame with some samples from the raster table
(10 rows by default).

.. seealso::
    See the :ref:`api_reference` for more details.

.. _`GCP documentation`: https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key

To enable compression of the band data, which can significantly reduce storage size, use the ``compress`` parameter:

.. code-block:: python

    connection.upload_raster(
        file_path = 'path/to/raster.tif',
        fqn = 'database.schema.tablename',
        compress = True,  # Enable gzip compression of band data
        compression_level = 3  # Optional: Set compression level (1-9, default=6)
    )

The compression information will be stored in the metadata of the table, and the data will be automatically decompressed when reading it back.
