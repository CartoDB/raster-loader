.. _python:

Usage with Python projects
==========================

After installing Raster Loader, you can use it in your Python project.

First, import the corresponding connection class from the ``raster_loader`` package.
For Snowflake, use ``SnowflakeConnection``:

.. code-block:: python

   from raster_loader import SnowflakeConnection

For BigQuery, use ``BigQueryConnection``:

.. code-block:: python

   from raster_loader import BigQueryConnection

Then, create a connection object with the appropriate parameters.

For Snowflake:

.. code-block:: python

    connection = SnowflakeConnection('my-user', 'my-password', 'my-account', 'my-database', 'my-schema')

For BigQuery:

.. code-block:: python

    connection = BigQueryConnection('my-project')

.. note::

    Accessing BigQuery with Raster Loader requires the ``GOOGLE_APPLICATION_CREDENTIALS``
    environment variable to be set to the path of a JSON file containing your BigQuery
    credentials. See the `GCP documentation`_ for more information.

Uploading a raster file to BigQuery
-----------------------------------

To upload a raster file, use the ``upload_raster`` function


For example:

.. code-block:: python

    connector.upload_raster(
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

    records = connector.get_records(
        fqn = 'database.schema.tablename',
    )

This function returns a DataFrame with some samples from the raster table on BigQuery
(10 rows by default).

.. seealso::
    See the :ref:`api_reference` for more details.

.. _`GCP documentation`: https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key
