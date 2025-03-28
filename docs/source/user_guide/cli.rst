.. _cli:

Using the Raster Loader CLI
===========================

Most functions of the Raster Loader are accessible through the carto
command-line interface (CLI). To start the CLI, use the ``carto`` command in a
terminal.

Currently, Raster Loader allows you to upload a local raster file to a BigQuery, Snowflake, or Databricks table.
You can also download and inspect a raster file from a BigQuery, Snowflake, or Databricks table.


Using the Raster Loader with BigQuery
-----------------------------------------

Before you can upload a raster file, you need to have set up the following in
BigQuery:

#. A `GCP project`_
#. A `BigQuery dataset`_

To use the bigquery utilities, use the ``carto bigquery`` command. This command has
several subcommands, which are described below.

.. note::

    Accessing BigQuery with Raster Loader requires the ``GOOGLE_APPLICATION_CREDENTIALS``
    environment variable to be set to the path of a JSON file containing your BigQuery
    credentials. See the `GCP documentation`_ for more information.

Using the Raster Loader with Snowflake
-----------------------------------------

Before you can upload a raster file, you need to have set up the following in
Snowflake:

#. A Snowflake account
#. A Snowflake database
#. A Snowflake schema

To use the snowflake utilities, use the ``carto snowflake`` command. This command has
several subcommands, which are described below.

Using the Raster Loader with Databricks
-----------------------------------------

Before you can upload a raster file, you need to have set up the following in
Databricks:

#. A `Databricks server hostname`_
#. A `Databricks cluster id`_
#. A `Databricks token`_

To use the databricks utilities, use the ``carto databricks`` command. This command has
several subcommands, which are described below.

Uploading a raster layer
------------------------

To upload a raster file, use the ``carto [bigquery|snowflake|databricks] upload`` command.

The input raster must be a ``GoogleMapsCompatible`` raster. You can make your raster compatible
by converting it with the following GDAL command:

.. code-block:: bash

   gdalwarp -of COG -co TILING_SCHEME=GoogleMapsCompatible -co COMPRESS=DEFLATE -co OVERVIEWS=IGNORE_EXISTING -co ADD_ALPHA=NO -co RESAMPLING=NEAREST -co BLOCKSIZE=512 <input_raster>.tif <output_raster>.tif

You have the option to also set up a table in your provider and use this table to upload
your data to. In case you do not specify a table name, Raster Loader will automatically
generate a table name for you and create that table.

At a minimum, the ``carto upload`` command requires a ``file_path`` to a local
raster file that can be `read by GDAL`_ and processed with `rasterio`_. It also requires
the ``project`` (the GCP project name) and ``dataset`` (the BigQuery dataset name)
parameters in the case of Bigquery; the ``database`` and ``schema`` parameters in the
case of Snowflake; or the ``catalog`` and ``schema`` parameters in the case of Databricks.

There are also additional parameters, such as ``table`` (table
name) and ``overwrite`` (to overwrite existing data). For example:

.. code-block:: bash

   carto bigquery upload \
     --file_path /path/to/my/raster/file.tif \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table \
     --overwrite

This command uploads the TIFF file from ``/path/to/my/raster/file.tif`` to a BigQuery
project named ``my-gcp-project``, a dataset named ``my-bigquery-dataset``, and a table
named ``my-bigquery-table``. If the table already contains data, this data will be
overwritten because the ``--overwrite`` flag is set.

The same operation, performed with Snowflake, would be:

.. code-block:: bash

   carto snowflake upload \
     --file_path /path/to/my/raster/file.tif \
     --database my-snowflake-database \
     --schema my-snowflake-schema \
     --table my-snowflake-table \
     --account my-snowflake-account \
     --username my-snowflake-user \
     --password my-snowflake-password \
     --overwrite

Authentication parameters are explicitly required in this case for Snowflake, since they
are not set up in the environment.

The same operation, performed with Databricks, would be:

.. code-block:: bash

   carto databricks upload \
     --file_path /path/to/my/raster/file.tif \
     --catalog my-databricks-catalog \
     --schema my-databricks-schema \
     --table my-databricks-table \
     --server-hostname my-databricks-server-hostname \
     --cluster-id my-databricks-cluster-id \
     --token my-databricks-token \
     --overwrite

Authentication parameters are also explicitly required in the case of Databricks, since they
are not set up in the environment.

If no band is specified, the first band of the raster will be uploaded. If the
``--band`` flag is set, the specified band will be uploaded. For example, the following
command uploads the second band of the raster:

.. code-block:: bash

   carto bigquery upload \
     --file_path /path/to/my/raster/file.tif \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table \
     --band 2

Band names can be specified with the ``--band_name`` flag. For example, the following
command uploads the ``red`` band of the raster:

.. code-block:: bash

   carto bigquery upload \
     --file_path /path/to/my/raster/file.tif \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table \
     --band 2 \
     --band_name red

If the raster contains multiple bands, you can upload multiple bands at once by
specifying a list of bands. For example, the following command uploads the first and
second bands of the raster:

.. code-block:: bash

   carto bigquery upload \
     --file_path /path/to/my/raster/file.tif \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table \
     --band 1 \
     --band 2

Or, with band names:

.. code-block:: bash

   carto bigquery upload \
     --file_path /path/to/my/raster/file.tif \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table \
     --band 1 \
     --band 2 \
     --band_name red \
     --band_name green

You can enable compression of the band data using the ``--compress`` flag. This uses gzip compression which can significantly reduce storage size. By default, it uses compression level 6, which provides a good balance between compression ratio and performance. You can adjust this using the ``--compression-level`` parameter (values from 1 to 9, where 1 is fastest but least compressed, and 9 gives maximum compression):

.. code-block:: bash

   carto bigquery upload \
     --file_path /path/to/my/raster/file.tif \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table \
     --compress \
     --compression-level 3

The same works for Snowflake:

.. code-block:: bash

   carto snowflake upload \
     --file_path /path/to/my/raster/file.tif \
     --database my-snowflake-database \
     --schema my-snowflake-schema \
     --table my-snowflake-table \
     --account my-snowflake-account \
     --username my-snowflake-user \
     --password my-snowflake-password \
     --compress \
     --compression-level 3

And for Databricks:

.. code-block:: bash

   carto databricks upload \
     --file_path /path/to/my/raster/file.tif \
     --catalog my-databricks-catalog \
     --schema my-databricks-schema \
     --table my-databricks-table \
     --server-hostname my-databricks-server-hostname \
     --cluster-id my-databricks-cluster-id \
     --token my-databricks-token \
     --compress \
     --compression-level 3

.. seealso::
   See the :ref:`cli_details` for a full list of options.

For large raster files, you can use the ``--chunk_size`` flag to specify the number of
rows to upload at once, and preventing BigQuery from showing you an exception like the following,
due to excessive operations in the destination table:

```
Exceeded rate limits: too many table update operations for this table. For more information, see https://cloud.google.com/bigquery/troubleshooting-errors
```

The default chunk size is 10000 rows.

For example, the following command uploads the raster in chunks
of 20000 rows:

.. code-block:: bash

   carto bigquery upload \
     --file_path /path/to/my/raster/file.tif \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table \
     --chunk_size 20000

For large raster files in Databricks, you might get the following error:

```
Error uploading records: Cannot convert pyarrow.lib.ChunkedArray to pyarrow.lib.Array
```

This error is due to the size of the raster file being too large to be uploaded in one go,
and the default chunk size being too large. In this case, you can try to reduce the number of
rows to upload at once by using the ``--chunk_size`` flag. 

Inspecting a raster file
------------------------------------

You can also use Raster Loader to retrieve information about a raster file stored in a
BigQuery, Snowflake, or Databricks table. This can be useful to make sure a raster file was transferred correctly
or to get information about a raster file's metadata, for example.

To access a raster file in a BigQuery table, use the ``carto bigquery describe`` command.

At a minimum, this command requires a `GCP project name <GCP project>`_, a
`BigQuery dataset name <BigQuery dataset>`_, and a
`BigQuery table name <BigQuery table>`_. For example:

.. code-block:: bash

   carto bigquery describe \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table

The same operation, performed with Snowflake, would be:

.. code-block:: bash

   carto snowflake describe \
     --database my-snowflake-database \
     --schema my-snowflake-schema \
     --table my-snowflake-table \
     --account my-snowflake-account \
     --username my-snowflake-user \
     --password my-snowflake-password

Authentication parameters are explicitly required in this case for Snowflake, since they
are not set up in the environment.

The same operation, performed with Databricks, would be:

.. code-block:: bash

   carto databricks describe \
     --catalog my-databricks-catalog \
     --schema my-databricks-schema \
     --table my-databricks-table \
     --server-hostname my-databricks-server-hostname \
     --cluster-id my-databricks-cluster-id \
     --token my-databricks-token

Authentication parameters are also explicitly required in the case of Databricks, since they
are not set up in the environment.

.. seealso::
   See the :ref:`cli_details` for a full list of options.

.. _cli_details:

CLI details
-----------

The following is a detailed overview of all of the CLI's subcommands and options:

.. click:: raster_loader.cli:main
   :prog: carto
   :nested: full

.. _`GCP documentation`: https://cloud.google.com/docs/authentication/provide-credentials-adc#local-key
.. _`read by GDAL`: https://gdal.org/drivers/raster/index.html
.. _`rasterio`: https://rasterio.readthedocs.io/en/latest/
.. _`GCP project`: https://cloud.google.com/resource-manager/docs/creating-managing-projects
.. _`BigQuery dataset`: https://cloud.google.com/bigquery/docs/datasets-intro
.. _`BigQuery table`: https://cloud.google.com/bigquery/docs/tables-intro
.. _`Databricks server hostname`: https://docs.databricks.com/aws/en/integrations/compute-details
.. _`Databricks cluster id`: https://learn.microsoft.com/en-us/azure/databricks/workspace/workspace-details#cluster-url
.. _`Databricks token`: https://docs.databricks.com/aws/en/dev-tools/auth/pat