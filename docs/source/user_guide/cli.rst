.. _cli:

Using the Raster Loader CLI
===========================

Most functions of the Raster Loader are accessible through the carto
command-line interface (CLI). To start the CLI, use the ``carto`` command in a
terminal.

Currently, Raster Loader allows you to upload a local raster file to a BigQuery table.
You can also download and inspect a raster file from a BigQuery table.

.. note::

    Accessing BigQuery with Raster Loader requires the ``GOOGLE_APPLICATION_CREDENTIALS``
    environment variable to be set to the path of a JSON file containing your BigQuery
    credentials. See the `GCP documentation`_ for more information.

Uploading to BigQuery
---------------------

To upload a raster file to a BigQuery table, use the ``carto bigquery upload`` command.

Before you can upload a raster file, you need to have set up the following in
BigQuery:

#. A `GCP project`_
#. A `BigQuery dataset`_

You have the option to also set up a `BigQuery table`_ and use this table to upload
your data to. In case you do not specify a table name, Raster Loader will automatically
generate a table name for you and create that table.

At a minimum, the ``carto bigquery upload`` command requires a ``file_path`` to a local
raster file that can be `read by GDAL`_ and processed with `rasterio`_. It also requires
the ``project`` (the GCP project name) and ``dataset`` (the BigQuery dataset name)
parameters. There are also additional parameters, such as ``table`` (BigQuery table
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

.. seealso::
   See the :ref:`cli_details` for a full list of options.

Inspecting a raster file on BigQuery
------------------------------------

You can also use Raster Loader to retrieve information about a raster file stored in a
BigQuery table. This can be useful to make sure a raster file was transferred correctly
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
