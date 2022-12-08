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

At a minimum, this command requires a file path to a local raster file that can be
`read by GDAL`_ and processed with `rasterio`_. It also requires a `GCP project name`_,
a `BigQuery dataset`_, and a `BigQuery table name`_. For example:

.. code-block:: bash

   carto bigquery upload \
     --file_path /path/to/my/raster/file.tif \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table

See the :ref:`cli_details` for a full list of options.

Inspecting a raster file on BigQuery
------------------------------------

You can also use Raster Loader to retrieve information about a raster file stored in a
BigQuery table. This can be useful to make sure a raster file was transferred correctly
or to get information about a raster file's metadata, for example.

To access a raster file in a BigQuery table, use the ``carto bigquery inspect`` command.

At a minimum, this command requires a `GCP project name`_, a `BigQuery dataset`_, and a
`BigQuery table name`_. For example:

.. code-block:: bash

   carto bigquery inspect \
     --project my-gcp-project \
     --dataset my-bigquery-dataset \
     --table my-bigquery-table

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
.. _`GCP project name`: https://cloud.google.com/resource-manager/docs/creating-managing-projects
.. _`BigQuery dataset`: https://cloud.google.com/bigquery/docs/datasets-intro
.. _`BigQuery table name`: https://cloud.google.com/bigquery/docs/tables-intro
