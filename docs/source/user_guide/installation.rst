.. _installation:

Installing Raster Loader
========================

Raster Loader is available on PyPI_ and can be installed with pip_:

.. code-block:: bash

   pip install -U raster-loader

To install from source:

.. code-block:: bash

   git clone https://github.com/cartodb/raster-loader
   cd raster-loader
   pip install -U .

.. tip::

   In most cases, it is recommended to install Raster Loader in a virtual environment.
   Use venv_ to create and manage your virtual environment.

The above will install the dependencies required to work with the different cloud providers (BigQuery, Snowflake, Databricks). In case you only want to work with one of them, you can install the
dependencies for each of them separately:

.. code-block:: bash

   pip install -U raster-loader"[bigquery]"
   pip install -U raster-loader"[snowflake]"
   pip install -U raster-loader"[databricks]"

For the case of Databricks, additionally you will need to install the databricks-connect_ package corresponding to your Databricks Runtime Version. For example, if your cluster uses DBR 15.1, install:

.. code-block:: bash

   pip install databricks-connect==15.1

You can find your cluster's DBR version in the Databricks UI under Compute > Your Cluster > Configuration > Databricks Runtime version.
Or you can run the following SQL query from your cluster:

.. code-block:: sql

   SELECT current_version();

After installing the Raster Loader package, you will have access to the
:ref:`carto CLI <cli>`. To make sure the installation was successful, run the
following command in your terminal:

.. code-block:: bash

   carto info

This command should print some basic system information, including the version of Raster
Loader installed on your system. For example:

.. code-block:: bash

    Raster Loader version: 0.1
    Python version: 3.11.0 | packaged by conda-forge |
    Platform: Linux-5.10.16.3-microsoft-standard-WSL2-x86_64-with-glibc2.35
    System version: Linux 5.10.16.3-microsoft-standard-WSL2
    Machine: x86_64
    Processor: x86_64
    Architecture: 64bit

.. _PyPI: https://pypi.org/project/raster-loader/
.. _pip: https://pip.pypa.io/en/stable/
.. _venv: https://docs.python.org/3/library/venv.html
.. _databricks-connect: https://pypi.org/project/databricks-connect/