from threading import Timer
from functools import partial

from raster_loader.io.bigquery import BigQueryConnection
from raster_loader.io.snowflake import SnowflakeConnection
from raster_loader.io.databricks import DatabricksConnection


def bigquery_client(load_error=False):
    class BigQueryClient:
        def __init__(self, load_error):
            self.load_error = load_error

        def load_table_from_dataframe(self, *args, **kwargs):
            if load_error:  # pragma: no cover
                raise Exception

            class job(object):
                def result():
                    return True

                def add_done_callback(callback):
                    # need to simulate async behavior
                    # simulating calling callback after chunk download
                    # is completed
                    Timer(0.2, partial(lambda: callback(job))).start()

            return job

        def query(self, query):
            class job(object):
                def result():
                    return True

            return job

        def create_table(self, table):
            return True

    return BigQueryClient(load_error=load_error)


class MockBigQueryConnection(BigQueryConnection):
    def __init__(self, *args, **kwargs):
        self.client = bigquery_client()


def snowflake_client(load_error=False):
    class SnowflakeClient:
        def __init__(self, load_error):
            self.load_error = load_error

        def cursor(self):
            return self

        def execute(self, *args, **kwargs):
            return self

        def fetchall(self):
            return [[1]]

        def fetchone(self):
            return [1]

        def close(self):
            return True

        def _log_telemetry_job_data(self, *args, **kwargs):
            return True

    return SnowflakeClient(load_error=load_error)


class MockSnowflakeConnection(SnowflakeConnection):
    def __init__(self, *args, **kwargs):
        self.client = snowflake_client()


def databricks_session():
    class SparkSession:
        def sql(self, query):
            class DataFrame:
                def toPandas(self):
                    import pandas as pd

                    return pd.DataFrame({"col_1": [1, 2], "col_2": ["a", "b"]})

                def collect(self):
                    return [[1]]

                def repartition(self, n):
                    return self

                def write(self):
                    return self

                def format(self, fmt):
                    return self

                def mode(self, mode):
                    return self

                def saveAsTable(self, table_name):
                    return True

            return DataFrame()

        def createDataFrame(self, data, schema=None):
            class DataFrame:
                def repartition(self, n):
                    return self

                @property
                def write(self):
                    return self

                def format(self, fmt):
                    return self

                def mode(self, mode):
                    return self

                def saveAsTable(self, table_name):
                    return True

            return DataFrame()

    return SparkSession()


class MockDatabricksConnection(DatabricksConnection):
    def __init__(self, *args, **kwargs):
        self.server_hostname = "test.cloud.databricks.com"
        self.access_token = "test-token"
        self.cluster_id = "test-cluster"
        self.parallelism = 1000
        self.spark = databricks_session()
