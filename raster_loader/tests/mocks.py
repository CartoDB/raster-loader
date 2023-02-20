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

            return job

        def query(self, query):
            class job(object):
                def result():
                    return True

            return job

        def create_table(self, table):
            return True

    return BigQueryClient(load_error=load_error)
