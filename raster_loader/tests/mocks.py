def bigquery_client(load_error=False):
    class BigQueryClient:
        def __init__(self, load_error):
            self.load_error = load_error

        def load_table_from_dataframe(self, *args, **kwargs):
            if load_error:  # pragma: no cover
                raise Exception

    return BigQueryClient(load_error=load_error)
