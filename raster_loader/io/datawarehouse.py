import pandas as pd

from typing import List


class DataWarehouseConnection:
    def __init__(self, *args, **kwargs):
        pass

    def execute(self, sql):
        """Execute a SQL query."""
        raise NotImplementedError

    def execute_to_dataframe(self, sql):
        """Execute a SQL query and return the result as a pandas dataframe.
        Parameters
        ----------
        sql : str
            SQL query to execute.
        Returns
        -------
        pandas.DataFrame
            Result of the query.
        """
        raise NotImplementedError

    def create_table(self, fqn):
        """Create a table.
        Parameters
        ----------
        fqn : str
            Fully qualified name of the table.
        """
        self.execute(f"CREATE TABLE IF NOT EXISTS {self.quote(fqn)})")

    def delete_table(self, fqn):
        """Delete a table.
        Parameters
        ----------
        fqn : str
            Fully qualified name of the table.
        """
        self.execute(f"DROP TABLE IF EXISTS {self.quote_name(fqn)}")

    def quote(self, value):
        """Quote a value.
        Parameters
        ----------
        value : str
            Value to quote.
        Returns
        -------
        str
            Quoted value.
        """
        if isinstance(value, str):
            value = value.replace("\\", "\\\\")
            return f"'{value}'"
        return str(value)

    def upload_raster(
        self,
        file_path: str,
        fqn: str,
        band: int = 1,
        band_name: str = None,
        chunk_size: int = 10000,
        overwrite: bool = False,
        append: bool = False,
    ):
        """Upload a raster file to the data warehouse.
        Parameters
        ----------
        file_path : str
            Path to the raster file.
        fqn : str
            Fully qualified name of the table.
        band : int, optional
            Band to upload, by default 1
        band_name : str, optional
            Name of the band
        chunk_size : int, optional
            Number of blocks to upload in each chunk, by default 10000
        overwrite : bool, optional
            Overwrite existing data in the table if it already exists, by default False
        append : bool, optional
            Append records into a table if it already exists, by default False
        """
        raise NotImplementedError

    def get_records(self, fqn: str, limit=10) -> pd.DataFrame:
        """Get records from a table.
        Parameters
        ----------
        fqn : str
            Fully qualified name of the table.
        limit : int, optional
            Maximum number of records to return, by default 10
        Returns
        -------
        pandas.DataFrame
            Records from the table.
        """
        query = f"SELECT * FROM {self.quote_name(fqn)} LIMIT {limit}"
        return self.execute_to_dataframe(query)

    def band_rename_function(self, band_name: str):
        return band_name

    def insert_in_table(
        self,
        rows: List[dict],
        fqn: str,
    ) -> bool:
        """Insert records into a table.
        Parameters
        ----------
        rows : List[dict]
            Records to insert.
        fqn : str
            Fully qualified name of the table.
        Returns
        -------
        bool
            True if the insertion was successful, False otherwise.
        """
        columns = rows[0].keys()
        values = ",".join(
            [
                "(" + ",".join([self.quote(row[column]) for column in columns]) + ")"
                for row in rows
            ]
        )
        query = f"""
            INSERT INTO {self.quote_name(fqn)}({','.join(columns)})
            VALUES {values}
            """
        self.execute(query)

        return True

    def write_metadata(
        self,
        metadata,
        append_records,
        fqn,
    ):
        """Write metadata to a table.
        Parameters
        ----------
        metadata : dict
            Metadata to write.
        append_records : bool
            Whether to update the metadata of an existing table or insert a new record.
        fqn : str
            Fully qualified name of the table.
        Returns
        -------
        bool
            True if the insertion was successful, False otherwise.
        """
        raise NotImplementedError

    def get_metadata(self, fqn):
        """Get metadata from a table.
        Parameters
        ----------
        fqn : str
            Fully qualified name of the table.
        Returns
        -------
        dict
            Metadata from the table.
        """
        raise NotImplementedError

    def quote_name(self, name):
        """Quote a table name.
        Parameters
        ----------
        name : str
            Name to quote.
        Returns
        -------
        str
            Quoted name.
        """
        return f"{name}"
