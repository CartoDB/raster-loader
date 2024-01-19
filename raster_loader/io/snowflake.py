import json
import rasterio
import pandas as pd

from typing import Iterable, List, Tuple

from raster_loader.errors import (
    IncompatibleRasterException,
)

from raster_loader.utils import ask_yes_no_question, batched, snowflake_sql_quote

from raster_loader.io.common import (
    rasterio_metadata,
    rasterio_windows_to_records,
    get_number_of_blocks,
    check_metadata_is_compatible,
    update_metadata,
)

try:
    from snowflake.connector.pandas_tools import write_pandas
except ImportError:  # pragma: no cover
    _has_snowflake = False
else:
    _has_snowflake = True


def table_columns(bands: List[str]) -> List[Tuple[str, str, str]]:
    columns = [("BLOCK", "INTEGER", "REQUIRED"), ("METADATA", "STRING", "NULLABLE")]
    columns += [(band_name.upper(), "VARIANT", "NULLABLE") for band_name in bands]
    return columns


def write_metadata(
    metadata,
    append_records,
    database,
    schema,
    table,
    client=None,
):
    """Write the metadata of a raster file to a Snowflake table.

    Parameters
    ----------
    metadata : dict
        The metadata of the raster file.
    append_records : bool
        Whether the table already contains records.
    database : str
        Database name.
    schema : str
        Schema name.
    table : str
        Table name.
    client : snowflake.connector
        The Snowflake client.

    Returns
    -------
    bool
        True if the metadata was written.
    """

    if append_records:
        query = f"""
            UPDATE {database}.{schema}.{table}
            SET metadata = (
                SELECT TO_JSON(
                    PARSE_JSON(
                        {snowflake_sql_quote(json.dumps(metadata))}
                    )
                )
            ) WHERE block = 0
        """

        client.cursor().execute(query)

        return True
    else:
        client.cursor().execute(
            f"""
                ALTER TABLE {database}.{schema}.{table}
                ADD COLUMN metadata STRING;
            """
        )
        return insert_in_table(
            [
                {
                    "BLOCK": 0,  # store metadata in the record with this block number
                    "METADATA": json.dumps(metadata),
                }
            ],
            database,
            schema,
            table,
            client=client,
        )


def get_metadata(database, schema, table, client):
    """Get the metadata of the raster contained in a Snowflake table.

    Parameters
    ----------
    database : str
        Database name.
    schema : str
        Schema name.
    table : str
        Table name.
    client : snowflake.connector
        The Snowflake client.

    Returns
    -------
    dict
        The metadata of the table.
    """

    query = f"""
        SELECT metadata
        FROM {database}.{schema}.{table}
        WHERE block = 0
    """
    cur = client.cursor()
    cur.execute(query)
    result = cur.fetchone()
    return json.loads(result[0])


def records_to_table(
    records: Iterable,
    table: str,
    schema: str,
    database: str,
    overwrite: bool,
    client,
):
    """Write a record to a Snowflake table."""

    records_list = []
    for record in records:
        del record["METADATA"]
        records_list.append(record)

    data_df = pd.DataFrame(records_list)

    return write_pandas(
        conn=client,
        df=data_df,
        table_name=table,
        database=database,
        schema=schema,
        chunk_size=1000,
        auto_create_table=True,
        overwrite=overwrite,
    )[0]


def table_to_records(
    database: str, schema: str, table: str, client, limit=10
) -> pd.DataFrame:  # pragma: no cover
    """Read a Snowflake table into a records pandas.DataFrame.

    Parameters
    ----------
    database : str
        Database name.
    schema : str
        Schema name.
    table : str
        Table name.
    client : snowflake.connector
        The Snowflake client.
    limit : int, optional
        Max number of records to return, by default 10.

    Returns
    -------
    pandas.DataFrame
        Records as a pandas.DataFrame.

    """

    query = f"SELECT * FROM {database}.{schema}.{table} LIMIT {limit}"

    cur = client.cursor()
    cur.execute(query)
    return cur.fetch_pandas_all()


def create_table(
    database: str,
    schema: str,
    table: str,
    columns: List[Tuple[str, str, str]],
    clustering: List[str],
    client=None,
) -> bool:  # pragma: no cover
    query = f"""
        CREATE TABLE {database}.{schema}.{table} (
            {','.join(
                [
                    f"{column_name} {column_type}"
                    for [column_name, column_type, _] in columns
                ]
            )}
        )
        CLUSTER BY ({','.join(clustering)})
    """
    client.cursor().execute(query)

    return True


def insert_in_table(
    rows: List[dict],
    database: str,
    schema: str,
    table: str,
    client,
) -> bool:
    """Insert rows in a Snowflake table.

    Parameters
    ----------
    rows : List[dict],
        Rows to be inserted.
    database : str
        Database name.
    schema : str
        Schema name.
    table : str
        Table name.
    client : snowflake.connector
        The Snowflake client.

    Returns
    -------
    bool
        True if the rows were inserted

    """

    columns = rows[0].keys()
    values = ",".join(
        [
            "("
            + ",".join([snowflake_sql_quote(row[column]) for column in columns])
            + ")"
            for row in rows
        ]
    )
    client.cursor().execute(
        f"""
        INSERT INTO {database}.{schema}.{table} ({','.join(columns)})
        VALUES {values}
        """
    )

    return True


def delete_table(
    database: str,
    schema: str,
    table: str,
    client,
) -> bool:  # pragma: no cover
    """Delete a Snowflake table.

    Parameters
    ----------
    database : str
        Database name.
    schema : str
        Schema name.
    table : str
        Table name.
    client : snowflake.connector
        The Snowflake client.

    Returns
    -------
    bool
        True if the table was deleted.

    """

    client.cursor().execute(
        f"""
        DROP TABLE IF EXISTS {database}.{schema}.{table}
        """
    )

    return True


def check_if_table_exists(
    database: str,
    schema: str,
    table: str,
    client,
):  # pragma: no cover
    """Check if a Snowflake table exists.

    Parameters
    ----------
    database : str
        Database name.
    schema : str
        Schema name.
    table : str
        Table name.
    client : snowflake.connector
        The Snowflake client.

    Returns
    -------
    bool
        True if the table exists, False otherwise.
    """

    cursor = client.cursor()
    res = cursor.execute(
        f"""
    SELECT *
    FROM {database}.INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = '{schema.upper()}'
    AND TABLE_NAME = '{table.upper()}';
    """
    ).fetchall()

    return len(res) > 0


def check_if_table_is_empty(
    database: str,
    schema: str,
    table: str,
    client,
):  # pragma: no cover
    """Check if a Snowflake table is empty.

    Parameters
    ----------
    database : str
        Database name.
    schema : str
        Schema name.
    table : str
        Table name.
    client : snowflake.connector
        The Snowflake client.

    Returns
    -------
    bool
        True if the table is empty, False otherwise.
    """

    cur = client.cursor()
    cur.execute(
        f"""
        SELECT ROW_COUNT
        FROM {database}.INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = '{schema.upper()}'
        AND TABLE_NAME = '{table.upper()}';
        """
    )
    res = cur.fetchone()
    return res[0] == 0


def rasterio_to_table(
    file_path: str,
    database: str,
    schema: str,
    table: str,
    client,
    bands_info: List[Tuple[int, str]] = None,
    chunk_size: int = None,
    overwrite: bool = False,
    append: bool = False,
) -> bool:
    """Write a rasterio-compatible raster file to a Snowflake table.

    Parameters
    ----------
    file_path : str
        Path to the rasterio-compatible raster file.
    database : str
        Database name.
    schema : str
        Schema name.
    table : str
        Table name.
    bands_info : [(int, str)], optional
        Band number(s) and column name(s) to read from the
        raster file, by default [(1, None)].
        If column_name is None default column name is band_<band_num>.
    client : snowflake.connector
        The Snowflake client.
    chunk_size : int, optional
        Number of records to write to Snowflake at a time, the default (None)
        writes all records in a single batch.
    overwrite : bool, optional
        Overwrite the table if it already contains data, by default False
    append : bool, optional
        Append records into a table if it already exists, by default False

    Returns
    -------
    bool
        True if upload was successful.
    """

    print("Loading raster file to Snowflake...")

    bands_info = bands_info or [(1, None)]

    append_records = False

    try:
        if (
            check_if_table_exists(database, schema, table, client)
            and not check_if_table_is_empty(database, schema, table, client)
            and not overwrite
        ):
            append_records = append or ask_yes_no_question(
                f"Table {table} already exists in schema {schema} "
                "and is not empty. Append records? [yes/no] "
            )

            if not append_records:
                exit()

        metadata = rasterio_metadata(file_path, bands_info, lambda x: x.upper())

        records_gen = rasterio_windows_to_records(
            file_path,
            lambda *args: None,
            table_columns,
            lambda x: x.upper(),
            bands_info,
        )

        total_blocks = get_number_of_blocks(file_path)

        if chunk_size is None:
            ret = records_to_table(
                records_gen, table, schema, database, overwrite, client
            )
            if not ret:
                raise IOError("Error uploading to Snowflake.")
        else:
            from tqdm.auto import tqdm

            print(f"Writing {total_blocks} blocks to Snowflake...")
            with tqdm(total=total_blocks) as pbar:
                if total_blocks < chunk_size:
                    chunk_size = total_blocks
                for records in batched(records_gen, chunk_size):
                    ret = records_to_table(
                        records, table, schema, database, overwrite, client
                    )
                    pbar.update(chunk_size)
                    if not ret:
                        raise IOError("Error uploading to Snowflake.")
                pbar.update(1)

        print("Writing metadata to Snowflake...")
        if append_records:
            old_metadata = get_metadata(database, schema, table, client)
            check_metadata_is_compatible(metadata, old_metadata)
            update_metadata(metadata, old_metadata)

        write_metadata(
            metadata,
            append_records,
            database,
            schema,
            table,
            client=client,
        )

    except IncompatibleRasterException as e:
        raise IOError("Error uploading to Snowflake: {}".format(e.message))

    except KeyboardInterrupt:
        delete = ask_yes_no_question(
            "Would you like to delete the partially uploaded table? [yes/no] "
        )

        if delete:
            delete_table(database, schema, table, client)

        raise KeyboardInterrupt

    except rasterio.errors.CRSError as e:
        raise e

    except Exception as e:
        delete = ask_yes_no_question(
            (
                "Error uploading to Snowflake. "
                "Would you like to delete the partially uploaded table? [yes/no] "
            )
        )

        if delete:
            delete_table(database, schema, table, client)

        raise IOError("Error uploading to Snowflake: {}".format(e))

    print("Done.")
    return True
