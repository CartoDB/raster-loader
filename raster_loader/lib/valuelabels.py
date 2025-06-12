import json
from typing import Literal
import click
from osgeo import gdal

DEFAULT_VALUES_COLUMN_IDX = 0
DEFAULT_LABELS_COLUMN_IDX = 1

gdal.UseExceptions()


def get_band_valuelabels(
    file_path: str,
    band: int,
    band_valuelabels: list[dict[int, str]],
    rat_valuelabels_mode: Literal["auto", "interactive"],
) -> dict[int, str]:
    if len(band_valuelabels) >= band and band_valuelabels[band - 1] is not None:
        # Using valuelabels provided by the user
        valuelabels = band_valuelabels[band - 1]
    else:
        # Computing valuelabels from the Raster Attribute Table (RAT)
        valuelabels = get_valuelabels_from_rat(file_path, band, rat_valuelabels_mode)
    return valuelabels


def get_valuelabels_from_rat(
    dataset_uri: str, band_idx: int, mode: Literal["auto", "interactive"]
) -> dict[int, str]:
    """
    Get the value labels for a given dataset and band.
    """
    print(f"Computing value labels for band {band_idx}")
    try:
        dataset = gdal.Open(dataset_uri)  # dataset_uri is path to .tif file
        band = dataset.GetRasterBand(band_idx)

        rat = band.GetDefaultRAT()
        if rat is None:
            print(f"No Raster Attribute Table (RAT) found for band {band_idx}")
            return None

        print(f"Available columns in Raster Attribute Table (RAT) for band {band_idx}:")
        for col_idx in range(rat.GetColumnCount()):
            print(f"\t{col_idx}: {rat.GetNameOfCol(col_idx)}")

        if mode == "interactive":
            values_column_idx = click.prompt(
                f"Introduce the column index for Values for band {band_idx}",
                type=int,
                default=DEFAULT_VALUES_COLUMN_IDX,
            )
            labels_column_idx = click.prompt(
                f"Introduce the column index for Labels for band {band_idx}",
                type=int,
                default=DEFAULT_LABELS_COLUMN_IDX,
            )
        else:
            # Guess columns for values and labels
            values_column_idx = get_values_column_idx(rat)
            labels_column_idx = get_labels_column_idx(rat, [values_column_idx])

        print(
            f"\tSelected column for Values: "
            f"[{values_column_idx}: {rat.GetNameOfCol(values_column_idx)}]"
        )
        print(
            f"\tSelected column for Labels: "
            f"[{labels_column_idx}: {rat.GetNameOfCol(labels_column_idx)}]"
        )

        # Convert RAT to dictionary
        value_labels = {}
        for i in range(rat.GetRowCount()):
            value = rat.GetValueAsInt(i, values_column_idx)
            label = rat.GetValueAsString(i, labels_column_idx)
            value_labels[value] = label
        return value_labels
    except ValueError:
        return None


def get_values_column_idx(rat: gdal.RasterAttributeTable) -> int:
    values_column_idx = DEFAULT_VALUES_COLUMN_IDX
    for col_idx in range(rat.GetColumnCount()):
        col_name = rat.GetNameOfCol(col_idx)
        if col_name.lower() == "value":
            values_column_idx = col_idx

    return values_column_idx


def get_labels_column_idx(
    rat: gdal.RasterAttributeTable, skip_columns: list[int] = []
) -> int:
    """
    Get the column index of the labels column in a Raster Attribute Table (RAT).

    This function uses a heuristic approach to identify the most likely column
    containing descriptive labels by:
    1. Excluding columns specified in skip_columns (typically the values column)
    2. Ranking remaining columns based on:
       - Number of unique values in the column
       - Number of unique words in the column
    3. Selecting the column with the highest combined score (unique values * unique words)
       If no column has a score > 0, falls back to selecting the column with most unique values

    Args:
        rat: GDAL Raster Attribute Table
        skip_columns: List of column indices to exclude from consideration

    Returns:
        int: The index of the column most likely to contain labels
    """
    labels_column_idx = DEFAULT_LABELS_COLUMN_IDX
    unique_values_count = count_column_unique_values(rat)

    unique_words_count = count_column_unique_words(rat)

    # Ranked columns by (unique values count * bigrams count)
    ranked_columns = {}
    for col_idx in range(rat.GetColumnCount()):
        if col_idx in skip_columns:
            continue
        ranked_columns[col_idx] = (
            unique_values_count[col_idx] * unique_words_count[col_idx]
        )

    # Sort the ranked columns by the rank
    ranked_columns = sorted(ranked_columns.items(), key=lambda x: x[1], reverse=True)

    # Get the column with the highest rank
    if ranked_columns[0][1] > 0:
        labels_column_idx = ranked_columns[0][0]
    else:
        # If no bigrams found, select column with most unique values
        labels_column_idx = max(unique_values_count.items(), key=lambda x: x[1])[0]

    return labels_column_idx


# Return a dictionary with the column_idx and the number of unique values in it
def count_column_unique_values(rat: gdal.RasterAttributeTable) -> dict[int, int]:
    unique_values_count = {}
    for col_idx in range(rat.GetColumnCount()):
        col_unique_values = set()
        for row_idx in range(rat.GetRowCount()):
            value = rat.GetValueAsString(row_idx, col_idx)
            col_unique_values.add(value)
        unique_values_count[col_idx] = len(col_unique_values)

    return unique_values_count


def count_column_unique_words(rat: gdal.RasterAttributeTable) -> dict[int, int]:
    columns_words_count = {}
    for col_idx in range(rat.GetColumnCount()):
        column_text = ""
        for row_idx in range(rat.GetRowCount()):
            column_text += rat.GetValueAsString(row_idx, col_idx).lower() + " "
        words_list = set(column_text.split())
        # Remove words which are not text
        words_list = [w for w in words_list if w.isalpha()]
        columns_words_count[col_idx] = len(words_list)

    return columns_words_count


def validate_band_valuelabels(_, __, value):
    """
    Validate the band valuelabels parameter for click library callback.
    """
    try:
        return [json.loads(item) if item != "None" else None for item in value]
    except json.JSONDecodeError:
        raise click.BadParameter(
            "Invalid JSON format. Please provide a valid JSON object."
        )
