from osgeo import gdal

DEFAULT_VALUES_COLUMN_IDX = 0
DEFAULT_LABELS_COLUMN_IDX = 1

gdal.UseExceptions()


def get_value_labels(dataset_uri: str, band: int):
    """
    Get the value labels for a given dataset and band.
    """
    print(f"Computing value labels for band {band}")
    try:
        dataset = gdal.Open(dataset_uri)  # dataset_uri is path to .tif file
        band = dataset.GetRasterBand(band)

        rat = band.GetDefaultRAT()
        if rat is None:
            print(f"\tNo Raster Attribute Table (RAT) found for band {band}")
            return None

        # Pick columns for values and labels
        values_column_idx = get_values_column_idx(rat)
        labels_column_idx = get_labels_column_idx(rat, [values_column_idx])

        print(f"\tSelected column for Values: {rat.GetNameOfCol(values_column_idx)}")
        print(f"\tSelected column for Labels: {rat.GetNameOfCol(labels_column_idx)}")

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
    1. Excluding the column already selected for values
    2. Ranking remaining columns based on:
       - Number of unique values in the column
       - Number of different words in the column
    3. Selecting the column with the highest combined score

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
