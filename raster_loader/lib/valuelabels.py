import json
import click
from typing import Dict, List, Optional


def get_band_valuelabels(
    file_path: str, band: int, band_valuelabels: List[Dict[int, str]]
) -> Optional[Dict[int, str]]:
    valuelabels = None
    if len(band_valuelabels) >= band and band_valuelabels[band - 1] is not None:
        print(f"Using the provided valuelabels for band {band}")
        valuelabels = band_valuelabels[band - 1]
    return valuelabels


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
