import sys
import json
import requests

from raster_loader.errors import UploadError
from raster_loader.utils import some_function


class RasterLoader:
    """CARTO Authentication object used to gather connect with the CARTO services.

    Args:
        some_args (str): some_args comments.
        some_other_args (str, optional): some_other_args comments.
            Default True.
    """

    def __init__(
        self,
        mode,
        use_cache=True
    ):
        self._mode = mode
        self._use_cache = use_cache

        UploadError("Some error")


    @classmethod
    def from_oauth(
        cls,
        use_cache=True,
    ):
        """Some doc.

        Args:
            use_cache (bool, optional): Whether cache should be used.
                Default True.
        """
        mode = "oauth"

        return cls(
            mode=mode,
            use_cache=use_cache,
        )
