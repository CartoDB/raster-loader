import os
import uuid
import re

def get_default_table_name(base_path: str, band: tuple[int]):
    table = os.path.basename(base_path).split(".")[0]
    table = "_".join([table, "band", str(band), str(uuid.uuid4())])
    return re.sub(r"[^a-zA-Z0-9_-]", "_", table)
