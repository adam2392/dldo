import os
from pathlib import Path
from typing import Union

import numpy as np


def _load_from_txt(fpath: Union[Path, str]):
    data = np.genfromtxt(fpath)
    return data
