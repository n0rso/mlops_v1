import logging
from typing import Dict, Union

import pandas as pd

from game_rater import __version__ as _version
from game_rater.configs.config import config
from game_rater.utils.data_utils import load_pipeline

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: Union[pd.DataFrame, Dict],
) -> Dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    results = {"predictions": None, "version": _version}
    return results
