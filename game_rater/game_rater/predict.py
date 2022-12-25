import logging
from typing import Dict, Union

import pandas as pd
from game_rater.configs.config import config
from game_rater.utils.data_utils import load_pipeline

from game_rater import __version__ as _version

_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(
    *,
    input_data: Union[pd.DataFrame, Dict],
) -> Dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    preds = pipe.predict(X=data[config.model_config.features])
    _logger.info(f"Making predictions with model version: {_version} " f"Predictions: {preds}")
    results = {"predictions": preds, "version": _version}
    return results
