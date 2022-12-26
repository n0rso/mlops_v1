import json
import logging
import sys
import typing as t
from enum import Enum
from pathlib import Path

sys.path.append(Path(__file__).parent.parent.parent)
from api.persistence.models import ModelPredictions
from sqlalchemy.orm.session import Session

from game_rater.game_rater.predict import make_prediction as make_prediction

_logger = logging.getLogger(__name__)


class ModelType(Enum):
    GB = "default"


class PredictionResult(t.NamedTuple):
    errors: None
    predictions: t.Optional[t.List]
    model_version: str


MODEL_PREDICTION_MAP = {
    ModelType.GB: make_prediction,
}


class PredictionPersistence:
    def __init__(
        self,
        *,
        db_session: Session,
        user_id: t.Optional[str] = None,
    ) -> None:
        self.db_session = db_session
        if not user_id:
            # in reality, here we would use something like a UUID for anonymous users
            # and if we had user logins, we would record the user ID.
            self.user_id = "007"

    def make_save_predictions(self, *, db_model: ModelType, input_data: t.List) -> PredictionResult:
        """Get the prediction from a given model and persist it."""

        result = MODEL_PREDICTION_MAP[db_model](input_data=input_data)
        errors = None
        try:
            errors = result["errors"]
        except KeyError:
            # lasso model `make_prediction` does not include errors
            pass

        prediction_result = PredictionResult(
            errors=errors,
            predictions=list(result.get("predictions")) if not errors else None,
            model_version=result.get("version"),
        )

        if prediction_result.errors:
            return prediction_result

        self.save_predictions(
            inputs=input_data,
            prediction_result=prediction_result,
            db_model=db_model,
        )

        return prediction_result

    def save_predictions(
        self,
        *,
        inputs: t.List,
        prediction_result: PredictionResult,
        db_model: ModelType,
    ) -> None:
        """Persist model predictions to storage."""

        prediction_data = ModelPredictions(
            user_id=self.user_id,
            model_version=prediction_result.model_version,
            inputs=json.dumps(inputs),
            outputs=json.dumps(prediction_result.predictions),
        )

        self.db_session.add(prediction_data)
        self.db_session.commit()
        _logger.debug(f"saved data for model: {db_model}")
