import json
import logging
import typing as t

from api.config import APP_NAME
from api.persistence.data_access import ModelType, PredictionPersistence
from flask import Response, current_app, jsonify, request
from prometheus_client import Gauge, Histogram, Info

_logger = logging.getLogger(__name__)


PREDICTION_TRACKER = Histogram(
    name="board_game_rating",
    documentation="ML Model for Board Game Rating Prediction",
    labelnames=["app_name", "model_name", "model_version"],
)

PREDICTION_GAUGE = Gauge(
    name="board_game_gauge_rating",
    documentation="ML Model for Board Game Rating Prediction",
    labelnames=["app_name", "model_name", "model_version"],
)

PREDICTION_GAUGE.labels(app_name=APP_NAME, model_name=ModelType.LASSO.name, model_version="0.1.0")

MODEL_VERSIONS = Info(
    "model_version_details",
    "Capture model version information",
)

MODEL_VERSIONS.info(
    {
        "model": ModelType.GB.name,
        "version": "0.1.0",
    }
)


def health():
    if request.method == "GET":
        return jsonify({"status": "ok"})


def predict():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()

        # Step 2a: Get and save live model predictions
        persistence = PredictionPersistence(db_session=current_app.db_session)
        result = persistence.make_save_predictions(db_model=ModelType.LASSO, input_data=json_data)
        # Step 3: Handle errors
        if result.errors:
            _logger.warning(f"errors during prediction: {result.errors}")
            return Response(json.dumps(result.errors), status=400)

        # Step 4: Monitoring
        for _prediction in result.predictions:
            PREDICTION_TRACKER.labels(app_name=APP_NAME, model_name=ModelType.GB.name, model_version="0.1.0").observe(
                _prediction
            )

            PREDICTION_GAUGE.labels(app_name=APP_NAME, model_name=ModelType.GB.name, model_version="0.1.0").set(
                _prediction
            )

        # Step 5: Prepare prediction response
        return jsonify(
            {
                "predictions": result.predictions,
                "version": result.model_version,
                "errors": result.errors,
            }
        )
