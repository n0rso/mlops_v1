import json
import logging
import typing as t

from api.config import APP_NAME
from api.persistence.data_access import ModelType, PredictionPersistence
from flask import Response, current_app, jsonify, request

from game_rater import __version__ as _version
from game_rater.predict import make_prediction

_logger = logging.getLogger(__name__)


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

        # # Step 4: Monitoring
        # for _prediction in result.predictions:
        #     PREDICTION_TRACKER.labels(
        #         app_name=APP_NAME, model_name=ModelType.LASSO.name, model_version=live_version
        #     ).observe(_prediction)

        #     PREDICTION_GAUGE.labels(app_name=APP_NAME, model_name=ModelType.LASSO.name, model_version=live_version).set(
        #         _prediction
        #     )

        # Step 5: Prepare prediction response
        return jsonify(
            {
                "predictions": result.predictions,
                "version": result.model_version,
                "errors": result.errors,
            }
        )
