import logging
import sys
from pathlib import Path

import api.persistence.models as sql_models
import connexion
from api.config import Config

sys.path.append(Path(__file__).parent)
from api.monitoring.middleware import setup_metrics
from api.persistence.core import init_database
from sqlalchemy.orm import scoped_session

_logger = logging.getLogger(__name__)


def create_app(*, config_object: Config, db_session: scoped_session = None) -> connexion.App:
    """Create app instance."""

    connexion_app = connexion.App(
        __name__, debug=config_object.DEBUG, specification_dir="spec/"
    )  # create the application instance
    flask_app = connexion_app.app
    flask_app.config.from_object(config_object)

    # Setup database
    init_database(flask_app, config=config_object, db_session=db_session, base=sql_models.Base)

    # Setup prometheus monitoring
    setup_metrics(flask_app)

    connexion_app.add_api("swagger.yml")  # read the swagger.yml file to configure the endpoints
    _logger.info("Application instance created")

    return connexion_app
