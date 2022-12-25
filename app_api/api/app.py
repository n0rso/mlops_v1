import logging

import api.persistence.models as sql_models
import connexion
from api.config import Config
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

    connexion_app.add_api("api.yaml")  # read the swagger.yml file to configure the endpoints
    _logger.info("Application instance created")

    return


app = connexion.App(__name__, specification_dir="./")
app.add_api("swagger.yml")


@app.route("/")
def home():
    return "Hello!"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
