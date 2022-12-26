import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
from api.app import create_app
from api.config import DevelopmentConfig, setup_app_logging
from werkzeug.middleware.dispatcher import DispatcherMiddleware

_config = DevelopmentConfig()

# setup logging as early as possible
setup_app_logging(config=_config)

main_app = create_app(config_object=_config).app

# Add prometheus wsgi middleware to route /metrics requests
# this middleware allows us to distinguish between our Flask wsgi app and prometheus wsgi app
# read more here: https://flask.palletsprojects.com/en/2.2.x/patterns/appdispatch/
application = DispatcherMiddleware(app=main_app.wsgi_app)  # , mounts={"/metrics": prometheus_client.make_wsgi_app()})


if __name__ == "__main__":
    print(_config)
    main_app.run(port=_config.SERVER_PORT, host=_config.SERVER_HOST)
