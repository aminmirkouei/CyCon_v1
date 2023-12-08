from flask import Flask, g
from logging.config import dictConfig

from backend import auth, home, about, modules, config, experiments, datasets, gallery
from backend.db import DBManager

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "wsgi": {"class": "logging.StreamHandler", "formatter": "default"}
        },
        "root": {"level": "DEBUG", "handlers": ["wsgi"]},
    }
)

UPLOAD_FOLDER = "data/"


def create_app(password_file="/run/secrets/db-password"):
    app = Flask(__name__, static_folder="static")
    app.config.from_mapping(**config.config)
    app.config["DB_PASSWORD"] = password_file
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # security
    app.config.update(
        SESSION_COOKIE_SECURE=True,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )

    app.register_blueprint(auth.bp)
    app.register_blueprint(home.bp)
    app.register_blueprint(about.bp)
    app.register_blueprint(modules.bp)
    app.register_blueprint(experiments.bp)
    app.register_blueprint(datasets.bp)
    app.register_blueprint(gallery.bp)

    app.add_url_rule("/", endpoint="index")
    return app
