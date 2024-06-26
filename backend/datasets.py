from flask import Blueprint, render_template
from werkzeug.utils import secure_filename
from flask import current_app, flash
from flask import request, g, redirect, url_for
from flask import session
import logging
from pathlib import Path
import os

from mysql.connector import Error

from backend.utils import add_security_headers, allowed_file, ALLOWED_EXTENSIONS

from backend.db import DBManager

bp = Blueprint("datasets", __name__)


@bp.route("/datasets", methods=["GET", "POST"])
def index():
    extensions = list(ALLOWED_EXTENSIONS)
    if request.method == "GET":
        user_id = session.get("user_id")
        logging.debug("=== Uploading files: %s", user_id)

        if user_id is None:
            g.user = None
            flash("You need to login to upload your datasets!")
            return redirect(url_for("auth.login"))
        else:
            try:
                manager = DBManager.instance(
                    password_file=current_app.config["DB_PASSWORD"]
                )
                plan = manager.get_plan_by_user_id(user_id)
            except Error as e:
                logging.debug(f"=== Failed to get user plan: {e} ===")
            if plan is None:
                flash('You have not yet created a plan.j')
                return redirect(url_for("auth.plans"))
            else:
                user_storage = plan.get("storage_url")
                files_path = Path('data/') / user_storage
                files = [f.name for f in list(files_path.glob("*"))]
                logging.debug(f"=== Retrieving files: %s", files)
                return add_security_headers(
                    render_template("datasets/index.html",
                                    files=files, extensions=extensions)
                )

    return add_security_headers(
        render_template("datasets/index.html",
                        extensions=extensions))


@bp.route('/datasets/upload', methods=['GET', 'POST'])
def upload_file():
    user_id = session.get("user_id")
    logging.debug("=== Uploading files: %s", user_id)

    if user_id is None:
        g.user = None
        flash("You need to login to upload your datasets!")
        return redirect(url_for("auth.login"))
    else:
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for("datasets.index"))
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for("datasets.index"))

        logging.debug("=== + Upload file: %s, %s", file, file.filename)

        if file and allowed_file(file.filename):
            try:
                manager = DBManager.instance(
                    password_file=current_app.config["DB_PASSWORD"]
                )
                plan = manager.get_plan_by_user_id(user_id)
            except Error as e:
                logging.debug(f"=== Failed to get user plan: {e} ===")

            logging.debug("=== Upload plan: %s", plan)
            if plan is None:
                flash("You have not subscribed to a plan!")
                return redirect(url_for("auth.plans"))
            else:
                # if plan.get('tier') == 'free':
                # TODO check file size
                filename = secure_filename(file.filename)
                user_storage = plan.get("storage_url")
                base_folder = os.getcwd()
                upload_path = Path(base_folder + '/data/') / user_storage
                Path(upload_path).mkdir(parents=True, exist_ok=True)
                if os.path.getsize(upload_path) > 104857600:  # 100 MB
                    error = "You have uploaded more than 100MB"
                    flash(error)
                elif len(os.listdir(upload_path)) >= 10:  # more than 10 files
                    error = "You have uploaded more than 10 files"
                    flash(error)
                else:
                    file.save(os.path.join(upload_path, filename))
                    flash("File uploaded successfully")
    return redirect(url_for("datasets.index"))


@bp.route('/datasets/delete', methods=['GET', 'POST'])
def delete_file():
    user_id = session.get("user_id")
    logging.debug("=== Deleting files: %s", user_id)
    filename = request.form.get("filename")

    if user_id is None:
        g.user = None
        flash("You need to login to upload your datasets!")
        return redirect(url_for("auth.login"))
    else:
        # check if the post request has the file part
        try:
            manager = DBManager.instance(
                password_file=current_app.config["DB_PASSWORD"]
            )
            plan = manager.get_plan_by_user_id(user_id)
        except Error as e:
            logging.debug(f"=== Failed to get user plan: {e} ===")

        logging.debug("=== Upload plan: %s", plan)
        if plan is None:
            flash("You have not subscribed to a plan!")
            return redirect(url_for("auth.plans"))
        else:
            # if plan.get('tier') == 'free':
            # TODO check file size
            filename = secure_filename(filename)
            user_storage = plan.get("storage_url")
            base_folder = os.getcwd()
            upload_path = Path(base_folder+'/data/') / user_storage
            file_path = upload_path / filename
            Path(upload_path).mkdir(parents=True, exist_ok=True)
            if os.path.exists(file_path):
                os.remove(file_path)
                flash("File deleted successfully")
            else:
                flash("File does not exist on the server")
    return redirect(url_for("datasets.index"))
