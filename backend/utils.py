from flask import make_response

ALLOWED_EXTENSIONS = {'csv', 'mp4', 'mp3',
                      'png', 'jpg', 'jpeg', 'gif', 'wav', "txt", "npy"}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def user_active(user):
    return user and user.get("status") == "active"


def add_security_headers(template):
    response = make_response(template)
    # tls
    response.headers[
        "Strict-Transport-Security"
    ] = "max-age=31536000; includeSubDomains"
    # csp
    # response.headers[
    #     "Content-Security-Policy"
    # ] = "script-src 'self' https://cycon.nkn.uidaho.edu; script-src 'unsafe-eval'; object-src 'self'; img-src https: data:;"
    # xss
    response.headers["X-Content-Type-Options"] = "nosniff"
    # frame protection
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    # caching
    response.headers["Cache-control"] = "no-store"

    return response
