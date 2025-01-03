from flask import Blueprint

handle_nan = Blueprint('handle_nan', __name__, template_folder='templates', static_folder='static')

from . import routes
