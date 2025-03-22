from flask import Blueprint

regression = Blueprint('regression', __name__,template_folder='templates',static_folder='static')
from . import route 