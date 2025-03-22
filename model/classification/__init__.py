from flask import Blueprint

classification = Blueprint('classification',__name__,template_folder='templates',static_folder='static')

from . import route