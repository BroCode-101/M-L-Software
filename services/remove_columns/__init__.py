from flask import Blueprint

remove_columns = Blueprint('remove_columns',__name__,template_folder='templates',static_folder='static')

from . import route