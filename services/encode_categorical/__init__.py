from flask import Blueprint


encode_categorical = Blueprint('encode_categorical',__name__,template_folder='templates',static_folder='static')

from . import route