from pathlib import Path

from flask import Blueprint
from loguru import logger

from ..extentions import Api, app
from .react_app_view import ReactAppView

folder = Path(app.config.get('REACT_APP_DIST', '../../web/dist/')).resolve()
logger.info(f'react_app folder: {folder}')
react_app_blue = Blueprint('react_app', __name__,  static_folder=folder, static_url_path='', template_folder=folder)
react_app_api = Api(react_app_blue, prefix='')
react_app_api.add_resource(ReactAppView, '/')
