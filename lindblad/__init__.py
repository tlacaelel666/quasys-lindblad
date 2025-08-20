from flask import Blueprint

beta_decay_blueprint = Blueprint('beta_decay_blueprint', __name__, url_prefix='/beta_decay')

from .lindblad_beta_decay import BetaDecayAnalyzer

from flask import Blueprint

your_blueprint = Blueprint('your_blueprint', __name__)

from . import routes  # Import routes within the blueprint
