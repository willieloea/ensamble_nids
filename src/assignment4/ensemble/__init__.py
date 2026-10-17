"""Classification ensembles with user-supplied training and prediction functions."""

from .bagging import bagging
from .boosting import boosting
from ._common import VotingEnsemble

__all__ = ["bagging", "boosting", "VotingEnsemble"]
