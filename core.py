"""Backward-compatible import shim: the original ``from core import ...`` keeps working."""
from talentflow.core import *  # noqa: F401,F403
from talentflow.core import (FEATURES, TARGET, aggregate, generate, scenario, train, validate)  # noqa: F401
