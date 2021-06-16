import sys
from pkg_resources import (
    DistributionNotFound as _DistributionNotFound,
    get_distribution as _get_distribution,
)

if sys.version_info < (3,7):
    raise Exception("FusionKit does not support Python < 3.7")

# import core framework
from .core.dataspine import DataSpine
from .core.equilibrium import Equilibrium
from .core.plasma import Plasma
from .core.remote import Remote
from .core.utils import find

# import framework extensions
from .extensions.ex2gk import EX2GK
from .extensions.gene import GENE
from .extensions.jet_ppf import JET_PPF
from .extensions.qualikiz import QLK
from .extensions.tglf import TGLF

__all__ = [
    "DataSpine",
    "EX2GK",
    "Equilibrium",
    "find",
    "GENE",
    "JET_PPF",
    "Plasma",
    "Remote",
    "QLK",
    "TGLF",
]

__author__ = "Garud Snoep"
__created__ = "10 May 2021"

try:
    _distribution = _get_distribution("fusionkit")
    __version__ = _distribution.version
except _DistributionNotFound:
    __version__ = "unknown"
