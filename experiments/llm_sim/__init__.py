"""S2 simulated-persona evaluation facade.

The package is deliberately independent from the 8700 application/session
store.  It uses the production catalog, scoring, mastery, and selector modules
as pure in-process dependencies and writes only to an explicit simulation root.
"""

from .models import Persona, ProviderSpec, StudyPanel
from .config import LLMSimConfig, load_frozen_config
from .panel import freeze_manipulation_panel, load_frozen_panel
from .personas import build_personas
from .runner import (
    CircuitOpenError,
    LLMSimulationRunner,
    ModelDriftError,
    ProviderCallPolicy,
)
from .store import SimulationStore
from .transport import HTTPProviderTransport, PROVIDER_SPECS, ProviderTransport

__all__ = [
    "CircuitOpenError",
    "HTTPProviderTransport",
    "LLMSimulationRunner",
    "LLMSimConfig",
    "ModelDriftError",
    "PROVIDER_SPECS",
    "Persona",
    "ProviderCallPolicy",
    "ProviderSpec",
    "ProviderTransport",
    "SimulationStore",
    "StudyPanel",
    "build_personas",
    "freeze_manipulation_panel",
    "load_frozen_panel",
    "load_frozen_config",
]
