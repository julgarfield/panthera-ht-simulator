from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from panthera.config import load_config
from panthera.environment import SimulatedPanthera


@pytest.fixture(scope='session')
def shared_env():
    return SimulatedPanthera(load_config())


@pytest.fixture
def env(shared_env):
    shared_env.reset()
    return shared_env
