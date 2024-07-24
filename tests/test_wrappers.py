import os
import sys
import pytest
import gymnasium as gym

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from tarware.warehouse import Warehouse, Direction, Action, RewardType
from tarware.utils.wrappers import FlattenAgents, DictAgents


@pytest.fixture
def env_single_agent():
    env = Warehouse(3, 8, 3, 1, 0, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_double_agent():
    env = Warehouse(3, 8, 3, 2, 0, 0, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


@pytest.fixture
def env_double_agent_with_msg():
    env = Warehouse(3, 8, 3, 2, 0, 2, 1, 5, None, None, RewardType.GLOBAL)
    env.reset()
    return env


def test_flatten_agents_0(env_single_agent):
    env = FlattenAgents(env_single_agent)
    obs = env.reset()
    assert len(obs.shape) == 1
    obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
    assert len(obs.shape) == 1
    assert rew.shape == ()
    assert type(terminated) is bool
    assert type(truncated) is bool


def test_flatten_agents_1(env_double_agent):
    env = FlattenAgents(env_double_agent)
    obs = env.reset()
    assert len(obs.shape) == 1
    obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
    assert len(obs.shape) == 1
    assert rew.shape == ()
    assert type(terminated) is bool
    assert type(truncated) is bool


def test_flatten_agents_2(env_double_agent_with_msg):
    env = FlattenAgents(env_double_agent_with_msg)
    obs = env.reset()
    assert len(obs.shape) == 1
    obs, rew, terminated, truncated, _ = env.step(env.action_space.sample())
    assert len(obs.shape) == 1
    assert rew.shape == ()
    assert type(terminated) is bool
    assert type(truncated) is bool


def test_dict_agents(env_double_agent):
    env = DictAgents(env_double_agent)
    obs = env.reset()
