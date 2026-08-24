from tests.test_msc_env import record

from src.gnarl.envs.msc_env import MSCEnvironment
from src.gnarl.models.gnarl_msc import GNARLMSC


def test_actor_returns_one_logit_per_pdnar_set_and_scalar_value():
    env = MSCEnvironment(record())
    logits, value = GNARLMSC(hidden_dim=16, message_passing_rounds=1)(env)
    assert logits.shape == (3,)
    assert value.shape == ()
