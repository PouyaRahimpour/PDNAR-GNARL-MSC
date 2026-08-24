from tests.test_msc_env import record

from src.gnarl.envs.msc_env import MSCEnvironment


def test_selected_set_cannot_be_selected_twice():
    env = MSCEnvironment(record())
    env.step(0)
    assert env.action_mask().tolist() == [False, True, True]
    try:
        env.step(0)
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated set selection must be rejected")
