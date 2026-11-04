"""Environment registry for uncertain_worms.

Three families ship here: :mod:`~uncertain_worms.environments.minigrid`,
:mod:`~uncertain_worms.environments.blocksworld` and
:mod:`~uncertain_worms.environments.popgym`. Only MiniGrid is star-imported
below, because its environments are registered with gym by import side
effect; the other two are reached through the full path a configuration
gives, so importing them here would only slow every start-up down.

Earlier alternatives (rocksample, tiger, a real-robot integration) are not
part of this release.
"""

from .minigrid import *  # noqa: F401,F403
