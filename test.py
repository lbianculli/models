import numpy as np
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv
from pysc2.env import environment
from pysc2.env import sc2_env
from pysc2.lib import features, actions

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index

from common import common


def worker(remote, map_name, nscripts, i):

  agent_format = sc2_env.AgentInterfaceFormat(
      feature_dimensions=sc2_env.Dimensions(
          screen=(32,32),
          minimap=(32,32)
      )
  )

  with sc2_env.SC2Env(
      agent_interface_format=[agent_format],
      map_name=map_name,
      step_mul=2) as env:
    available_actions = []
    result = None
    group_list = []
    xy_per_marine = {}
    while True:
      cmd, data = remote.recv()
      if cmd == 'step':
        reward = 0

        if len(group_list) == 0 or common.check_group_list(env, result):
          print("init group list")
          result, xy_per_marine = common.init(env, result)
          group_list = common.update_group_list(result)

        action1 = data[0][0]
        action2 = data[0][1]
        # func = actions.FUNCTIONS[action1[0]]
        # print("agent(",i," ) action : ", action1, " func : ", func)
        func = actions.FUNCTIONS[action2[0]]
        # print("agent(",i," ) action : ", action2, " func : ", func)


        result = env.step(actions=[action1])
        reward += result[0].reward
        done = result[0].step_type == environment.StepType.LAST

