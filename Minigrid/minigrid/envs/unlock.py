from __future__ import annotations

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Lava
from minigrid.minigrid_env import MiniGridEnv

class UnlockEnv(MiniGridEnv):

    """
    ## Description

    The agent has to open a locked door. This environment can be solved without
    relying on language.

    ## Mission Space

    "open the door"

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Unused                    |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent opens the door.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-Unlock-v0`

    """

    def __init__(self, size, obstacle_type=Lava, max_steps: int | None = None, **kwargs):
        self.obstacle_type= obstacle_type
        self.size = 9

        if obstacle_type==Lava:
            mission_space = MissionSpace(mission_func=self._gen_mission_lava)
        else:
            mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 8 * size**2

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission_lava():
        return "avoid the lava and get to the green goal square"

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        assert width==9 and height==9

        self.grid= Grid(width,height)
        self.grid.wall_rect(0, 0, width, height)
        # Make sure the two rooms are directly connected by a locked door
        
        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0        
        
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # Generate and store random gap position
        self.gap_pos = np.array(
            (
                self._rand_int(2, width - 2),
                self._rand_int(1, height - 1),
            )
        )        

        # Place obstacle walls
        self.grid.vert_wall(self.gap_pos[0], 1, height - 2, obj_type=self.obstacle_type)
        self.grid.horz_wall(length=3,x=self._rand_int(2,5), y=self._rand_int(2,6), obj_type=self.obstacle_type)        

        # Put a hole in the wall
        self.grid.set(*self.gap_pos, None)

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )




#    def step(self, action):
#        obs, reward, terminated, truncated, info = super().step(action)
#        
#        if self.agent_sees_trigger()==True:
#            print("Get to the lava")
#            if distance between agent and centre of trigger decreases, then reward=self._reward()
#            if self.agent_steps_towards_trigger()==True:
#                print("It is stepping towards the trigger, give reward!!!!")
#                reward= self._reward_trigger()
#                print(reward)



#        return obs, reward, terminated, truncated, info
