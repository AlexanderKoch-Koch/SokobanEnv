import numpy as np
from room_generation import RoomGenerator
import gym
import render_utils
from PIL import Image

try:
    from gym.envs.classic_control import rendering
except:
    pass


class SokobanEnv:

    def __init__(self, puzzles_folder, max_steps=100):
        self.generator = RoomGenerator(puzzles_folder)
        self.actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self.room = None
        self.goal_positions = None
        self.player_position = None
        self.num_finished_boxes = 0
        self.current_step = 0
        self.max_steps = max_steps
        self.num_boxes = None

        self.reward_step = -0.1
        self.reward_box_on_goal = 1.0
        self.reward_box_moved_off_goal = -1.0
        self.reward_solved = 10

        self.viewer = None

    def reset(self, normalized=True):
        """
        resets the environment and starts a new game
        :return: first observation of new game
        """
        self.room, self.goal_positions, self.player_position = self.generator.generate()
        self.num_finished_boxes = 0
        self.current_step = 0
        self.num_boxes = len(self.goal_positions)

        state = self.room
        if normalized:
            state = np.subtract(np.divide(self.room, 5), 0.5)

        return state

    def step(self, action, normalized=True, tiny_observation=True):
        """
        executes action n game
        :param action:
        :return: new observation, reward, isDone, info
        """
        new_position = list(self.player_position)
        new_position[0] += self.actions[action][0]
        new_position[1] += self.actions[action][1]
        new_position = tuple(new_position)
        reward = self.reward_step
        is_done = False
        x = self.room[new_position]
        # check conditions where player can move
        if self.room[new_position] == 1 or self.room[new_position] == 2:
            # empty space or empty goal position -> change player position
            self.room[self.player_position] = 2 if self.goal_positions[self.player_position] else 1
            self.room[new_position] = 5
            self.player_position = new_position
        elif self.room[new_position] == 3 or self.room[new_position] == 4:
            # box -> check if it can be moved
            critical_position = list(new_position)
            critical_position[0] += self.actions[action][0]
            critical_position[1] += self.actions[action][1]
            critical_position = tuple(critical_position)
            if self.room[critical_position] == 1:
                # box can be moved on empty position
                if self.goal_positions[new_position]:
                    # box moved away from goal state
                    reward = self.reward_box_moved_off_goal
                    self.num_finished_boxes -= 1

                self.room[critical_position] = 3
                self.room[new_position] = 5
                self.room[self.player_position] = 2 if self.goal_positions[self.player_position] else 1
                self.player_position = new_position
            elif self.room[critical_position] == 2:
                # box can be moved on goal position
                self.num_finished_boxes += 1
                self.room[critical_position] = 4
                self.room[new_position] = 5
                self.room[self.player_position] = 2 if self.goal_positions[self.player_position] else 1
                self.player_position = new_position

                self.num_finished_boxes += 1
                reward = self.reward_box_on_goal
                if self.num_finished_boxes == self.num_boxes:
                    is_done = True
                    reward = self.reward_solved

        self.current_step += 1
        if self.current_step >= self.max_steps:
            is_done = True

        if tiny_observation:
            state = self.room
            if normalized:
                state = np.subtract(np.divide(self.room, 5), 0.5)
        else:
            state = render_utils.room_to_rgb(self.room, goal_positions=self.goal_positions)
            if normalized:
                # reduce size from 160x160x3 to 80x80x3
                state = state[::2, ::2, :]
                # state = misc.imresize(state, (80, 80), "bicubic")
                # normalize to [0; 1]
                state = state / 255

        return state, reward, is_done, None

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        img = render_utils.room_to_rgb(self.room, goal_positions=self.goal_positions)
        self.viewer.imshow(img)
