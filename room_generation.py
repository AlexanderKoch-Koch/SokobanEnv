import os
import numpy as np


class RoomGenerator:

    def __init__(self, folder_path):
        self.files = sorted(os.listdir(folder_path))
        self.folder_path = folder_path
        print(self.files)
        self.dir = folder_path
        self.current_file_index = 0
        self.current_file = open(folder_path + self.files[self.current_file_index], mode="r")
        self.saved_room_fixed = None
        self.saved_room_state = None

    def generate(self, same=False):
        """
        generates new room matrix from file
        :param same:
        :return: 10x10 int matrix where entries denote object at the position[x, y]:
            0 -> wall
            1 -> empty space
            2 -> box goal position
            3 -> box current position(that means not in goal position
            4 -> box on goal
            5 -> player position

        , 10x10 matrix with entries 1 if goal position, list of length 2 with player start position
        """
        if same:
            if self.saved_room_state is not None and self.saved_room_fixed is not None:
                return np.copy(self.saved_room_fixed), np.copy(self.saved_room_state)

        room = np.zeros(shape=(10, 10), dtype=np.int)
        room_goal_positions = np.zeros(shape=(10, 10), dtype=np.int)

        line = self.__advance_to_next()
        player_position = None
        index = 0
        while line and line != "\n":
            if line.startswith("#"):
                for i, char in enumerate(line):
                    if char == "#":
                        # wall
                        room[index, i] = 0
                    elif char == " ":
                        # empty space
                        room[index, i] = 1
                    elif char == ".":
                        # goal position
                        room[index, i] = 2
                        room_goal_positions[index, i] = 1
                    elif char == "$":
                        # box
                        room[index, i] = 3
                    elif char == "@":
                        # player
                        room[index, i] = 5
                        player_position = (index, i)

                index += 1
            line = self.current_file.readline()

        self.saved_room_fixed = room
        self.saved_room_state = room_goal_positions
        assert player_position is not None, "room description does not contain player"
        return room, room_goal_positions, player_position

    def __advance_to_next(self):
        """
        reads file lines until next room begins
        :return: first line of new room
        """

        line = self.current_file.readline()
        while line and not line.startswith("#"):
            line = self.current_file.readline()

        if line == "":
            self.current_file_index += 1
            if self.current_file_index >= len(self.files):
                self.current_file_index = 0
            self.current_file = open(self.folder_path + self.files[self.current_file_index], mode="r")
            return self.__advance_to_next()

        return line