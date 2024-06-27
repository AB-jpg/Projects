# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util


class map(object):
    # creates a grid for mapping the current state.
    # includes information about ghosts and food and their rewards
    # uses references from code provided in MapAgent

    # value definitions for parameter optimization
    ghost_r = -500
    food_r = 10
    capsule_r = 10
    no_reward = -0.4
    wall_r = -10

    def __init__(self, state):
        # initialise empty map
        cor_l = api.corners(state)
        for i in range(len(cor_l)):
            if cor_l[i][0] != 0:
                self.x_max = cor_l[i][0]
            if cor_l[i][1] != 0:
                self.y_max = cor_l[i][1]
        self.x_max += 1
        self.y_max += 1

        self.grid = [[self.no_reward for y in range(self.y_max + 1)] for x in range(self.x_max + 1)]

        # add walls reward
        self.wall_list = list(api.walls(state))

        for elem in self.wall_list:
            self.grid[elem[0]][elem[1]] = self.wall_r

    def updateMap(self, state):
        # updates current state to reward map
        # 1. by removing old food
        # 2. increase reward of ghost if edible and within 6 steps of time running out
        # 3. food loses value close to ghost
        # 4. make last food desirable
        # 5. for non edible ghosts, make pacman avoid them by making states around them have low rewards

        # remove old food
        for i in range(self.x_max):
            for j in range(self.y_max):
                if (i, j) not in self.wall_list:
                    self.grid[i][j] = self.no_reward

        self.ghost_list = api.ghostStatesWithTimes(state)

        # convert ghost to integer
        new_ghost_list = []
        for ghost in self.ghost_list:
            x = int(ghost[0][0])
            y = int(ghost[0][1])
            new_ghost_list.append([(x, y), ghost[1]])
        self.ghost_list = new_ghost_list

        pacman = api.whereAmI(state)
        # increase reward of ghost if edible and within 6 steps of time running out
        for ghost in self.ghost_list:
            self.grid[ghost[0][0]][ghost[0][1]] = self.ghost_r
            dist_p_to_g = util.manhattanDistance(pacman, ghost[0])
            if (ghost[1] - dist_p_to_g) > 6:
                self.grid[ghost[0][0]][ghost[0][1]] = self.food_r * 100

        # food loses value close to ghost
        self.food_list = api.food(state)
        for food in self.food_list:
            min_dist_f_to_g = [0]
            for ghost in self.ghost_list:
                # only if ghost not eatable
                if ghost[1] == 0:
                    dist_f_to_g = util.manhattanDistance(food, ghost[0])
                    min_dist_f_to_g.append(dist_f_to_g)
            food_after_g = self.food_r - (min(min_dist_f_to_g))*4
            self.grid[food[0]][food[1]] = food_after_g

        capsule_list = api.capsules(state)
        for capsule in capsule_list:
            self.grid[capsule[0]][capsule[1]] = self.capsule_r

        # make last food desirable
        if len(self.food_list) == 1:
            self.grid[self.food_list[0][0]][self.food_list[0][1]] = self.food_r * 2

        # for non edible ghosts, make pacman avoid them by making states around them have low rewards
        # for medium let anything 3 steps away be affected and for small anything 2 steps away

        # create ghost zone coordinates for non edible ghost
        ghost_zone_1 = []
        ghost_zone_2 = []
        ghost_zone_3 = []
        for ghost in self.ghost_list:
            if ghost[1] == 0:
                (x, y) = ghost[0]

                # 1 step away zone
                ghost_zone_1 = [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]
                ghost_zone_check = []
                for step in ghost_zone_1:
                    if step not in self.wall_list and step not in self.ghost_list:
                        ghost_zone_check.append(step)
                ghost_zone_1 = ghost_zone_check
                # Set rewards in the zone
                for (x, y) in ghost_zone_1:
                    if x in range(1, self.x_max - 1) and y in range(1, self.y_max - 1):
                        if self.grid[x][y] != self.wall_r:
                            self.grid[x][y] = self.ghost_r * 0.75

                # 2 step away zone
                for (x, y) in ghost_zone_1:
                    ghost_zone_2 = [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]
                    ghost_zone_check = []
                    for step in ghost_zone_2:
                        if step not in self.wall_list and step not in self.ghost_list:
                            ghost_zone_check.append(step)
                    ghost_zone_2 = ghost_zone_check
                # Set rewards in the zone
                for (x, y) in ghost_zone_2:
                    if x in range(1, self.x_max - 1) and y in range(1, self.y_max - 1):
                        if self.grid[x][y] != self.wall_r:
                            self.grid[x][y] = self.ghost_r * 0.5

                # 3 step away zone
                if self.y_max >= 9:
                    for (x, y) in ghost_zone_2:
                        ghost_zone_3 = [(x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)]
                        ghost_zone_check = []
                        for step in ghost_zone_3:
                            if step not in self.wall_list and step not in self.ghost_list:
                                ghost_zone_check.append(step)
                        ghost_zone_3 = ghost_zone_check
                    # Set rewards in the zone
                    for (x, y) in ghost_zone_3:
                        if x in range(1, self.x_max - 1) and y in range(1, self.y_max - 1):
                            if self.grid[x][y] != self.wall_r:
                                self.grid[x][y] = self.ghost_r * 0.25




class MDPAgent(Agent):
    # uses value iteration to get optimal decision by:
    # 1. remove old utility from 1st utility map
    # 2. remove old utility from 2nd utility map
    # 3. update reward map
    # 4. update utilities on utility map until almost converged, then copies last state into old map and calculates again
    # 5. get move with maximum utility from utility map and make that the final_move

    gamma = 0.5

    def __init__(self):
        print "game start"


    def registerInitialState(self, state):
        # initialise maps
        self.utility_map = map(state)
        self.utility_map_new = map(state)
        self.reward_map = map(state)

    def final(self, _):
        # delete maps
        self.legal = None
        self.utility_map = None
        self.utility_map_new = None
        self.reward_map = None

    def getAction(self, state):
        self.legal = set(api.legalActions(state))
        pacman = api.whereAmI(state)

        # remove old utility from 1st utility map
        for i in range(self.utility_map.x_max):
            for j in range(self.utility_map.y_max):
                if (i, j) not in self.utility_map.wall_list:
                    self.utility_map.grid[i][j] = 0
        # remove old utility from 2nd utility map
        for i in range(self.utility_map_new.x_max):
            for j in range(self.utility_map_new.y_max):
                if (i, j) not in self.utility_map_new.wall_list:
                    self.utility_map_new.grid[i][j] = 0
        # update reward map
        self.reward_map.updateMap(state)
        # update utilities on utility map until almost converged
        self.loop_convergence(25)
        # get move with maximum utility from utility map and make that the final_move
        move_utility = {}
        for move in self.legal:
            if move == Directions.NORTH:
                (dx, dy) = (0, 1)
            elif move == Directions.SOUTH:
                (dx, dy) = (0, -1)
            elif move == Directions.EAST:
                (dx, dy) = (1, 0)
            elif move == Directions.WEST:
                (dx, dy) = (-1, 0)
            else:
                (dx, dy) = (0, 0)
            (x, y) = (pacman[0] + dx, pacman[1] + dy)
            move_utility[move] = self.utility_map.grid[x][y]

        max_value = -100000
        final_move = None
        for k, v in move_utility.items():
            if v > max_value:
                max_value = v
                final_move = k

        return api.makeMove(final_move, list(self.legal))

    def deep_copy(self, map):
        return [e[:] for e in map]
    def loop_convergence(self, n):
        # loops over map and updates utilities, then copies last state into old map and calculates again
        for _ in range(n):
            for y in range(1, self.utility_map.y_max - 1):
                for x in range(1, self.utility_map.x_max - 1):
                    if self.utility_map.grid[x][y] != self.utility_map.wall_r:
                        self.utility_calc((x, y))
            self.utility_map.grid = self.deep_copy(self.utility_map_new.grid)


    def utility_calc(self, coord):
        # calculates max expected utility for a map coordinate
        utility_set = []
        #get utility for north
        utility = 0.8 * self.get_prev_util(Directions.NORTH, coord) \
                  + 0.1 * self.get_prev_util(Directions.WEST, coord) \
                  + 0.1 * self.get_prev_util(Directions.EAST, coord)
        utility_set.append(utility)
        # get utility for south
        utility = 0.8 * self.get_prev_util(Directions.SOUTH, coord) \
                  + 0.1 * self.get_prev_util(Directions.EAST, coord) \
                  + 0.1 * self.get_prev_util(Directions.WEST, coord)
        utility_set.append(utility)
        # get utility for east
        utility = 0.8 * self.get_prev_util(Directions.EAST, coord) \
                  + 0.1 * self.get_prev_util(Directions.NORTH, coord) \
                  + 0.1 * self.get_prev_util(Directions.SOUTH, coord)
        utility_set.append(utility)
        # get utility for west
        utility = 0.8 * self.get_prev_util(Directions.WEST, coord) \
                  + 0.1 * self.get_prev_util(Directions.NORTH, coord) \
                  + 0.1 * self.get_prev_util(Directions.SOUTH, coord)
        utility_set.append(utility)
        # get utility for stop
        utility = 1 * self.get_prev_util(Directions.STOP, coord)
        utility_set.append(utility)

        (x, y) = coord
        utility = self.reward_map.grid[x][y] + self.gamma * max(utility_set)
        self.utility_map_new.grid[x][y] = utility

    def get_prev_util(self, move, coord):
        #gets prev utilities of future moves from a coord
        if move == Directions.NORTH:
            offset = (0, 1)
        elif move == Directions.SOUTH:
            offset = (0, -1)
        elif move == Directions.EAST:
            offset = (1, 0)
        elif move == Directions.WEST:
            offset = (-1, 0)
        else:
            offset = (0, 0)

        # future move
        new_coord = (coord[0] + offset[0], coord[1] + offset[1])

        if self.utility_map.grid[new_coord[0]][new_coord[1]] != self.utility_map.wall_r:
            (x, y) = new_coord
        else:
            (x, y) = coord

        return self.utility_map.grid[x][y]

