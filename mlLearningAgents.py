# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
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

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

# python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
# python pacman.py -p QLearnAgent -x 1000 -n 2010 -l smallGrid

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        
        self.state = state
        self.pos = state.getPacmanPosition()
        self.ghosts = state.getGhostPositions()
        self.food = state.getFood()
        self.score = state.getScore()
        self.walls = state.getWalls()
        self.legal = state.getLegalPacmanActions()

    def __hash__(self):
        """
        Returns a hash value for the GameStateFeatures object.

        Args:
            None

        Returns:
            int: The hash value of the object.
        """
        return hash((self.pos, self.walls, tuple(self.ghosts), self.food))

    def __eq__(self, other):
        """
        Checks if two GameStateFeatures objects are equal.

        Args:
            other (GameStateFeatures): Another GameStateFeatures object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        if other is None:
            return False

        # check if two GameStateFeatures objects are equal
        return (self.walls == other.walls and
                self.food == other.food and
                self.pos == other.pos and
                self.ghosts == other.ghosts)


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.1,
                 gamma: float = 0.8,
                 maxAttempts: int = 100,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # q dict for Q values for each (state, action)
        self.q = {}
        # counter dict for number of visits of (state, action) 
        self.counter = {}
        # prevState is the previous state
        self.prevState = None
        # prevAction is the previous action
        self.prevAction = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        food_reward = endState.getScore() - startState.getScore()
        return food_reward 

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        # Check if (state, action) pair exists in the Q-table
        q_value = self.q.get((state, action))
        
        # If (state, action) pair not found, initialize its value to 0.0
        if q_value is None:
            q_value = 0.0
            self.q[(state, action)] = q_value
        
        return q_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        # Get the legal Pacman actions for the given state
        legal_actions = state.state.getLegalPacmanActions()
        
        # Remove the STOP action from the legal actions if it exists
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)
        
        # If there are no legal actions, return 0
        if not legal_actions:
            return 0
        
        # Calculate the maximum Q-value among the legal actions
        max_q_value = max(self.getQValue(state, action) for action in legal_actions)
        
        return max_q_value
                

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        # main q learning equation implementation using equation
        q_val = self.getQValue(state, action)
        q_val_max = self.maxQValue(nextState)
        self.q[(state, action)] = q_val + self.alpha * (reward + self.gamma * q_val_max - q_val)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        # Get the current count for the (state, action) pair from the counter dictionary
        count = self.counter.get((state, action), 0)
        
        # Increment the count by 1
        count += 1
        
        # Update the counter dictionary with the new count for the (state, action) pair
        self.counter[(state, action)] = count

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        # Get the count for the (state, action) pair from the counter dictionary
        count = self.counter.get((state, action), 0)
        
        return count

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        # Check if the count is less than the maximum attempts
        if counts < self.maxAttempts:
            # Calculate the exploration value based on the count and maximum attempts
            exploration_value = utility * (self.maxAttempts - counts) / self.maxAttempts
        else:
            # If the count exceeds or equals the maximum attempts, use the utility value as is
            exploration_value = utility
        
        return exploration_value

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore())

        stateFeatures = GameStateFeatures(state)

        # Perform learning for (prevState, action) - Q(s, a)
        if self.prevState is not None:
            prevStateFeatures = GameStateFeatures(self.prevState)
            self.learn(prevStateFeatures, self.prevAction, self.computeReward(self.prevState, state), stateFeatures)

        # Choose the action to take using probability epsilon (a')
        if util.flipCoin(self.epsilon):
            # Explore
            utility = [self.getQValue(stateFeatures, a) for a in legal]
            counts = [self.getCount(stateFeatures, a) for a in legal]

            # Find explore values from utilities via exploration function
            explore_value = [self.explorationFn(u, c) for u, c in zip(utility, counts)]

            # Choose action with the maximum explore value
            max_index = explore_value.index(max(explore_value))
            action = legal[max_index]
        else:
            # Exploit
            # Choose action with the maximum Q-value
            q_values = [self.getQValue(stateFeatures, a) for a in legal]
            max_index = q_values.index(max(q_values))
            action = legal[max_index]

        # Update the count
        self.updateCount(stateFeatures, action)

        # Update the prevState and prevAction
        self.prevState = state
        self.prevAction = action

        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Update the final (state, action) Q-value bc fencepost error
        if self.prevState is not None:
            prevStateFeatures = GameStateFeatures(self.prevState)
            stateFeatures = GameStateFeatures(state)
            self.learn(prevStateFeatures, self.prevAction, self.computeReward(self.prevState, state), stateFeatures)

        # Reset the prevState and prevAction
        self.prevState = None
        self.prevAction = None

        # Update the count for the current state and STOP action
        self.updateCount(GameStateFeatures(state), Directions.STOP)

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
