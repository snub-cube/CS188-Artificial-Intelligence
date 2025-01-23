# multiAgents.py
# --------------
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


from util import manhattanDistance
import numpy as np
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minFoodToPac = 1000000
        for pos in newFood.asList():
            minFoodToPac = min(manhattanDistance(newPos, pos), minFoodToPac)

        minGhostToPac = 1000000
        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0:
                minGhostToPac = min(manhattanDistance(newPos, ghostState.getPosition()), minGhostToPac)

        return successorGameState.getScore() + 10/minFoodToPac - 0.1/(minGhostToPac-0.9)

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        return self.maxValue(gameState, 0, 0, True)

    def maxValue(self, state, depth, agentIndex, getAction):
        value = -np.inf
        if state.isWin() or state.isLose() or self.depth == depth:
            return self.evaluationFunction(state)

        for action in state.getLegalActions(agentIndex):
            nextState = state.generateSuccessor(agentIndex, action)
            temp = self.minValue(nextState, depth, agentIndex + 1)
            if temp > value:
                value = temp
                bestAction = action

        if getAction:
            return bestAction

        return value

    def minValue(self, state, depth, agentIndex):
        value = np.inf
        if state.isWin() or state.isLose() or self.depth == depth:
            return self.evaluationFunction(state)

        numGhosts = state.getNumAgents() - 1
        for action in state.getLegalActions(agentIndex):
            nextState = state.generateSuccessor(agentIndex, action)
            if agentIndex == numGhosts:
                temp = self.maxValue(nextState, depth + 1, 0, False)
            else:
                temp = self.minValue(nextState, depth, agentIndex + 1)

            value = min(temp,value)

        return value

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -np.inf
        beta = np.inf
        return self.maxValue(gameState, 0, 0, True, alpha, beta)
        util.raiseNotDefined()

    def maxValue(self, state, depth, agentIndex, getAction,alpha, beta):
        value = -np.inf
        if  state.isWin() or state.isLose() or self.depth == depth:
            return self.evaluationFunction(state)

        for action in state.getLegalActions(agentIndex):
            nextState = state.generateSuccessor(agentIndex, action)
            temp = self.minValue(nextState, depth, agentIndex + 1,alpha, beta)
            if temp > value:
                value = temp
                bestAction = action

            if value > beta:
                return value
            alpha = max(alpha,value)

        if getAction:
            return bestAction

        return value

    def minValue(self, state, depth, agentIndex, alpha, beta):
        value = np.inf
        if  state.isWin() or state.isLose() or self.depth == depth:
            return self.evaluationFunction(state)

        numGhosts = state.getNumAgents() - 1
        for action in state.getLegalActions(agentIndex):
            nextState = state.generateSuccessor(agentIndex, action)
            if agentIndex == numGhosts:
                temp = self.maxValue(nextState, depth + 1, 0, False, alpha, beta)
            else:
                temp = self.minValue(nextState, depth, agentIndex + 1, alpha, beta)

            value = min(temp,value)
            if value < alpha:
                return value
            beta = min(beta,value)

        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return self.maxValue(gameState, 0, 0, True)

    def maxValue(self, state, depth, agentIndex, getAction):
        value = -np.inf
        if state.isWin() or state.isLose() or self.depth == depth:
            return self.evaluationFunction(state)

        for action in state.getLegalActions(agentIndex):
            nextState = state.generateSuccessor(agentIndex, action)
            temp = self.expValue(nextState, depth, agentIndex + 1)
            if temp > value:
                value = temp
                bestAction = action
        if getAction:
            return bestAction
        return value

    def expValue(self, state, depth, agentIndex):
        value = 0
        if state.isWin() or state.isLose() or self.depth == depth:
            return self.evaluationFunction(state)

        numGhosts = state.getNumAgents() - 1
        prob = 1 / len(state.getLegalActions(agentIndex))
        for action in state.getLegalActions(agentIndex):
            nextState = state.generateSuccessor(agentIndex, action)
            if agentIndex == numGhosts:
                value += prob * self.maxValue(nextState, depth + 1, 0, False)
            else:
                value += prob * self.expValue(nextState, depth, agentIndex + 1)

        return value

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: return extreme values in case of win/lose state. gather state info
    and use it to calculate parameters used in evaluating function.
    (closest food, ghost, sum of scared timer).
    """

    #Win/Lose cases
    if currentGameState.isWin():
        return np.inf
    if currentGameState.isLose():
        return -np.inf

    #State info
    pacPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ghostTimers = [ghostState.scaredTimer for ghostState in ghostStates]

    #vars used in eval func
    minFoodToPac = 1000000
    for pos in foodList:
        minFoodToPac = min(manhattanDistance(pacPos, pos), minFoodToPac)

    minGhostToPac = 1000000
    for ghostState in ghostStates:
        if ghostState.scaredTimer == 0:
            minGhostToPac = min(manhattanDistance(pacPos, ghostState.getPosition()), minGhostToPac)

    scaredTimer = sum(ghostTimers)
    return currentGameState.getScore() - minFoodToPac - 0.9 + minGhostToPac - 0.9 + scaredTimer

# Abbreviation
better = betterEvaluationFunction