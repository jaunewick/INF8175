# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from custom_types import Direction
from pacman import GameState
from typing import Any, Tuple, List
import util

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self) -> Any:
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state: Any) -> bool:
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state: Any) -> List[Tuple[Any, Direction, int]]:
        """
        state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions: List[Direction]) -> int:
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem: SearchProblem) -> List[Direction]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem: SearchProblem) -> List[Direction]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 1 ICI
    '''
    state = problem.getStartState() # Initial state for graph search
    actions = []
    L = util.Stack() # initialize the stack structure for the graph search
    L.push((state, actions)) # stack initial state and actions tuple
    V = [] # Visited states  are stored in memory to avoid infinite loops
    while not L.isEmpty():
        state, actions = L.pop() # pop the last stacked tuple
        if state not in V: # if state is not already visited
            if problem.isGoalState(state): # check if state is goal state
                return actions
            else: # if not goal state
                successors = problem.getSuccessors(state) # get current node successors in graph
                for successor in successors: # for each successor in nodes successors list, evaluate
                    newState, direction, _ = successor # get successor state, direction and cost
                    if newState not in V: # visit successor if not already visited
                        newActions = list(actions)
                        newActions.append(direction)
                        L.push((newState, newActions))
                V.append(state)
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Direction]:
    """Search the shallowest nodes in the search tree first."""

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 2 ICI
    '''
    state = problem.getStartState()
    actions = []
    L = util.Queue()
    L.push((state, actions))
    V = []
    while not L.isEmpty():
        state, actions = L.pop()
        if state not in V:
            if problem.isGoalState(state):
                return actions
            else:
                successors = problem.getSuccessors(state)
                for successor in successors:
                    newState, direction, _ = successor
                    if newState not in V:
                        newActions = list(actions)
                        newActions.append(direction)
                        L.push((newState, newActions))
                V.append(state)
    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem) -> List[Direction]:
    """Search the node of least total cost first."""

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 3 ICI
    '''
    s = problem.getStartState()
    Actions = []
    g_s = 0
    L = util.PriorityQueue()
    L.push((s, Actions, g_s), g_s)
    V = []
    while not L.isEmpty():
        s, Actions, g_s = L.pop()
        if s not in V:
            if problem.isGoalState(s):
                return Actions
            else:
                C = problem.getSuccessors(s)
                for c in C:
                    state, direction, cost = c
                    if state not in V:
                        newActions = list(Actions)
                        newActions.append(direction)
                        g_c = float(g_s + cost)
                        L.update((state, newActions, g_c), g_c)
                V.append(s)
    util.raiseNotDefined()


def nullHeuristic(state: GameState, problem: SearchProblem = None) -> List[Direction]:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Direction]:
    """Search the node that has the lowest combined cost and heuristic first."""
    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 4 ICI
    '''
    s = problem.getStartState()
    L = util.PriorityQueue()
    g_s = 0
    h_s = float(heuristic(s, problem))
    f_s = g_s + h_s
    L.push((s, [], g_s), f_s)
    V = []
    while not L.isEmpty():
        s, Actions, g_s = L.pop()
        if s not in V:
            if problem.isGoalState(s):
                return Actions
            else:
                C = problem.getSuccessors(s)
                for c in C:
                    state, direction, cost = c
                    if state not in V:
                        newActions = list(Actions)
                        newActions.append(direction)
                        g_c = float(g_s + cost)
                        h_c = float(heuristic(state, problem))
                        f_c = g_c + h_c
                        L.update((state, newActions, g_c), f_c)
                V.append(s)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
