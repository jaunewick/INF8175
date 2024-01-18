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
    s = problem.getStartState()
    L = util.Stack()
    L.push((s, []))
    V = set()
    while not L.isEmpty():
        s = L.pop()
        if s[0] not in V:
            if problem.isGoalState(s[0]):
                return s[1]
            else:
                C = problem.getSuccessors(s[0])
                for c in C:
                    if c[0] not in V:
                        directions = list(s[1])
                        directions.append(c[1])
                        L.push((c[0], directions))
                V.add(tuple(s[0]))
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Direction]:
    """Search the shallowest nodes in the search tree first."""

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 2 ICI
    '''
    s = problem.getStartState()
    L = util.Queue()
    L.push((s, []))
    V = set()
    while not L.isEmpty():
        s = L.pop()
        if s[0] not in V:
            if problem.isGoalState(s[0]):
                return s[1]
            else:
                C = problem.getSuccessors(s[0])
                for c in C:
                    if c[0] not in V:
                        directions = list(s[1])
                        directions.append(c[1])
                        L.push((c[0], directions))
                V.add(tuple(s[0]))
    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem) -> List[Direction]:
    """Search the node of least total cost first."""

    '''
        INSÉREZ VOTRE SOLUTION À LA QUESTION 3 ICI
    '''
    s = problem.getStartState()
    L = util.PriorityQueue()
    L.push((s, [], 0), 0)
    V = set()
    while not L.isEmpty():
        s = L.pop()
        if s[0] not in V:
            if problem.isGoalState(s[0]):
                return s[1]
            else:
                C = problem.getSuccessors(s[0])
                for c in C:
                    if c[0] not in V:
                        directions = list(s[1])
                        directions.append(c[1])
                        g = float(s[2] + c[2])
                        L.update((c[0], directions, g), g)
                V.add(tuple(s[0]))
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
    h = float(heuristic(s,problem))
    L.push((s, [], 0), h)
    V = set()
    while not L.isEmpty():
        s = L.pop()
        if s[0] not in V:
            if problem.isGoalState(s[0]):
                return s[1]
            else:
                C = problem.getSuccessors(s[0])
                for c in C:
                    if c[0] not in V:
                        directions = list(s[1])
                        directions.append(c[1])
                        g = float(s[2] + c[2])
                        h = float(heuristic(c[0], problem))
                        f = g + h
                        L.update((c[0], directions, g), f)
                V.add(tuple(s[0]))
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
