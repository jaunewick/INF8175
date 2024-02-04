# Nom                 Matricule
# Daniel Giao         xxxxxxxx
# Renel Lherisson     xxxxxxxx


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
    # Get initial state for graph search
    state = problem.getStartState()
    actions = []
    # initialize the stack structure for the graph search (LIFO)
    L = util.Stack()
    # stack initial state and actions tuple
    L.push((state, actions))
    # All Visited states  are stored in memory to avoid infinite loops in revisiting a visited state
    V = []
    # loop on states and successors to find goal state
    # avoid visiting already visited states
    while not L.isEmpty():
        # pop two components of the last stacked tuple from the stack
        state, actions = L.pop()
        # if state is not already visited
        if state not in V:
            # check if current state is goal state and return its actions
            if problem.isGoalState(state):
                return actions
            # if not goal state
            else:
                # get current node successors in graph
                successors = problem.getSuccessors(state)
                # for each successor in nodes successors list, evaluate
                for successor in successors:
                    # get successor state, direction and cost from the successor tuple
                    newState, direction, _ = successor
                    # visit successor if not already visited
                    if newState not in V:
                        # update actions list indicating the path to the successor from the initial state
                        newActions = list(actions)
                        newActions.append(direction)
                        # push the new state and its actions to the stack
                        L.push((newState, newActions))
                # mark the current state as visited
                V.append(state)
    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem) -> List[Direction]:
    """Search the shallowest nodes in the search tree first."""

    # Get initial state for graph search
    state = problem.getStartState()
    actions = []
    # initialize the Queue structure for the graph search (FIFO)
    L = util.Queue()
    # add initial state and actions tuple to the queue
    L.push((state, actions))
    # All Visited states  are stored in memory to avoid infinite loops in revisiting a visited state
    V = []
    # loop on states and successors to find goal state
    # avoid visiting already visited states
    while not L.isEmpty():
        # pop two components of the last queued tuple from the queue
        state, actions = L.pop()
        # if state is not already visited
        if state not in V:
            # check if current state is goal state and return its actions
            if problem.isGoalState(state):
                return actions
            # if not goal state
            else:
                # get current node successors in graph
                successors = problem.getSuccessors(state)
                # for each successor in nodes successors list, evaluate
                for successor in successors:
                    # get successor state, direction and cost from the successor tuple
                    newState, direction, _ = successor
                    # visit successor if not already visited
                    if newState not in V:
                        # update actions list indicating the path to the successor from the initial state
                        newActions = list(actions)
                        newActions.append(direction)
                        # push the new state and its actions to the queue
                        L.push((newState, newActions))
                # mark the current state as visited
                V.append(state)
    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem) -> List[Direction]:
    """Search the node of least total cost first."""
    # Get initial state for graph search
    s = problem.getStartState()
    actions = []
    # cost from the initial state to the current state (initially 0)
    g_s = 0
    # initialize the PriorityQueue structure for the graph search
    L = util.PriorityQueue()
    # add initial state, actions and cost tuple to the priority queue
    L.push((s, actions, g_s), g_s)
    # All Visited states  are stored in memory to avoid infinite loops in revisiting a visited state
    V = []
    # loop on states and successors to find goal state
    while not L.isEmpty():
        # pop three components tuple of the priority queue (state, actions, cost) based on the least cost
        s, actions, g_s = L.pop()
        # if state is not already visited
        if s not in V:
            # check if current state is goal state and return its actions
            if problem.isGoalState(s):
                return actions
            # if not goal state
            else:
                # get current node successors in graph
                C = problem.getSuccessors(s)
                # for each successor in nodes successors list, evaluate
                for c in C:
                    # get successor state, direction and cost from the successor tuple
                    state, direction, cost = c
                    if state not in V:
                        # update actions list indicating the path to the successor from the initial state
                        newActions = list(actions)
                        # add the direction to the actions list
                        newActions.append(direction)
                        # update the cost from the initial state to the current state
                        g_c = float(g_s + cost)
                        # push the new state, its actions and cost to the priority queue
                        L.update((state, newActions, g_c), g_c)
                # mark the current state as visited
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

    # Get initial state for graph search
    s = problem.getStartState()
    # initialize the PriorityQueue structure for the graph search
    L = util.PriorityQueue()
    # cost from the initial state to the current state (initially 0)
    g_s = 0
    # cost from current node to final node
    h_s = float(heuristic(s, problem))
    # sum of the two costs
    f_s = g_s + h_s
    # add tuple to the priority queue
    L.push((s, [], g_s), f_s)
    # All Visited states  are stored in memory to avoid infinite loops in revisiting a visited state
    V = []
    # loop on state and successors to find goal state
    while not L.isEmpty():
        # pop three components tuple of the priority queue based on the least total cost
        s, actions, g_s = L.pop()
        # if state is not already visited
        if s not in V:
            # check if current state is goal state and return its actions
            if problem.isGoalState(s):
                return actions
            # if not goal state
            else:
                # get current node successors in graph
                C = problem.getSuccessors(s)
                # for each successor in nodes successors list, evaluate
                for c in C:
                    # get successor state, direction and cost from the successor tuple
                    state, direction, cost = c
                    if state not in V:
                        # update actions list indicating the path to the successor from the initial state
                        newActions = list(actions)
                        newActions.append(direction)
                        # update the cost from the initial state to the current state and the cost from current node to final node
                        g_c = float(g_s + cost)
                        h_c = float(heuristic(state, problem))
                        f_c = g_c + h_c
                        # push the new state, its actions and cost to the priority queue
                        L.update((state, newActions, g_c), f_c)
                # mark the current state as visited
                V.append(s)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
