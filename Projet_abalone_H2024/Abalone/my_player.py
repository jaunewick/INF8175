"""
    Binome 1 : Renel Lherisson (2089776)
    Binome 2 : Daniel Giao (2120073)
"""
from player_abalone import PlayerAbalone
from game_state_abalone import GameStateAbalone
from seahorse.game.action import Action
from seahorse.game.game_state import GameState
from typing import Tuple
from player_abalone import PlayerAbalone


class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game.

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(self, piece_type: str, name: str = "bob", time_limit: float = 60*15, *args) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name, time_limit, *args)

    def compute_action(self, current_state: GameStateAbalone, **kwargs) -> Action:
        """
        Function to implement the logic of the player.

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        return self.alpha_beta_search(current_state)

    def alpha_beta_search(self, state: GameStateAbalone, max_depth=3) -> Action:
        """
        Function to implement the alpha-beta search algorithm.

        Args:
            state (GameStateAbalone): Current game state representation
            max_depth (int): Maximum depth of the search tree

        Returns:
            Action: selected feasible action
        """
        # Initialize alpha and beta values
        alpha = float('-inf')
        beta = float('inf')
        # Get the best action possible and its corresponding score
        action, _ = self.max_value(state, alpha, beta, 0, max_depth)
        return action

    def max_value(self, state, alpha, beta, depth=0, max_depth=3):
        """
        Function to implement the max value logic for alpha-beta pruning.

        Args:
            state (GameStateAbalone): Current game state representation
            alpha (float): Alpha value for alpha-beta pruning
            beta (float): Beta value for alpha-beta pruning
            depth (int): Current depth of the search tree
            max_depth (int): Maximum depth of the search tree

        Returns:
            Tuple: Tuple containing the best action and its corresponding value
        """
        # Check if the game is done or the maximum depth is reached
        if state.is_done() or depth == max_depth:
            # Return None and the heuristic value corresponding to the state
            return None, self.apply_heuristic(state) or float('-inf')
        # Initialize v to the value of negative infinity
        v = float('-inf')
        # Initialize the best action to None as no action is selected yet
        best_action = None
        # Loop through all possible actions from the current state
        for action in state.generate_possible_actions():
            # Get the next game state possible and the value of the min value function
            _, value = self.min_value(
                action.get_next_game_state(), alpha, beta, depth+1, max_depth)
            # test if the value is not None and greater than the current v value
            if value is not None and value > v:
                # If the value is greater than the current v value, update v and the best action and alpha for max node
                v = value
                best_action = action
                alpha = max(alpha, v)
            # If v is greater than or equal to beta, return the best action and v
            if v >= beta:
                return best_action, v
        return best_action, v

    def min_value(self, state, alpha, beta, depth=0, max_depth=3):
        """
        Function to implement the min value logic for alpha-beta pruning.

        Args:
            state (GameStateAbalone): Current game state representation
            alpha (float): Alpha value for alpha-beta pruning
            beta (float): Beta value for alpha-beta pruning
            depth (int): Current depth of the search tree
            max_depth (int): Maximum depth of the search tree

        Returns:
            Tuple: Tuple containing the best action and its corresponding value
        """
        # Check if the game is done or the maximum depth is reached
        if state.is_done() or depth == max_depth:
            # Return None and the heuristic value corresponding to the state
            return None, self.apply_heuristic(state) or float('inf')
        # Initialize v to the value of positive infinity
        v = float('inf')
        # Initialize the best action to None as no action is selected yet
        best_action = None
        # Loop through all possible actions from the current state
        for action in state.generate_possible_actions():
            # Get the next game state possible and the value of the max value function
            _, value = self.max_value(
                # Get the next game state possible and the value of the max value function
                action.get_next_game_state(), alpha, beta, depth+1, max_depth)
            # test if the value is not None and less than the current v value, update v and the best action and beta for min node
            if value is not None and value < v:
                v = value
                best_action = action
                beta = min(beta, v)
            # If v is less than or equal to alpha, return the best action and v
            if v <= alpha:
                return best_action, v
        return best_action, v

    def apply_heuristic(self, state: GameStateAbalone) -> float:
        """
        Function to implement the apply_heuristic function.

        Args:
            state (GameStateAbalone): Current game state representation

        Returns:
            float: apply_heuristic value
        """
        # Initialize the number of marbles for the player and the opponent
        my_marbles, opponent_marbles = 0, 0
        # Initialize the center control and formation strength score to 0
        center_control, formation_strength_score = 0, 0

        # Loop through all the positions and pieces in the game state
        for position, piece in state.get_rep().get_env().items():
            # Check if the piece is for the "self" player and compute the score applying the heuristic
            if piece.get_owner_id() == self.get_id():
                # Increment the number of marbles for the player
                my_marbles += 1
                # Check if the position is in the center and increment the center control
                if self.is_center(position):
                    center_control += 1
                # Increment the formation strength score
                formation_strength_score += self.formation_strength(
                    position, state)
            # If the piece is for the opponent, increment the number of marbles for the opponent
            else:
                opponent_marbles += 1

        # Compute the score applying the heuristic function
        score = (my_marbles - opponent_marbles) + 0.5 * \
            center_control + 0.2 * formation_strength_score
        return score

    def is_center(self, position: Tuple[int, int]) -> bool:
        """
        Function to check if the given position is in the center of the board.

        Args:
            position (Tuple[int, int]): Position to check

        Returns:
            bool: True if the position is in the center, False otherwise
        """
        # hard coded best center positions for the board defined experimentally
        center_positions = [(7, 5), (9, 5), (10, 4),
                            (8, 4), (9, 3), (7, 3), (6, 4)]
        return position in center_positions

    def formation_strength(self, position: Tuple[int, int], state: GameStateAbalone) -> int:
        """
        Function to calculate the formation strength of the player.
        Args:
            position (Tuple[int, int]): position of the player
            state (GameStateAbalone): current game state representation

        Returns:
            int: strength of the formation
        """
        strength = 0
        # Get the neighbors items from the current position
        neighbors = state.get_neighbours(position[0], position[1])
        # Loop through all the neighbors and check if the neighbor is owned by the player
        for direction, neighbor_pos in neighbors.items():
            neighbor = state.get_rep().get_env().get(neighbor_pos[1])
            if neighbor and neighbor.get_owner_id() == self.get_id():
                # If the neighbor is owned by the player, increment the strength by 1
                strength += 1
        # Return the strength of the formation
        return strength
