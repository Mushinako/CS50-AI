"""
Tic Tac Toe Player
"""

from math import inf
from collections import Counter
from typing import Iterable, List, Literal, Optional, Set, Tuple

X = "X"
O = "O"
EMPTY = None

WINNER_SCORE_MAP = {
    X: 1,
    O: -1,
    EMPTY: 0,
}

# Type hints
Player = Literal["X", "O"]

Play = Literal["X", "O", None]
Board = List[List[Play]]

Index = Literal[0, 1, 2]
Coord = Tuple[Index, Index]

Score = Literal[-1, 0, 1]


def flattern(board: Board) -> List[Play]:
    """
    Flatten the board into a list.

    Args:
        board {Board}: The board (2D list)

    Returns:
        {list[Play]}: List of all 9 entries
    """
    return [entry for row in board for entry in row]


def initial_state() -> Board:
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def is_empty(cell: Play) -> bool:
    """
    Helper function. Check if cell is empty

    Args:
        cell {Play}: The cell value

    Returns:
        {bool}: Whether a cell is empty
    """
    if EMPTY is None:
        return cell is None
    else:
        return cell == EMPTY


def player(board: Board) -> Player:
    """
    Returns player who has the next turn on a board.

    Args:
        board {Board}: The board (2D list)

    Returns:
        {Player}: The next player to play

    Raises:
        {ValueError}: Invalid board
    """
    flattened = flattern(board)
    counts = Counter(flattened)
    diff = counts[X] - counts[O]
    # Same number of entries, next turn is X
    if not diff:
        return X
    # X has 1 more entry, next turn is O
    elif diff == 1:
        return O
    # Invalid board
    else:
        raise ValueError(f"Invalid board: {board=}")


def actions(board: Board) -> Set[Coord]:
    """
    Returns set of all possible actions (i, j) available on the board.

    Args:
        board {Board}: The board (2D list)

    Returns:
        {set[Coord]}: Set of free cells
    """
    return {(i, j) for i in range(3) for j in range(3) if is_empty(board[i][j])}


def result(board: Board, action: Coord) -> Board:
    """
    Returns the board that results from making move (i, j) on the board.

    Args:
        board  {Board}: The board (2D list)
        action {Coord}: The coord to place the next move

    Returns:
        {Board}: New board

    Raises:
        {IndexError}: Invalid index
        {ValueError}: Cell occupied
    """
    r, c = action
    # Check cell validity
    try:
        cell = board[r][c]
    except IndexError as err:
        raise IndexError(f"Invalid index: {board=} {action=}") from err
    if not is_empty(cell):
        raise ValueError(f"Cell occupied: {board=} {action=} {EMPTY=}")
    # Get next player
    next_player = player(board)
    # Copy board
    new_board: Board = [[cell for cell in row] for row in board]
    # Place move
    new_board[r][c] = next_player
    return new_board


def winner_of_line(line: Iterable[Play]) -> Optional[Player]:
    """
    Helper function of `winner()`. Checks one row/column/diagonal

    Args:
        line {Iterable[Play]}: A row/column/diagonal

    Returns:
        {Optional[Player]}: Winner of the line, or `None` if there isn't one
    """
    line_uniq = set(line)
    # Different elements in the list, can't have winner
    if len(line_uniq) != 1:
        return None
    # I'd like to think that there's a `,=` operator
    winner, = line_uniq
    return winner


def winner(board: Board) -> Optional[Player]:
    """
    Returns the winner of the game, if there is one.

    Args:
        board {Board}: The board (2D list)

    Returns:
        {Optional[Player]}: The winner, or `None` if there isn't one
    """
    # Check each row
    for row in board:
        row_winner = winner_of_line(row)
        if not is_empty(row_winner):
            return row_winner
    # Check each col
    for col in zip(*board):
        col_winner = winner_of_line(col)
        if not is_empty(col_winner):
            return col_winner
    # Check main diagonal
    main_diag = [board[i][i] for i in range(3)]
    main_diag_winner = winner_of_line(main_diag)
    if not is_empty(main_diag_winner):
        return main_diag_winner
    # Check antidiagonal
    antidiag = [board[i][2-i] for i in range(3)]
    antidiag_winner = winner_of_line(antidiag)
    if not is_empty(antidiag_winner):
        return antidiag_winner
    # No success
    return None


def terminal(board: Board) -> bool:
    """
    Returns True if game is over, False otherwise.

    Args:
        board {Board}: The board (2D list)

    Returns:
        {bool}: Whether the game is over
    """
    # Board if filled or a player has won
    return not actions(board) or winner(board) is not None


def utility(board: Board) -> Score:
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.

    Args:
        board {Board}: The board (2D list)

    Returns:
        {Score}: The score of the game
    """
    game_winner = winner(board)
    return WINNER_SCORE_MAP[game_winner]


def minimax_with_score(board: Board) -> Optional[Tuple[Coord, Score]]:
    """
    Returns the optimal action and score for the current player on the board.

    Args:
        board {Board}: The board (2D list)

    Returns:
        {Coord}: The optimal place to make the next move
    """
    # Terminal board
    if terminal(board):
        return None
    curr_player = player(board)
    potential_moves = sorted(actions(board))
    # Max player
    if WINNER_SCORE_MAP[curr_player] == 1:
        max_move = None
        max_score = -inf
        for move in potential_moves:
            new_board = result(board, move)
            solution = minimax_with_score(new_board)
            # Game ended
            if solution is None:
                score = utility(new_board)
                if score == 1:
                    return move, 1
                elif score > max_score:
                    max_move = move
                    max_score = score
            # Game continue
            else:
                score = solution[1]
                if score == 1:
                    return move, 1
                elif score > max_score:
                    max_move = move
                    max_score = score
        return max_move, max_score
    # Min player
    if WINNER_SCORE_MAP[curr_player] == -1:
        min_move = None
        min_score = inf
        for move in potential_moves:
            new_board = result(board, move)
            solution = minimax_with_score(new_board)
            # Game ended
            if solution is None:
                score = utility(new_board)
                if score == -1:
                    return move, -1
                elif score < min_score:
                    min_move = move
                    min_score = score
            # Game continue
            else:
                score = solution[1]
                if score == -1:
                    return move, -1
                elif score < min_score:
                    min_move = move
                    min_score = score
        return min_move, min_score
    raise ValueError(f"Invalid player: {board=} {curr_player=}")


def minimax(board: Board) -> Coord:
    """
    Returns the optimal action for the current player on the board.

    Args:
        board {Board}: The board (2D list)

    Returns:
        {Coord}: The optimal place to make the next move
    """
    solution = minimax_with_score(board)
    return None if solution is None else solution[0]
