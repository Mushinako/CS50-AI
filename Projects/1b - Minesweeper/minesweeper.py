# import itertools    # Could've used `itertools.combinations` but I guess no need
import random
from typing import Iterable, List, Optional, Set, Tuple

Coord = Tuple[int, int]
Board = List[List[bool]]


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height: int = 8, width: int = 8, mines: int = 8) -> None:

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines: Set[Coord] = set()

        # Initialize an empty field with no mines
        self.board: Board = []
        for i in range(self.height):
            row: List[bool] = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found: Set[Coord] = set()

    def print(self) -> None:
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell: Coord) -> bool:
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell: Coord) -> int:
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count: int = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self) -> bool:
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells: Iterable[Coord], count: int) -> None:
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other) -> bool:
        return self.cells == other.cells and self.count == other.count

    def __str__(self) -> str:
        return f"{self.cells} = {self.count}"

    def known_mines(self) -> Set[Coord]:
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return set(self.cells)
        else:
            return set()

    def known_safes(self) -> Set[Coord]:
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count:
            return set()
        else:
            return set(self.cells)

    def mark_mine(self, cell: Coord) -> None:
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell not in self.cells:
            return
        self.cells.remove(cell)
        self.count -= 1

    def mark_safe(self, cell: Coord) -> None:
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        self.cells.discard(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height: int = 8, width: int = 8) -> None:

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made: Set[Coord] = set()

        # Keep track of cells known to be safe or mines
        self.mines: Set[Coord] = set()
        self.safes: Set[Coord] = set()

        # List of sentences about the game known to be true
        self.knowledge: List[Sentence] = []

        # Full board, for choosing random cell
        self._full_board: Set[Coord] = {(i, j)
                                        for i in range(height)
                                        for j in range(width)}

    def mark_mine(self, cell: Coord) -> None:
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell: Coord) -> None:
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def nearby_unknown_cells(self, cell: Coord) -> Set[Coord]:
        """
        Get adjacent cells (cells that share an edge or a corner)
          that are not checked

        Args:
            cell {Coord}: The cell whose neighbors should be checked

        Returns:
            {set[Coord]}: Set of neighbors
        """
        r, c = cell
        r_start = max(0, r-1)
        r_end = min(self.height, r+2)
        c_start = max(0, c-1)
        c_end = min(self.width, c+2)
        neighbors: Set[Coord] = set()
        for i in range(r_start, r_end):
            for j in range(c_start, c_end):
                # Skip original cell
                if i == r and j == c:
                    continue
                coord = (i, j)
                if coord not in self.safes:
                    neighbors.add(coord)
        return neighbors

    def add_knowledge(self, cell: Coord, count: int) -> None:
        """
        Called when the Minesweeper board tells us, for a given
          safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge

        Args:
            cell  {Coord}: The safe cell checked
            count {int}  : The number shown on cell
        """
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)
        # 2) mark the cell as safe
        self.mark_safe(cell)
        # 3) add a new sentence to the AI's knowledge base
        #    based on the value of `cell` and `count`
        neighbors = self.nearby_unknown_cells(cell)
        # No need to append a sentence that does not give new information
        if neighbors:
            new_sentence = Sentence(neighbors, count)
            # Remove known mines
            for known_mine_cell in self.mines:
                new_sentence.mark_mine(known_mine_cell)
            self.knowledge.append(new_sentence)
        while True:
            knowledge_not_changed = True
            # Copy the knowledge. The sentences are shallow copied
            tmp_knowledge = self.knowledge[:]
            for i, sentence in enumerate(tmp_knowledge):
                # Ignore nonexisting sentences that are removed in this loop
                if sentence not in self.knowledge:
                    continue
                # Get rid of empty sentences
                if not sentence.cells:
                    self.knowledge.remove(sentence)
                    continue
                # 4) mark any additional cells as safe or as mines
                #    if it can be concluded based on the AI's knowledge base
                # Mark mines
                mine_cells = sentence.known_mines()
                if mine_cells:
                    # This does not trigger `knowledge_not_changed` flag
                    #   because all mines are marked on the last loop
                    for mine_cell in mine_cells:
                        self.mark_mine(mine_cell)
                # Mark safes
                safe_cells = sentence.known_safes()
                if safe_cells:
                    # This does not trigger `knowledge_not_changed` flag
                    #   because all mines are marked on the last loop
                    for safe_cell in safe_cells:
                        self.mark_safe(safe_cell)
                # 5) add any new sentences to the AI's knowledge base
                #    if they can be inferred from existing knowledge
                # Only compare with sentences after this one to avoid
                #   double-checking
                for other_sentence in tmp_knowledge[i+1:]:
                    this_cells = sentence.cells
                    other_cells = other_sentence.cells
                    this_count = sentence.count
                    other_count = other_sentence.count
                    # Check for sub/super sets
                    if this_cells < other_cells:
                        diff_cells = other_cells - this_cells
                        diff_count = other_count - this_count
                    elif this_cells > other_cells:
                        diff_cells = this_cells - other_cells
                        diff_count = this_count - other_count
                    # We got a duplicate, most likely through elimination of
                    #   cells in sentences. This does not trigger
                    #   `knowledge_not_changed` flag because no information
                    #   is added
                    elif this_cells == other_cells:
                        if this_count != other_count:
                            raise ValueError(
                                "Sentences same cells different counts: "
                                f"{this_cells} {this_count} {other_count}")
                        else:
                            # It's possible the `other_sentence` is already
                            #   removed by another check
                            if other_sentence in self.knowledge:
                                self.knowledge.remove(other_sentence)
                        continue
                    # No sub/super set relationship
                    else:
                        continue
                    # Check if `new_cells` already covered by another sentence
                    for test_sentence in self.knowledge:
                        if test_sentence.cells == diff_cells:
                            break
                    # No match found. Add sentence
                    else:
                        knowledge_not_changed = False
                        diff_sentence = Sentence(diff_cells, diff_count)
                        self.knowledge.append(diff_sentence)
            # Break if no more information can be derived
            if knowledge_not_changed:
                break

    def make_safe_move(self) -> Optional[Coord]:
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        # Get safes minus checked
        unused_safes = self.safes - self.moves_made
        if unused_safes:
            return next(iter(unused_safes))
        # List exhaused. No safe cells known
        else:
            return None

    def make_random_move(self) -> Coord:
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        # Get all cells minus checked and mines
        unused_cells_not_mines = self._full_board - self.moves_made - self.mines
        if unused_cells_not_mines:
            return random.choice(tuple(unused_cells_not_mines))
        else:
            return None
