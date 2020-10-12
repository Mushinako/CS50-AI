import sys
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword: Crossword) -> None:
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment: Dict[Variable, str]) -> List[List[Optional[str]]]:
        """
        Return 2D array representing a given assignment.
        """
        letters: List[List[Optional[str]]] = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment: Dict[Variable, str]) -> None:
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment: Dict[Variable, str], filename: str) -> None:
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self) -> None:
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var, domain in self.domains.items():
            length = var.length
            # Only put back the ones that are of the correct length
            self.domains[var] = {s for s in domain if len(s) == length}

    def revise(self, x: Variable, y: Variable) -> bool:
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        overlap = self.crossword.overlaps[x, y]

        # No overlap
        if overlap is None:
            return

        x_pos, y_pos = overlap
        # Set of possible chars at position `y_pos` in `y`'s domain
        y_chars = {s[y_pos] for s in self.domains[y]}

        # Mark current size of `x`'s domain
        before_domain_size = len(self.domains[x])
        # Only put back the ones that have the correct character at `x_pos`
        self.domains[x] = {s for s in self.domains[x] if s[x_pos] in y_chars}

        # If `x`'s domain size shrunk, then we know it changed
        return len(self.domains[x]) < before_domain_size

    def ac3(self, arcs: Optional[Deque[Tuple[Variable, Variable]]] = None) -> bool:
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            overlaps = self.crossword.overlaps
            # Get all arcs
            arcs = deque(arc for arc in overlaps if overlaps[arc] is not None)

        # While queue is not empty
        while arcs:
            x, y = arcs.popleft()
            # If `x`'s domain unchanged, check next arc
            if not self.revise(x, y):
                continue
            # `x`'s domain changed
            # Check if `x`'s domain is empty
            if not self.domains[x]:
                return False
            # Recheck neighbors
            unchecked_neighbors = self.crossword.neighbors(x) - {y}
            for neighbor in unchecked_neighbors:
                arcs.append((neighbor, x))

        # Queue exhausted
        return True

    def assignment_complete(self, assignment: Dict[Variable, str]) -> bool:
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        # All the variables must be in `assignment`
        return all(var in assignment for var in self.crossword.variables)

    def consistent(self, assignment: Dict[Variable, str]) -> bool:
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # Check strings for each variable is unique
        values = assignment.values()
        if len(values) != len(set(values)):
            return False

        # Check all node consistencies
        if any(var.length != len(s) for var, s in assignment.items()):
            return False

        # Check all arc consistencies
        for var, s in assignment.items():
            for neighbor in self.crossword.neighbors(var):
                # If the neighbor is not assigned, ignore
                if neighbor not in assignment:
                    continue
                overlap = self.crossword.overlaps[var, neighbor]
                assert overlap is not None
                var_pos, neighbor_pos = overlap
                # Mismatch at the position
                if s[var_pos] != assignment[neighbor][neighbor_pos]:
                    return False

        # All checked
        return True

    def order_domain_values(self, var: Variable, assignment: Dict[Variable, str]) -> List[str]:
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        # Sort the domain by number of choices eliminated
        return sorted(self.domains[var],
                      key=lambda s: self._num_choices_eliminated(s, var, assignment))

    def _num_choices_eliminated(self, string: str, var: Variable, assignment: Dict[Variable, str]) -> int:
        """
        Get the number of choices eliminated by assigning `s` to `var`

        Args:
            string     {str}                : The string to be checked
            var        {Variable}           : The variable to assign to
            assignment {dict[Variable, str]}: Existing assignments
        """
        unassigned_neighbors = self.crossword.neighbors(var) - set(assignment)

        accumulator: int = 0
        # Get eliminated number from each neighbor
        for neighbor in unassigned_neighbors:
            overlap = self.crossword.overlaps[var, neighbor]
            assert overlap is not None
            var_pos, neighbor_pos = overlap
            target_char = string[var_pos]
            # Add number of eliminated for this neighbor
            eliminated_domains = [None for s in self.domains[neighbor]
                                  if s[neighbor_pos] != target_char]
            accumulator += len(eliminated_domains)

        return accumulator

    def select_unassigned_variable(self, assignment: Dict[Variable, str]) -> Variable:
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_variables = tuple(self.crossword.variables
                                     - set(assignment))

        # Get variables with the minimum number of remaining values
        min_domain_size = min(len(self.domains[var])
                              for var in unassigned_variables)
        min_domain_size_variables = [var for var in unassigned_variables
                                     if len(self.domains[var]) == min_domain_size]

        # If only 1 variable has the minimum number of remaining values, return it
        if len(min_domain_size_variables) == 1:
            return min_domain_size_variables[0]

        # Get variables with the highest degree
        highest_degree = max(len(self.crossword.neighbors(var))
                             for var in min_domain_size_variables)

        # Return the first variable that has the highest degree
        for var in min_domain_size_variables:
            if len(self.crossword.neighbors(var)) == highest_degree:
                return var

    def backtrack(self, assignment: Dict[Variable, str]) -> Optional[Dict[Variable, str]]:
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # Check goal condition
        if self.assignment_complete(assignment):
            return assignment

        # Select an unassigned variable (optimally)
        var = self.select_unassigned_variable(assignment)
        # Iterate through the domain (optimally sorted)
        for value in self.order_domain_values(var, assignment):
            assignment[var] = value
            # Assignment not consistent. Remove assignment
            if not self.consistent(assignment):
                del assignment[var]
                continue
            # Assignment consistent
            # Backup current domains and assign. Only shallow copy of
            #   domains are needed because my implementation of `revise`
            #   Does not modify it
            domains = {v: d for v, d in self.domains.items()}
            self.domains[var] = {value}
            # Run AC3 on all the neighbors
            queue = deque((neighbor, var)
                          for neighbor in self.crossword.neighbors(var))
            arc_consistent = self.ac3(queue)
            # AC3 successful
            if arc_consistent:
                # Run search on assignment
                result = self.backtrack(assignment)
                # Got result
                if result is not None:
                    return result
            # Either AC3 failed or backtrack failed
            # Remove current assignment
            del assignment[var]
            # Recover domains
            self.domains = domains

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)
            import subprocess
            command = ["start", output]
            process = subprocess.Popen(command, shell=True)
            process.wait()


if __name__ == "__main__":
    main()
