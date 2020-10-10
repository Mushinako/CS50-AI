from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # A is knight xor knave
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    Or(
        # If A honest, then A is both a knight and a knave
        And(AKnight, And(AKnight, AKnave)),
        # If A dishonest, then A is **not** both a knight and a knave
        And(AKnave, Not(And(AKnight, AKnave)))
    )
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # A is knight xor knave
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    # B is knight xor knave
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    Or(
        # A is honest, then both are knaves
        And(AKnight, And(AKnave, BKnave)),
        # A is dishonest, the not both are knaves
        And(AKnave, Not(And(AKnave, BKnave)))
    )
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # A is knight xor knave
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    # B is knight xor knave
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    Or(
        And(
            AKnight,    # A is honest, then both are the same kind
            Or(
                And(AKnight, BKnight),      # Both are knights
                And(AKnave, BKnave)         # Both are knaves
            )
        ),
        And(
            AKnave,     # A is dishonest, then both are **not** the same kind
            Not(Or(
                And(AKnight, BKnight),      # (Not) both are knights
                And(AKnave, BKnave)         # (Not) both are knaves
            ))
        )
    ),
    Or(
        And(
            BKnight,    # B is honest, then both are of different kinds
            Or(
                And(AKnight, BKnave),       # A is knight and B is knave
                And(AKnave, BKnight)        # A is knave and B is knight
            )
        ),
        And(
            BKnave,     # B is dishonest, then both are the same kind
            Or(
                And(AKnight, BKnight),      # Both are knights
                And(AKnave, BKnave)         # Both are knaves
            )
        )
    )
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # A is knight xor knave
    And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    # B is knight xor knave
    And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    # C is knight xor knave
    And(Or(CKnight, CKnave), Not(And(CKnight, CKnave))),
    Or(
        And(
            BKnight,    # If B is honest, then A said "I am a knave"
            Or(
                And(AKnight, AKnave),   # If A is honest, then A is a knave
                And(AKnave, AKnight)    # If A is dishonest, then A is a knight
            )
        ),
        And(
            BKnave,     # If B is dishonest, then A said "I am a knight"
            Or(
                And(AKnight, AKnight),  # If A is honest, then A is a knight
                And(AKnave, AKnave)     # If A is dishonest, then A is a knave
            )
        )
    ),
    Or(
        And(BKnight, CKnave),   # If B is honest, then C is knave
        And(BKnave, CKnight)    # If B is dishonest, then C is knight
    ),
    Or(
        And(CKnight, AKnight),  # If C is honest, then A is knight
        And(CKnave, AKnave),    # IF C is dishonest, then A is knave
    )
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
