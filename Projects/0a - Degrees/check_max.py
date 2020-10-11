import csv
import sys
from typing import List, Optional, Tuple

from util import Node, QueueFrontier

# Maps names to a set of corresponding person_ids
names = {}

# Maps person_ids to a dictionary of: name, birth, movies (a set of movie_ids)
people = {}

# Maps movie_ids to a dictionary of: title, year, stars (a set of person_ids)
movies = {}


def load_data(directory):
    """
    Load data from CSV files into memory.
    """
    # Load people
    with open(f"{directory}/people.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            people[row["id"]] = {
                "name": row["name"],
                "birth": row["birth"],
                "movies": set()
            }
            if row["name"].lower() not in names:
                names[row["name"].lower()] = {row["id"]}
            else:
                names[row["name"].lower()].add(row["id"])

    # Load movies
    with open(f"{directory}/movies.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            movies[row["id"]] = {
                "title": row["title"],
                "year": row["year"],
                "stars": set()
            }

    # Load stars
    with open(f"{directory}/stars.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                people[row["person_id"]]["movies"].add(row["movie_id"])
                movies[row["movie_id"]]["stars"].add(row["person_id"])
            except KeyError:
                pass


def main():
    if len(sys.argv) > 2:
        sys.exit("Usage: python degrees.py [directory]")
    directory = sys.argv[1] if len(sys.argv) == 2 else "large"

    # Load data from files into memory
    print("Loading data...")
    load_data(directory)
    print("Data loaded.")

    max_pair = ('', '')
    max_diff = 0
    max_len = 0
    print("Sorting IDs...")
    ids = sorted([id_ for ids_ in names.values() for id_ in ids_], key=int)
    print("IDs sorted.")
    for i, name_i in reversed(list(enumerate(ids))):
        for j, name_j in enumerate(ids):
            if i >= j:
                continue
            path = shortest_path(name_i, name_j)
            if path is None:
                continue
            length = len(path)
            if length < max_len:
                continue
            diff = int(name_j) - int(name_i)
            pair = (name_i, name_j)
            print(length, diff, pair)
            if length > max_len:
                max_len = length
                max_diff = diff
                max_pair = pair
                print("  Length increase")
                continue
            if diff <= max_diff:
                continue
            max_diff = diff
            max_pair = pair
            print("  Diff increase")
    print("Finished:", max_len, max_diff, max_pair)


def shortest_path(source: str, target: str) -> Optional[List[Tuple[str, str]]]:
    """
    Returns the shortest list of (movie_id, person_id) pairs
    that connect the source to the target.

    If no possible path, returns None.
    """
    # Queue init: Add root node
    root_node = Node(source, None, None)
    queue = QueueFrontier()
    queue.add(root_node)
    # Set of visited people
    visited = {source}
    # Iterate till queue empty
    while not queue.empty():
        # Popleft
        test_node = queue.remove()
        # Check if target reached and Append non-visited neighbors to queue
        all_neighbor_ids = neighbors_for_person(test_node.state)
        for movie_id, person_id in all_neighbor_ids:
            if person_id == target:
                return test_node.get_path() + [(movie_id, person_id)]
            if person_id not in visited:
                visited.add(person_id)
                node = Node(person_id, test_node, movie_id)
                queue.add(node)
    # Queue exhausted. No solution
    return None


def person_id_for_name(name):
    """
    Returns the IMDB id for a person's name,
    resolving ambiguities as needed.
    """
    person_ids = list(names.get(name.lower(), set()))
    if len(person_ids) == 0:
        return None
    elif len(person_ids) > 1:
        print(f"Which '{name}'?")
        for person_id in person_ids:
            person = people[person_id]
            name = person["name"]
            birth = person["birth"]
            print(f"ID: {person_id}, Name: {name}, Birth: {birth}")
        try:
            person_id = input("Intended Person ID: ")
            if person_id in person_ids:
                return person_id
        except ValueError:
            pass
        return None
    else:
        return person_ids[0]


def neighbors_for_person(person_id):
    """
    Returns (movie_id, person_id) pairs for people
    who starred with a given person.
    """
    movie_ids = people[person_id]["movies"]
    neighbors = set()
    for movie_id in movie_ids:
        for person_id in movies[movie_id]["stars"]:
            neighbors.add((movie_id, person_id))
    return neighbors


if __name__ == "__main__":
    main()
