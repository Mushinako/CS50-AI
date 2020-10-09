from collections import deque
from typing import List, Optional, Tuple


class Node():
    def __init__(self, state: str, parent, action: Optional[str]):
        self.state = state
        self.parent: Optional[Node] = parent
        self.action = action

    def get_path(self) -> List[Tuple[str, str]]:
        """
        Get path up to this node
        """
        if self.parent is None:
            return []
        else:
            return self.parent.get_path() + [(self.action, self.state)]


class QueueFrontier():
    def __init__(self):
        self.frontier: deque[Node] = deque()

    def add(self, node: Node):
        self.frontier.append(node)

    def contains_state(self, state: str) -> bool:
        return any(node.state == state for node in self.frontier)

    def empty(self) -> bool:
        return len(self.frontier) == 0

    def remove(self) -> Node:
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier.popleft()
            return node
