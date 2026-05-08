import numpy as np
import numpy.typing as npt
from typing import Annotated, Any, Optional, Union

Matrix2D = Annotated[npt.NDArray[np.float64], ('rows', 'cols')]


class Node:
    def __init__(self, centroid: npt.NDArray[np.float64],
                 radius: float,
                 datapoints: Matrix2D = np.array([])) -> None:
        self.ctr = centroid
        self.datapoints = datapoints
        self.radius = radius
        self.left: Optional["Node"] = None
        self.right: Optional["Node"] = None

        
class BallTree:
    def __init__(self, data: npt.NDArray[np.float64]) -> None:
        self.head = self.build_tree(data)

    def drawTree(self):
        if self.head is None:
            return "Tree is empty"

        lines, _, _, _ = self._build_tree(self.head)
        return "\n".join(lines)

    def _build_tree(self, node: Node):
        if node is None:
            return [], 0, 0, 0

        label = str(list(map(lambda x: round(float(x), 2), node.ctr)))
        width = len(label)

        # Leaf node
        if node.left is None and node.right is None:
            return [label], width, 1, width // 2

        # Recursively build subtrees
        left_lines, left_width, left_height, left_middle = \
            self._build_tree(node.left) if node.left else ([], 0, 0, 0)

        right_lines, right_width, right_height, right_middle = \
            self._build_tree(node.right) if node.right else ([], 0, 0, 0)

        # Calculate height difference
        height = max(left_height, right_height)

        # Fill missing levels
        left_lines += [" " * left_width] * (height - left_height)
        right_lines += [" " * right_width] * (height - right_height)

        # Build first line
        first_line = ""
        second_line = ""

        # Left branch
        if left_width > 0:
            first_line += " " * (left_middle + 1)
            first_line += "_" * (left_width - left_middle - 1)
            second_line += " " * left_middle + "/"
            second_line += " " * (left_width - left_middle - 1)
        else:
            first_line += ""
            second_line += ""

        # Node label
        first_line += label
        second_line += " " * width

        # Right branch
        if right_width > 0:
            first_line += "_" * right_middle
            first_line += " " * (right_width - right_middle)
            second_line += " " * right_middle + "\\"
            second_line += " " * (right_width - right_middle - 1)
        else:
            first_line += ""
            second_line += ""

        # Merge children
        merged_lines = [
            left + " " * width + right
            for left, right in zip(left_lines, right_lines)
        ]

        total_width = left_width + width + right_width
        middle = left_width + width // 2

        return (
            [first_line, second_line] + merged_lines,
            total_width,
            height + 2,
            middle
        )

    def level_order(self) -> list[Matrix2D]:
        '''
        Using library for better performance O(n²) -> O(n)
        because if using list to represent queue (queue.pop(0) make O(n))
        '''
        from collections import deque

        list_traversal: list[Matrix2D] = []
        if self.head is None:
            return list_traversal
        
        queue = deque([self.head])
        while queue:
            node = queue.popleft()
            list_traversal.append(node.datapoints)
            if node.left:
                queue.append(node.left) 
            if node.right: 
                queue.append(node.right)

        return list_traversal
    
    def build_tree(self,
                   data: Optional[npt.NDArray[np.float64]], 
                   leafsize: int= 4) -> Optional[Node]:

        if data is None or len(data) == 0:
            return None

        centroid: npt.NDArray[np.float64] = np.mean(data, axis=0)
        radius = float(np.max(np.linalg.norm(data - centroid, axis=1)))

        if len(data) <= leafsize:
            return Node(centroid, radius=radius, datapoints=data)

        rand_point = data[np.random.choice(data.shape[0], size=1)]
        f_point = data[np.argmax(np.linalg.norm(data - rand_point, axis=1))]
        ff_point = data[np.argmax(np.linalg.norm(data - f_point, axis=1))]

        d1 = np.linalg.norm(data - f_point, axis=1)
        d2 = np.linalg.norm(data - ff_point, axis=1)
        mask = d1 < d2

        clus1 = data[mask]
        clus2 = data[~mask]

        if len(clus1) == 0 or len(clus2) == 0:
            return Node(centroid, radius= radius, datapoints= data)

        node = Node(centroid, radius= radius)
        node.left = self.build_tree(clus1)
        node.right = self.build_tree(clus2)

        return node

    def search(self, query_point: npt.NDArray[np.float64], 
               current_branch: Node):

        points = current_branch.datapoints
        if len(points) == 0:
            if current_branch.left:
                left_node = current_branch.left
            if current_branch.right:
                right_node = current_branch.right

            ctr1 = left_node.ctr
            ctr2 = right_node.ctr

            d1 = np.linalg.norm(query_point - ctr1)
            d2 = np.linalg.norm(query_point - ctr2)

            if d1 < d2:
                near_point, best_dist = self.search(query_point, left_node)
                check_branch = right_node
            else:
                near_point, best_dist = self.search(query_point, right_node)
                check_branch = left_node

            low_bound = np.linalg.norm(query_point - check_branch.ctr) - check_branch.radius
            if low_bound < best_dist:
                check_near_point , check_dist = self.search(query_point, check_branch)

                if check_dist < best_dist:
                    near_point = check_near_point
                    best_dist = check_dist

            return near_point, best_dist

        else:
            near_point = points[np.argmin(np.linalg.norm(points - query_point, axis= 1))]
            distance = np.linalg.norm(near_point - query_point)
            return near_point, distance


def main() -> None:
    points = np.array([[2,2],[4,1],[3,4],
                     [6,4],[4,7],[6,7],
                     [7,9],[8,6],[8,2],
                     [9,7],[9,2],[10,4]])
    X = np.array([
        [1, 2, 1],
        [2, 1, 2],
        [3, 2, 1],
        [8, 7, 9],
        [9, 8, 8],
        [10, 7, 9],
        [5, 5, 5],
        [6, 5, 4]
    ])
    data = X
    new_point = np.array([5,4,1])
    bf_near_point = data[np.argmin(np.linalg.norm(data - new_point, axis= 1))]
    bf_dist = np.linalg.norm(bf_near_point - new_point)

    btree = BallTree(data)
    if btree.head:
        bt_near_point, bt_dist = btree.search(new_point, btree.head) 

    print("brute force")
    print(f"point: {bf_near_point}")
    print(f"distance: {bf_dist}")
    print("ball tree")
    print(f"point: {bt_near_point}")
    print(f"distance: {bt_dist}")
    # levels = btree.level_order()
    # for level in levels:
    #     print(level)
    print(btree.drawTree())

if __name__ == '__main__':
    main()