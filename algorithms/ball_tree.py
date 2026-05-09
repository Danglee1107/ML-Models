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
    def __init__(self, data: npt.NDArray[np.float64], 
                 ord: int = 2) -> None:
        self.head = self.build_tree(data, ord= ord)

# =================== VISUALIZE TREE ===================
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
    
# =====================================================

    def build_tree(self,
                data: Optional[npt.NDArray[np.float64]], 
                leafsize: int = 4,
                ord: int = 2) -> Optional[Node]:
        """
        Recursively build a Ball Tree.

        Parameters
        ----------
        data : np.ndarray
            Dataset of shape (n_samples, n_features).

        leafsize : int
            Maximum number of datapoints allowed in a leaf node.
            If the number of points is <= leafsize, recursion stops
            and the node becomes a leaf.

        Returns
        -------
        Node or None
            Root node of the Ball Tree.
        """

        # Base case:
        # If dataset is empty or None, there is no node to create.
        if data is None or len(data) == 0:
            return None

        # Compute centroid of current cluster.
        centroid: npt.NDArray[np.float64] = np.mean(data, axis=0)

        # Radius = maximum distance from centroid to any point.
        # This guarantees all points are inside the ball.
        radius = np.max(np.linalg.norm(data - centroid, axis=1, ord= ord))

        # Leaf condition:
        # Stop splitting when cluster becomes small enough.
        if len(data) <= leafsize:
            return Node(centroid, radius=radius, datapoints=data)

        # Choose a random point from the dataset.
        # Used as starting point to find two far-apart pivots.
        rand_point = data[np.random.choice(data.shape[0], size=1)]

        # Find the point farthest from the random point.
        f_point = data[np.argmax(np.linalg.norm(data - rand_point, axis=1, ord= ord))]

        # Find the point farthest from f_point.
        ff_point = data[np.argmax(np.linalg.norm(data - f_point, axis=1, ord= ord))]

        # Distance from every point to first pivot.
        d1 = np.linalg.norm(data - f_point, axis=1, ord= ord)

        # Distance from every point to second pivot.
        d2 = np.linalg.norm(data - ff_point, axis=1, ord= ord)

        # Points closer to f_point go to cluster 1.
        # Points closer to ff_point go to cluster 2.
        mask = d1 < d2
        clus1 = data[mask]
        clus2 = data[~mask]

        # Safety check:
        # Sometimes all points may end up in one cluster (duplicate points).
        if len(clus1) == 0 or len(clus2) == 0:
            return Node(centroid, radius=radius, datapoints=data)

        # Create internal node (no datapoints stored directly).
        node = Node(centroid, radius=radius)

        # Recursively build left and right subtrees.
        node.left = self.build_tree(clus1)
        node.right = self.build_tree(clus2)

        return node


    def search(self,
            query_point: npt.NDArray[np.float64],
            current_branch: Node,
            ord: int = 2):
        """
        Search for the nearest neighbor inside the Ball Tree.

        Parameters
        ----------
        query_point : np.ndarray
            Query point whose nearest neighbor we want to find.

        current_branch : Node
            Current node being explored.

        Returns
        -------
        (nearest_point, distance)
            nearest_point : np.ndarray
                Closest datapoint found.

            distance : float
                Euclidean distance from query_point to nearest_point.
        """

        # Leaf node:
        # Leaf nodes store actual datapoints.
        points = current_branch.datapoints

        # Internal node:
        # Internal nodes do not store datapoints directly.
        if len(points) == 0:

            # Get left and right child nodes.
            if current_branch.left:
                left_node = current_branch.left

            if current_branch.right:
                right_node = current_branch.right

            # Centers of both child balls.
            ctr1 = left_node.ctr
            ctr2 = right_node.ctr

            # Distance from query point to each child center.
            d1 = np.linalg.norm(query_point - ctr1, ord= ord)
            d2 = np.linalg.norm(query_point - ctr2, ord= ord)

            # Explore the closer child first.
            # This increases the chance of getting a good best_dist early.
            if d1 < d2:
                near_point, best_dist = self.search(query_point, left_node)
                check_branch = right_node
            else:
                near_point, best_dist = self.search(query_point, right_node)
                check_branch = left_node

            # Compute lower bound distance to the other branch.
            #
            # lower_bound = distance(query, ball_center) - ball_radius
            #
            # If this lower bound is already larger than best_dist,
            # then no point inside that ball can be closer.
            low_bound = (
                np.linalg.norm(query_point - check_branch.ctr, ord= ord)
                - check_branch.radius
            )

            # Only explore the other branch if it may contain a closer point.
            if low_bound < best_dist:

                check_near_point, check_dist = self.search(
                    query_point,
                    check_branch
                )

                # Update best result if better point found.
                if check_dist < best_dist:
                    near_point = check_near_point
                    best_dist = check_dist

            return near_point, best_dist

        # Leaf node search:
        # Compute distances from query point to all points in leaf.
        else:

            # Find index of nearest point in leaf.
            near_point = points[
                np.argmin(np.linalg.norm(points - query_point, axis=1, ord= ord))
            ]

            # Actual nearest neighbor distance.
            distance = np.linalg.norm(near_point - query_point, ord= ord)

            return near_point, distance

def main() -> None:
    # sample
    points = np.array([[2,2],[4,1],[3,4],
                     [6,4],[4,7],[6,7],
                     [7,9],[8,6],[8,2],
                     [9,7],[9,2],[10,4]])
    data = points
    new_point = np.array([7,3])

    btree = BallTree(data)
    if btree.head:
        bt_near_point, bt_dist = btree.search(new_point, btree.head) 

    print("ball tree")
    print(f"point: {bt_near_point}")
    print(f"distance: {bt_dist}")

    # levels = btree.level_order()
    # for level in levels:
    #     print(level)

    print(btree.drawTree())

if __name__ == '__main__':
    main()