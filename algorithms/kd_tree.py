import numpy as np
import numpy.typing as npt

class Node:
    def __init__(self, point: npt.NDArray[np.float64], 
                 split_dim: int) -> None:
        self.point = point
        self.dim = split_dim
        self.left: Node | None = None
        self.right: Node | None = None

        
class KDTree:
    def __init__(self, datapoints: npt.NDArray[np.float64]) -> None:
        self.root = self.build_tree(datapoints)

    def drawTree(self):
        if self.root is None:
            return "Tree is empty"

        lines, _, _, _ = self._build_tree(self.root )
        return "\n".join(lines)

    def _build_tree(self, node: Node):
        if node is None:
            return [], 0, 0, 0

        label = str(list(map(int, node.point)))
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

    def build_tree(self, data_points: npt.NDArray[np.float64],
                   depth: int = 0) -> Node | None:
        """
        Recursively build a KD-tree from a set of data points.

        The algorithm selects a splitting axis based on the current depth
        (``axis = depth % k``), sorts the data along that axis, and chooses
        the median point as the current node. The process is repeated
        recursively for the left and right subsets.

        Parameters
        ----------
        data_points : NDArray[np.float64]
            Input dataset of shape (n_samples, n_features).
        depth : int, optional
            Current depth of the tree, used to determine the splitting axis.
            Default is 0.

        Returns
        -------
        Node | None
            The root node of the constructed KD-tree. Returns None if the
            input dataset is empty.

        Notes
        -----
        - The splitting dimension is chosen cyclically across features.
        - The median point ensures the tree is approximately balanced.
        - Time complexity (average): O(n log n)

        Examples
        --------
        >>> import numpy as np
        >>> arr = np.array([[2, 1],
        ...                 [3, 3],
        ...                 [4, 4],
        ...                 [1, 6],
        ...                 [5, 2],
        ...                 [7, 5]])
        >>> kd = KDTree()
        >>> root = kd.build_tree(arr)

        Visualize::

                 _______[4, 4]________ 
                /                     \\
            __[3, 3]__            __[7, 5]
           /          \\          /
         [2, 1]      [1, 6]   [5, 2]
        """

        if len(data_points) == 0:
            return

        col = data_points.shape[1]
        axis = depth % col
        k = len(data_points) // 2
        idx = np.argpartition(data_points[:, axis], k)
        data_points = data_points[idx]

        median = data_points[k]
        node = Node(median, axis)

        node.left = self.build_tree(data_points[: k], 
                        depth= depth + 1)

        node.right= self.build_tree(data_points[k + 1 :], 
                        depth= depth + 1)
        
        return node

    def search(self, new_point: npt.NDArray[np.float64],
               ord: int = 2) -> tuple[Node | None, np.float64 | None]:
        """
        Find the nearest neighbor to a given point using KD-tree search.

        This method traverses the KD-tree to efficiently locate the closest point
        to the provided `new_point` by pruning branches that cannot contain a closer point.
        It uses a stack-based approach to avoid recursion and compares distances
        at each node, backtracking only when necessary.

        Parameters
        ----------
        new_point : NDArray[np.float64]
            The query point for which the nearest neighbor is to be found.

        Returns
        -------
        tuple[Node | None, np.float64 | None]
            A tuple containing the nearest node and the distance (Euclidean by default) to it.
            Returns (None, None) if the tree is empty.

        Examples
        --------
        >>> import numpy as np
        >>> arr = np.array([[2, 1], [3, 3], [4, 4], [1, 6], [5, 2], [7, 5]])
        >>> kd = KDTree(arr)
        >>> new_point = np.array([3, 2])
        >>> nearest, distance = kd.search(new_point)
        >>> print(nearest.point)
        [3 3]
        >>> print(distance)
        1.0

        Notes
        -----
        - Uses stack-based traversal for efficiency and to avoid recursion depth limits.
        - Prunes branches where the closest possible point cannot be closer than the current best.
        """

        if self.root is None:
            return (None, None)

        stack: list[tuple[Node | None, bool]] = [(self.root, False)]
        nearest = self.root
        best_dist = None
        while stack:
            node, visited = stack.pop()
            if node is None:
                continue

            axis = node.dim

            if not visited:
                current_dist = np.linalg.norm(node.point - new_point, ord= ord)
                if best_dist is None or current_dist < best_dist:
                    best_dist = current_dist
                    nearest = node
                
                if node.point[axis] < new_point[axis]:
                    near = node.right
                    far = node.left
                else:
                    near = node.left
                    far = node.right

                stack.append((node, True))
                stack.append((near, False)) 
            
            else:
                d_plane = abs(new_point[axis] - node.point[axis])
                if d_plane < best_dist:
                    if new_point[axis] < node.point[axis]:
                        far = node.right
                    else:
                        far = node.left
                    
                    stack.append((far, False))
        best_dist = np.float64(best_dist)
        return nearest, best_dist

def main() -> None:
    arr = np.array([[2,1],
                   [3,3],
                   [4,4],
                   [1,6],
                   [5,2],
                   [7,5]])
    kd = KDTree(arr)
    new_point = np.array([3,2])
    print(kd.drawTree())
    near, distance = kd.search(new_point)
    print(near.point) #type: ignore
    print(distance)

if __name__ == '__main__':
    main()