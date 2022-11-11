import doctest
import networkx as nx
import matplotlib.pyplot as plt

def neb(node: any) -> list:

    points_list1 = [('a','b'), ('b','c'), ('c','f'), ('c','g'), ('a','g'), ('g','d'), ('e','f'),('r','r')]
    Glist=nx.Graph()
    Glist.add_edges_from(points_list1)

    x = list(Glist.neighbors(node))
    # pos = nx.spring_layout(Glist)
    # nx.draw_networkx_nodes(Glist, pos)
    # nx.draw_networkx_edges(Glist, pos)
    # nx.draw_networkx_labels(Glist, pos)
    #
    # plt.show()
    return x

def four_neighbor_function(node: any) -> list:
    (x, y) = node

    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def breadth_first_search(start, end, neighbor_function):
    """
        >>> breadth_first_search(start=(6, 6), end=(5, -2), neighbor_function=four_neighbor_function)
        [(6, 6), (5, 6), (5, 5), (5, 4), (5, 3), (5, 2), (5, 1), (5, 0), (5, -1), (5, -2)]
        >>> breadth_first_search(start=(0,0), end=(2,2), neighbor_function=four_neighbor_function)
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        >>> breadth_first_search(start=(0,0), end=(-3,3), neighbor_function=four_neighbor_function)
        [(0, 0), (-1, 0), (-2, 0), (-3, 0), (-3, 1), (-3, 2), (-3, 3)]
        >>> breadth_first_search(start=(1.5,0.2), end=(2.5,2.2), neighbor_function=four_neighbor_function)
        [(1.5, 0.2), (2.5, 0.2), (2.5, 1.2), (2.5, 2.2)]
        >>> breadth_first_search(start=('a'), end=('r'), neighbor_function= neb)
        Traceback (most recent call last):
              File "D:\אלגוריתמים מחקריים\תרגיל בית 1\‏‏q2.py", line 62, in <module>
               print(breadth_first_search(start=('a'), end=('r'), neighbor_function= neb))
              File "D:\אלגוריתמים מחקריים\תרגיל בית 1\‏‏q2.py", line 56, in breadth_first_search
               raise Exception('no path found')
        Exception: no path found

        >>> breadth_first_search(start=('a'), end=('g'), neighbor_function= neb)
        ['a', 'g']


    """
    queue = []  # Initialize a queue
    queue.append([start])
    visited = []
    visited.append([start])
    while queue:  # Creating loop to visit each node
        path = queue.pop(0)
        node = path[-1]

        if node == end:
            return path
        for neighbour in neighbor_function(node):
            if neighbour not in visited:
                visited.append(node)
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)

    raise Exception('no path found')




# print(breadth_first_search(start=(1.5,0.2), end=(2.5,2.2), neighbor_function=four_neighbor_function))
# print(breadth_first_search(start=('a'), end=('g'), neighbor_function= neb))

doctest.testmod(verbose=True)






