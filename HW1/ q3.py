
import doctest


def print_sorted(l):
    """
        >>> print_sorted([1, 5, (8,4), 6 ,3,2, {"a": 5, "c": 6, "b": {1, 3, 5,2, 4}}])
        [(4, 8), 1, 2, 3, 5, 6, {'a': 5, 'b': {1, 2, 3, 4, 5}, 'c': 6}]
        >>> print_sorted([1,(8,4), {1, 5, 2, 4},6 ,3,2, {2:2,1:'assaf'}])
        [(4, 8), 1, 2, 3, 6, {1, 2, 4, 5}, {1: 'assaf', 2: 2}]
    """
    z=[]
    while l:
        x = l.pop(0)

        if type(x) is (int):
            z.append(x)

        elif type(x) is (str):
                z.append(x)

        else:
            match x:
                   case tuple():
                                y = sorted(x, key=str)
                                z.append(tuple(y))

                   case set():
                                y = sorted(x, key=str)
                                z.append(set(y))

                   case list():
                                y = sorted(x, key=str)
                                z.append(list(y))

                   case dict():
                                y=sorted(x.items())
                                z.append(dict(y))

    z=sorted(z, key=str)
    return z

doctest.testmod(verbose=True)



