import doctest
import sys
sys.tracebacklimit=0


def f(x: str, y: str, taz: int):
    return x + " " + y, taz


def safe_call(f, *args):
    """
        >>> safe_call(f, "Assaf", "Keller", 300852126)
        ('Assaf Keller', 300852126)

        >>> safe_call(f, "Assaf", 300000, 300852126)
        Traceback (most recent call last):
            File "C:\\Users\\aakel\\anaconda3\\lib\\doctest.py", line 1336, in __run
             exec(compile(example.source, filename, "single",
            File "<doctest __main__.safe_call[1]>", line 1, in <module>
             safe_call(f, "Assaf", 300000, 300852126)
            File "C:\\Users\\aakel\\AppData\\Roaming\\JetBrains\\PyCharmCE2021.3\\scratches\\scratch.py", line 42, in safe_call
             raise Exception(key, 'is not from type', args_type[i])
        Exception: (300000, 'is not from type', 'str')
        >>> safe_call(f, 300852126, "Keller", 300852126)
        Traceback (most recent call last):
            File "C:\\Users\\aakel\\anaconda3\\lib\\doctest.py", line 1336, in __run
             exec(compile(example.source, filename, "single",
            File "<doctest __main__.safe_call[1]>", line 1, in <module>
             safe_call(f, "Assaf", 300000, 300852126)
            File "C:\\Users\\aakel\\AppData\\Roaming\\JetBrains\\PyCharmCE2021.3\\scratches\\scratch.py", line 42, in safe_call
             raise Exception(key, 'is not from type', args_type[i])
        Exception: (300852126, 'is not from type', 'str')
        >>> safe_call(f,  "Keller", "Assaf", "Keller")
        Traceback (most recent call last):
            File "C:\\Users\\aakel\\anaconda3\\lib\\doctest.py", line 1336, in __run
             exec(compile(example.source, filename, "single",
            File "<doctest __main__.safe_call[1]>", line 1, in <module>
             safe_call(f, "Assaf", 300000, 300852126)
            File "C:\\Users\\aakel\\AppData\\Roaming\\JetBrains\\PyCharmCE2021.3\\scratches\\scratch.py", line 42, in safe_call
             raise Exception(key, 'is not from type', args_type[i])
        Exception: ('Keller', 'is not from type', 'int')

    """




    i = 0
    args_type = []
    for x in f.__annotations__:
        args_type.append(f.__annotations__[x].__name__)
        i = i + 1

    i = 0

    for key in args:
        t = type(key).__name__
        if args_type[i] != t:
            raise Exception(key, 'is not from type', args_type[i])
        i = i + 1

    return print(str(f(*args)))


# print(f.__annotations__['x'].__name__)


# safe_call(f, "Assaf", "Keller", 300852126)
# safe_call(f, "Assaf", 300000, 300852126)
# safe_call(f, 300852126, "Keller", 300852126)
# safe_call(f, "Assaf", "Keller", "Keller")
# safe_call(f, "Keller", "Assaf", "Keller")

if __name__ == "__main__":
    doctest.testmod(verbose=True)

# print(set(inspect.getfullargspec(f).args))
