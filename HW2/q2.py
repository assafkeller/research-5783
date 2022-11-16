import re

def lastcall(function):

    inputs = dict()
    def wrapper(x):
        key = function.__name__
        if key in inputs and inputs[key] == x:
            return "I already told you that the answer is " + str(function(x))
        else:
            inputs[key] = x
            return function(x)
    return wrapper


@lastcall
def f(x: int):
    return x**2


@lastcall
def check(email): #function for validating an Email
    regex = r'\b[A-Za-z0-9]+[._%+-]?[A-Za-z0-9]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' #regular expression for validating an Email

    if (re.fullmatch(regex, email)): #fullmatch() method
        return "email is Valid"

    else:
        return "email is inValid"




if __name__ == '__main__':

    print(f(2))
    print(f(2))

    print(check('abc-@mail.com'))
    print(check('abc-@mail.com'))
    print(check('abc-d@mail.com'))
    print(check('abc-d@mail.com'))