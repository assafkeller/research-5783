import re



def check(email): #function for validating an Email
    regex = r'\b[A-Za-z0-9]+[._%+-]?[A-Za-z0-9]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' #regular expression for validating an Email
    valid =[]
    Invalid = []

    for i in email:
        if (re.fullmatch(regex, i)): #fullmatch() method
            valid.append(i)
           
        else:
            Invalid.append(i)

    print("Valid Email: \n",valid)
    print("Invalid Email: \n", Invalid)

if __name__ == "__main__":
    file1 = open(r"D:\אלגוריתמים מחקריים\תרגיל בית 2\mail list.txt", "r+")

    data = str(file1.read())
    data_into_list = data.split("\n")
    # print(data)

    # print(data_into_list [1])
    check(data_into_list)

