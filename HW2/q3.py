class List(list):

    def __init__(self, *args):
        super().__init__(*args)

    def __getitem__(self, *args):
        if isinstance(*args, tuple):
            temp = super().__getitem__(args[0][0])
            for index in args[0][1:]:
                temp = temp[index]
            return temp
        return super().__getitem__(*args)

if __name__ == '__main__':
    mylist = List([
    [[1, 2, 3, 33], [4, 5, 6, 66]],
    [[7, 8, 9, 99], [10, 11, 12, 122]],
    [[13, 14, 15, 155], [16, 17, 18, 188]],
    ]
        )
    print(mylist)
    print(mylist[0,1,3])
    print(mylist[0,1])
    print(mylist[0])

    mylist1 = List([
        [['a', 2, 'c', 33], [4, 5, 'f', 66]],
        [[7, 8, 9, 99], [10, 11, 12, 122]],
        [[13, 'y', 15, 155], [16, 17, 18, 188]],
    ]
    )
    print(mylist1[0,1,2])
    mylist1[0].append("ass")
    print(mylist1)
    mylist1.pop()
    print(mylist1)


    print(mylist + mylist1)
    print(mylist == mylist1)
    mylist [0][1][2] = 'keller'
    print(mylist)