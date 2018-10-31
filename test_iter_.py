class Y:
    def __init__(self, data):
        self.data = data
        self.leng = len(data)
        self.start = 0

    def __iter__(self):
        for item in self.data:
            yield item

if __name__ == "__main__":

    y = Y([1,2,3,4,5])
    for i in y:
        print(i)
        