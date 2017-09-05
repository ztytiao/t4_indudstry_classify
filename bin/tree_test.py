import numpy as np

class tree():
    def __init__(self,left=[],right=[]):
        self.left=left
        self.right=right
    def printSon(self):
        self.left.printName()
    def getName(self,Name):
        self.name=Name
    def printName(self):
        print(self.name)


if __name__=="__main__":
    a=tree()
    a.getName('SonA')
    b=tree()
    b.getName('SonB')
    c=tree(a,b)
    c.printSon()