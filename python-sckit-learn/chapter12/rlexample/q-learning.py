# Practical Machine learning
# Reinforcement learning - Q learning example 
# Chapter 12
 
import matplotlib.pyplot as plt
 
class QBinaryTree(object):
    def __init__(self):
        self.right = None
        self.left = None
        self.parent = None
        self.reward = 0
        self.value = None
        self.depth = None
        self.Qval = 0
        self.Gdata = []
 
    def setLeft(self, node):
        self.left = node
        self.left.value = "Discharge"
 
    def setRight(self, node):
        self.right = node
        self.right.value = "Charge"
 
    def setParent(self, node):
        self.parent = node
 
    def getParent(self):
        return self.parent
 
    def getLeft(self):
        return self.left
 
    def getRight(self):
        return self.right
 
    def getTReward(self, Pwt, Dt, Rtc, Rtd, k, gamma):
 
        if self.getDepth() == 3:
            temp = self
            reward = 0
            while temp.getParent():
                n = temp.getDepth()
                n -= 1
                if temp.value == "Charge":
                    reward += gamma ** (n) * k * (Pwt[n] - Rtc)
 
                if temp.value == "Discharge":
                    reward += gamma ** (n) * (Pwt[n] / Dt[n]) * (Dt[n] - Rtd)
 
                temp = temp.getParent()
            return reward
        else:
            print("Not to be used for this purpose")
 
    def getDepth(self):
        cnt = 0
        temp = self
        while temp.getParent():
            cnt += 1
            temp = temp.getParent()
 
        return cnt
 
    def __str__(self):
        stringer = []
        tempstring = ""
        temp = self
        while temp.getParent():
            stringer.append(temp.value)
            temp = temp.getParent()
        stringer.reverse()
        for i in range(len(stringer)):
            tempstring += stringer[i] + ">>"
        return tempstring
 
   def DepnBtree(number):
        """Construct a binary tree of depth n for forseeing a depth of n"""
        number = 2 ** (number + 1) - 1
        root = QBinaryTree()
        queue = [root]
        counter = 1
        status = True
        while status:
            if len(queue) > 0 and counter != number:
                temp = queue.pop(0)
                temp.setLeft(QBinaryTree())
                temp.setRight(QBinaryTree())
                temp.getLeft().setParent(temp)
                temp.getRight().setParent(temp)
                queue.append(temp.getRight())
                queue.append(temp.getLeft())
                counter += 2
 
            else:
                status = False
        return root
 
   def QvalTable(root, Pwt, Dt, Rtd, Rtc, k, gamma, alpha):
        stack = [root]
        while len(stack) != 0:
            temp = stack.pop()
 
            if temp.getDepth() == 3:
                RF = temp.getTReward(Pwt, Dt, Rtd, Rtc, k, gamma)
                temp.Qval += alpha * (RF - temp.Qval)
 
                if len(temp.Gdata) == 0:
                    temp.Gdata.append(temp.Qval)
                elif int(temp.Qval) == int(temp.Gdata[len(temp.Gdata) - 1]):
                    print("Optimal Qval for sequence:" + str(temp) + "is")
                    print(temp.Gdata)
                    plt.show(temp.Gdata)
                    plt.show()
 
                #a="Optimal Qval for sequence:"+str(temp)+"is"
                #b=temp.Gdata
                #for
                #dic.keys()
                #if len(dic)==8:
                #    for i in dic.keys()
                #        print i,dic[i]
                else:
                    temp.Gdata.append(temp.Qval)
            if temp.getRight():
                stack.append(temp.getRight())
            if temp.getLeft():
                stack.append(temp.getLeft())
 
P = [6000.0, 4800.0, 4800.0]
D = [2800.0, 2800.0, 2800.0]
k = 1
g = 0.8
Rd = 1000
Rc = 1000
al = 0.8
d = input("Please Enter Depth of Prediction:")
mnode = DepnBtree(d)
d = input("Enter the number of iterations:")
for i in range(int(d)):
    QvalTable(mnode, P, D, Rd, Rc, k, g, al