import sys
import turtle
import random
import time

def midvalue(x1, y1, x2, y2):
    value = ((x1+(x2-x1)/2),(y1+(y2-y1)/2))
    return value

t = turtle.Pen()
x = [100, 200, 0, 60]
y = [180, 0, 0, 60]
a = [100, 180]
b = [200, 0]
c = [0, 0]
##plt.scatter(x, y)
##plt.show()
##pointer = [3,3]
x1 = [3]
y1 = [3]
i = 0
while(1):
    dice = random.randint(1, 6)
##    print('dice')
##    print(dice)
    if (dice == 1 or dice == 2):
##        print('here1')
##        print(x1[i], y1[i])
        v = midvalue(x1[i], y1[i], 100, 180)
        x1.append(v[0])
        y1.append(v[1])
        t.goto(v[0], v[1])
        t.pendown()
        t.dot(5)
        t.penup()
##        print(x1, y1)
    elif(dice == 3 or dice == 4):
##        print('here2')
##        print(x1[i], y1[i])
        v = midvalue(x1[i], y1[i], 200, 0)
        x1.append(v[0])
        y1.append(v[1])
        t.goto(v[0], v[1])
        t.pendown()
        t.dot(5)
        t.penup()
##        print(x1, y1)
    elif(dice == 5 or dice == 6):
##        print('here3')
##        print(x1[i], y1[i])
        v = midvalue(x1[i], y1[i], 0, 0)
        x1.append(v[0])
        y1.append(v[1])
        t.goto(v[0], v[1])
        t.pendown()
        t.dot(5)
        t.penup()
##        print(x1, y1)
##    plt.close()
##    plt.scatter(x1, y1)
##    plt.show()
##    plt.pause(0.001)
    print('i')
    print(i)
    i+=1
        
