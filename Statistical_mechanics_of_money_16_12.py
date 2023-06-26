import math
import numpy as np
import matplotlib.pyplot as plt

#Statistical_mechanics_of_money

agents=1000
steps=800001
quotient=steps//8
svalue=10.
slice=np.ones(agents)*svalue
board = np.ones((agents,steps))*svalue
player1=np.random.randint(agents,size=steps)
player2=np.random.randint(agents,size=steps)
for i in range(steps):
    a=slice[player1[i]]
    b=slice[player2[i]]
    if(a*b>0):
       slice[player1[i]]+=1
       slice[player2[i]]+=-1
    board[:,i]=slice
    
print(slice)
print(slice.sum())
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.hist(board[:,i*quotient], bins=np.linspace(0, 40, 10), density='True', label='generated dist.',color='C'+str(2*i))
plt.show()
