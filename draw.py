import matplotlib.pyplot as plt
import numpy as np
x = np.log([10, 100, 1000, 10000])

train = [-122.5416, -110.3160, -101.0720, -95.4374]
test = [-131.6493, -116.0917, -105.3082, -98.5283]
print(len(train))
plt.plot(x,train, label='Train')
plt.plot(x,test, label='Test')
plt.legend()
plt.show()
print(x)