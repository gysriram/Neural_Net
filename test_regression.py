from sys import argv
import numpy as np
import math
import nn_layer
from matplotlib import pyplot as plt
import time

print "Num of epochs?"
num_epochs=int(raw_input(">"))
print "Learning rate?"
learning_rate = float(raw_input(">"))

x=np.linspace(-2,2, num=100)

# check different func
#y=[math.exp(math.cos(x[i])) for i in range(0,len(x))]
#y=np.sign(x)
#y=[math.sin(x[i]) for i in range(0,len(x))]
y=[abs(x[i]) for i in range(0,len(x))]
batch_size=1

xo=x
x=np.reshape(x, (-1, 1))
train_x=np.reshape(x, (1, -1))

y=np.reshape(y,(len(x),1))
train_y=np.reshape(y, (1, len(x)))

noi = np.random.normal(0, 0.1, 100)
noi= np.reshape(noi,(len(x),1))
y = y+noi
plt.plot(xo,y,'g')

hidden_layer1 = nn_layer.fully_connected_layer(1, 3,"relu")
output_layer = nn_layer.fully_connected_layer(3, 1, "linear")

start=time.time()
layers = {"L1": hidden_layer1, "L2": output_layer}
model = nn_layer.nn_model(layers, learning_rate, num_epochs, batch_size)
model.train_network(train_x, train_y)
end=time.time()

model_out = model.predict(train_x)
print model_out.shape
print end-start
plt.plot(xo, model_out.reshape((-1,1)), 'b--')
plt.show()
print model_out

