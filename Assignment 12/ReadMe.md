## Assignment 12

In this article, David summons our attention to the Stanford's DAWNbench competition in which teams and individuals compete on aspects like training speed and cost while acheiving a benchmark accuracy on a standard dataset like CIFAR10, ImageNet etc.In particular he draws our attention to DAWNBench winning solution for Cifar-10 by David C. Page that had taken 70 seconds on a single GPU to train to 94% validation accuracy.

### DavidNet Network:
![DavidNet]().

### Eager Mode:

TensorFlow's eager execution is an imperative programming environment that evaluates operations immediately, without building graphs: operations return concrete values instead of constructing a computational graph to run later. 
Enabling eager execution changes how TensorFlow operations behave — now they immediately evaluate and return their values to Python. Example:
```
import tensorflow as tf
x = [ [1,2],
[3,4] ]
m = tf.matmul(x,x)
print(m)
Output: Tensor (“MatMul:0”, shape = (2,2), dtype = int32);
To make any sense of what a certain chunk of the model is undergoing, one would need to run a session:
with tf.Session() as sess:
print(sess.run(m)) 
Output: [[7 10] [15 22]]
However, when eager mode is enabled :
print(m)
Would give the following output:
tf.Tensor( [ [ 7 10] [ 15 22] ], shape= (2,2), dtype = int32)
```

#### Benefits:</br>
- Eager execution works nicely with NumPy. NumPy operations accept tf.Tensor arguments. The tf.Tensor.numpy method returns the object’s value as a NumPy ndarray.
- Even without training, call the model and inspect the output in eager execution. 
Note: Eager execution cannot be enabled after TensorFlow APIs have been used to create or execute graphs. It is typically recommended to invoke this function at program startup and not in a library.

