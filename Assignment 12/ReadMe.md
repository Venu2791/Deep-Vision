## Assignment 12

In this article, David summons our attention to the Stanford's DAWNbench competition in which teams and individuals compete on aspects like training speed and cost while acheiving a benchmark accuracy on a standard dataset like CIFAR10, ImageNet etc.In particular he draws our attention to DAWNBench winning solution for Cifar-10 by David C. Page that had taken 70 seconds on a single GPU to train to 94% validation accuracy.

### DavidNet:

![DavidNet](https://github.com/Venu2791/Deep-Vision/blob/master/Assignment%2012/Davidnet.png).

DavidNet has only 8 convolution layers and 1 fully-connected layer.Each convolution layer is always followed by a Batch Normalization layer and a Relu activation, forming a block.


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

### Init_pytorch?

we use init_pytorch for Weight initialization. We want the implementation to replicate the Davidnet, thus we customize the initialization in Keras as in pytorch. One reason is that different initializations lead to different scales (or, in math terms, vector norm) of the weights. For example, one method may init weights with random numbers like, 0.1, 0.2, etc., and another may init weights to be 10x bigger, like 1, 2, etc. In general, if you make a weight 10x as big, then its gradient will become 10x smaller, since calculus tells us that dy / d(10x) = dy/10dx. As a result, to get the same training results, you need to make your learning rate much bigger.

In the code, we see the range of initialization is made between the range based on the number of channels.(fanin -Number of channels coming in).
***Note: Keras and pytorch have different initialization algorithms; keras uses glorot uniform whereas pytorch uses Kaming He init.***

### self.linear(h) * self.weight?
Weight is the scaling operation after the final fully connected layer.This scaling factor is 0.125 and hand-tuned in DavidNet, and we follow the same hyperparameter. 

### Model Subclassing:
Large custom models usually contain many redundant blocks with skip connections between them. These networks are too deep to write every layer using the functional API. Here Model / Layer subclassing comes to our rescue.
It lets the user define a network or a part of a network as a python class that inherits from tf.keras.Model and exposes the call() method for it's consumer.The call() method defines the computation that happens within the model.
Each such Model class, by the way of inheritance, also exposes the predict(), fit() and evaluate() methods that internally use the call() method.
In the Code,We define a part of a Model that we call ConvBN. This block will perform the following computation when called:

ReLU << BatchNorm << Dropout << Conv2D-3x3 << Inputs


### Another block
In the code below, we define another reusable block. This block represents a single residual block of the ResNet architecture which is two convolution blocks with a skip connection joining adding the inputs to the outputs
If we attend to the call() method below, we will notice:
- If res is false, which means there is no skip connection needed - we return ConvBN << MaxPool << Inputs - we call this h.
- If res is True however, we add h back to the output of two consecutive ConvBN blocks and then return the result.

### Defining our final Model - DavidNet
In the code below, we use the blocks defined earlier to create our final model. The architecture here looks like the following:

- ConvBN
- Res Block
- Res Block without skip connection
- Res Block
- Global Maximum Pooling
- Dense Layer with 10 outputs
- Linear Scaling

The call() method returns the following :

loss : This is the cross entropy loss between the logits (output of the network) and the labels (provided as an argument)
correct : Number of correct predictions from among the input examples.

### Learning Rate Schedule and Optimizers:

Here are some salient features about the implementation:

- **lr_schedule()** is a lambda function that takes a value x and returns the linearly interpolated value of our learning rate schedule at x. The x,y coordinates that we pass to this mimic our LR schedule which is one-cycle increasing to it's highest value at one-fifth of the epochs and then deprecating back to zero.
- **global_step** variable stores the batch number from the training across the epochs. Essentially the apply_gradients() method increements this variable each time it is called.
- **lr_func :** This lambda function returns the learning rate at the current iteration of training. It uses the *lr_schedule function to arrive at this learning rate.
- **data_aug** is a lambda function that crops randomy to 32x32 shape and then randomly flips an image to the left / right.


### Training our model :

- We collect the training set x_train and y_train into a tensorflow Dataset object. ***From_tensor_Slices*** used to create a dataset from images and labels by passing a numpy array (or tensor) into tensorflow. ***Prefetch*** - Adding a prefetch buffer can improve performance by overlapping the preprocessing of data with downstream computation. Typically it is most useful to add a small prefetch buffer (with perhaps just a single element) at the very end of the pipeline, but more complex pipelines can benefit from additional prefetching, especially when the time to produce a single element can vary.
- We also accumulate the training loss and the number of correct prediction of the iteration into epoch level variables - train_loss, train_acc. **Why Sum?**  We know that the only difference between the resulting loss values is that the average loss is scaled down with respect to the sum by the factor of B, i.e. that L<sub>SUM</sub>=B⋅L<sub>AVG</sub>, where B is the batch size. We can easily prove that the same relation is true for a derivative of any variable wrt. the loss functions:(d(L<sub>SUM</sub>)/dx=Bd(L<sub>AVG</sub>)/dx). By summing the loss, we know the loss gets scaled up by B and this is taken care by the scaling factor in the Learning Rate.So we accumlate the loss of the iteration into epoch level variables and divide it by the total number of samples.

