# Assignment 4

## Problem Statement :

The task is to get 99.4% validation accuracy in the model.

## Constraints:
 * Number of parameters should be less than 15000.
 * Start from a Vanilla DNN without BN, Drop-Out etc.
 * Each Iteration can only have max 2 improvements over previous one.
 
 ### Step 1 :
 
 First, we start with a vanilla DNN, the objective here is ro fix the archietecture for the problem statement.

**Architecture:**

Network has 8 Convolution layers, which inturn placed in 2 blocks and one max pooling layer (Transition layer). The network has 13,664 parameters. For each layer, the convoultion and receptive field is shown below.

28X28X1 | 3X3X1X8 -> 26X26X8       | GRF - 3 </br>
26X26X8 | 3X3X8X16 -> 24X24X16     | GRF - 5 </br>
24X24X16 | 3X3X16X32 -> 22X22X32   | GRF - 7 </br>
22X22X32 ->Max Pooling -> 11X11X32 | GRF - 14 </br>
11X11X32 | 1X1X10 -> 11X11X10      | GRF - 14 </br>
11X11X10 | 3X3X10X10 -> 9X9X10     | GRF -16 </br>
9X9X10   | 3X3X10X16 -> 7X7X16     | GRF -18 </br>
7X7X16   | 1X1X10X10 -> 7X7X10     | GRF-18 </br>
7X7X10   | 3X3X10X10 -> 1X1X10     | GRF-24 </br>

Number of Parameter : 13,664 

Train data was 60000 images and 10000 images on the test set. The model was trianed with batch size of 64 for 15 epochs. 

##### Results of the model:
*   Achieved the basic archietecture with less than 15k parameters. 
*   The best error rate achieved was 1.16%. 
*   We have have achieved 99.38% in the training accuracy which leaves with .62% to improve the model while the validation accuracy has to go up by 1.16%. 
*   One more option is to find a possibility wherein reduce the gap between the train and test accuracy if possible as this will reduce overfitting giving us more chance to improvize the model. 

 ### Step 2 :
 
 In this step, we introduce batch normalization and drop-out.
 
**Batch Normalization?**
Normalizing the input of your network is a well-established technique for improving the convergence properties of a network. Is so, the second layer in the network accepts the activations from our first layer as input. Thus,one could posit that normalizing these values will help the network more effectively learn the parameters in the second layer.This ensures the data for the across layer to be on the same scale.  Batch Normalization reduces the problem of ![internal covariate shift](https://mc.ai/batch-normalization/).

![Batch Normalization](https://github.com/Venu2791/Deep-Vision/blob/master/Assignment4/BN.png)


**Drop-Out**:

Dropout is a technique where randomly selected neurons are ignored during training. They are “dropped-out” randomly. This means that their contribution to the activation of downstream neurons is temporally removed on the forward pass and any weight updates are not applied to the neuron on the backward pass.

The effect is that the network becomes less sensitive to the specific weights of neurons. This in turn results in a network that is capable of better generalization and is less likely to overfit the training data.

![Drop Out](https://github.com/Venu2791/Deep-Vision/blob/master/Assignment4/Dropout.gif)



#### Layers in Step 2 :

28X28X1 | 3X3X1X8 -> 26X26X8       | GRF - 3 </br>
Batch Normalization()
model.add(Dropout(0.1))
26X26X8 | 3X3X8X16 -> 24X24X16     | GRF - 5 </br>
Batch Normalization()
24X24X16 | 3X3X16X32 -> 22X22X32   | GRF - 7 </br>
Batch Normalization()
22X22X32 ->Max Pooling -> 11X11X32 | GRF - 14 </br>
11X11X32 | 1X1X10 -> 11X11X10      | GRF - 14 </br>
Batch Normalization()
11X11X10 | 3X3X10X10 -> 9X9X10     | GRF -16 </br>
Batch Normalization()
model.add(Dropout(0.1))
9X9X10   | 3X3X10X16 -> 7X7X16     | GRF -18 </br>
Batch Normalization()
model.add(Dropout(0.1))
7X7X16   | 1X1X10X10 -> 7X7X10     | GRF-18 </br>
7X7X10   | 3X3X10X10 -> 1X1X10     | GRF-24 </br>

Total params: 14,032
Trainable params: 13,848
Non-trainable params: 184


##### Results of Step 2:

* Compared to our current iteration, we could infer that our first iteration was overfitted and with introduction of batch normalization and drop-out, we are able to reduce the gap between the train and test accuracy.
* The error rate is .96% (99.04% accuracy).
* Adding to the above, training accuracy is at 99.31%, now we have a scope of .69% improvement to achieve the target of 99.4% in validation accuracy.

 ### Step 3 :


**Upgrades in the step 3 :**

1. **Drop-out:** 
We have introduced drop-out in each layer of the model. But one important point to understand is that we cannot retain the same value as the previous iteration. Since we are using it across all layers, it is better to start with a minimum value and then increase it. An ideal aproach is to run some epoch for multiple drop-out values in the model and then pick the ideal value.

2. **Learning Rate - Step Decay :**
High learning rate will lead to random to and fro moment of the vector around local minima while a slow learning rate results in getting stuck into false minima. Thus, knowing when to decay the learning rate can be hard to find out.We base our experiment on the principle of step decay. Here, we reduce the learning rate by a constant factor every  epochs. 

Mathematical form of Step Decay:
lr = lr0 * drop^floor(epoch / epochs_drop) 

*To implement this in Keras, we can define a step decay function and use LearningRateScheduler callback to take the step decay function as argument and return the updated learning rates for use in SGD optimizer.*

The architecture and parameters are very similar to the previous iteration.

##### Results of Step 3:

* The best error rate achieved is .60%.   

* Drop-out across layers has definitely help over-fitting in the model. The gap between the train and validation has reduced significantly from the previous iterations.

* We see decreasing the learning rate over epochs,If we constantly keep a learning rate high, we could overshoot these areas of low loss as we’ll be taking too large of steps to descend into those series. 


### Step 4 :

The Final was the increment in the Batch Size.

An ideal approach would be to give entire traning data in one shot. Since that comes with a high cost, we train in mini batches. Here, we tried to increase the batch size to 512 from 64. 

##### Results of Step 4:

 We are able to hit validation accuracy of 99.4% constantly and with the best of 99.50% in this iteration. However, compared to the previous iteration we see a slight overfitting as the gap between train and validation has increased slightly.

**Summary:**

The problem statement was to hit the target of 99.4% in validation with less 15k params.

We have achieved the result in four steps:

1. Started with a vanilla CNN to establish the model architecture. - 98.84%
2. Introduced the batch normalization and Drop-out(for regularization). - 99.13%
3. Structured the drop-out across all layers and introduced the decaying learning rate. - 99.40%
4. In addition to the above, increased the batch size to 512 from 64. - 99.49%



