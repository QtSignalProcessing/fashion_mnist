# Goal


| Accuracy            | Param storge | Inference time |
| :---                | :---         |  :--           |
| ~0.95               |  <10M        |   < 0.005s     |

Reason: In real projects, accuracy is the most important metric. Howerver, due to the limited computational resouce, 
we may have some constraints, two most common constraints are storage and inference time.

## Step 1: Check data distribution

Training set:

![alt text](https://github.com/QtSignalProcessing/fashion_mnist/blob/master/resource/train_dist.png)

Test set:
![alt text](https://github.com/QtSignalProcessing/fashion_mnist/blob/master/resource/test_dist.png)


## Step 2: Create a toy model to  better understand the data

CNN architecture: 3 conv layers and 2 fc layers. (random choice)

input ->  conv2d_relu->max_pool -> conv2d_relu->max_pool->conv2d_relu->max_pool->fc1->fc2

28x28x1  ->    28x28x32    ->        14x14x32    ->        14x14x64    ->      7x7x64    ->       7x7x128    ->       3x3x128   ->    512   ->        10


Optimizer: SGD + Momentum with learning rate decay

learning rate: lr = 0.001  

Momentum = 0.9

learning rate decay: at epchoch 20, 40 with parameter gamma = 0.1


| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | Batch size|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--
| toy cnn           | 2.75M      | 0.95             | 0.921         | 10    |    -                    | -              | 4            |


Analysis:

1. Training accuracy is not 100% -> Could try other more complex network architectures
2. Test accuracy is less than training accuracy -> The model is overfitted

Next step:

Try a larger model with the target of boosting training and test accuracies.


## Step 3:

CNN architecture: Resnet-14

input -> conv1 -> 3 x residual blocks (4) -> fc

         3x3 ,64       [ 64, 128, 256 ]       10


Optimizer: SGD + Momentum with learning rate decay

learning rate: lr = 0.001  

Momentum = 0.9

learning rate decay: at epchoch 20, 40 with parameter gamma = 0.1

| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | **Batch size**|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--
| Resnet-14           | 11.1M      | 0.992             | 0.941         | 43    |    987s                 |  0.002s             | 4            |


---
| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | **Batch size**|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--
| Resnet-14           | 11.1M      | 0.999             | 0.91         | -    |    15.9s                    | 0.002s              |  128           |

---

Analysis:

1. Resnet-14 can learn the training set well (0.99+ training accurcary).

2. This model requires 11.1M to store all parameters.

3. The model is overfitted.

4. Need to care about batch size or number of iterations.

Next step:

Try to reduce the number of parameters.



## Step 4:

CNN architecture: Resnet-14 like

input -> conv1 -> 3 x residual blocks (4) -> fc

         3x3 ,32       [ 32, 64, 128 ]       10


Optimizer: SGD + Momentum with learning rate decay

learning rate: lr = 0.001  

Momentum = 0.9

learning rate decay: at epchoch 20, 40 with parameter gamma = 0.1

| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | **Batch size**|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--      |
| Resnet-14 like      | 2.81M      | 0.995             | 0.935         | 42    |    98.7s                |    0.002s              | 4   |


Analysis:

1. Resnet-14 like also can learn the training set well (0.99+ training accurcary).

2. Compared to previous Resnet-14, this model has less parameters and is faster (training).

3. The model is overfitted.





Structure 1:       



epoch = 10

training accuracy: 95%

test accuracy: 92.11%



Structure 2: 

input ->  conv2d_relu x 3->max_pool -> conv2d_relu x 3->max_pool->conv2d_relu x 3->max_pool->fc1->fc2

28x28x1   28x28x32     14x14x32    14x14x64     7x7x64    7x7x128      3x3x128-> 512-> 10

epoch = 44

training accuracy: 98.70%

test accuracy: 91.93%

Structure 3:

Resnet-18

epoch: 43

training accuracy: 99.95%

test accuracy: 94.08%

Structure 4:
99.90833333333333 93.61 50

data aug: random crop
86.45166666666667 93.13 186

res 12

99.83 93.05 42

99.52333333333333 93.78 46
