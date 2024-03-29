# Goal


| Accuracy            | Param storge | Inference time |
| :---                | :---         |  :--           |
| ~0.95               |  <10M        |   < 0.005s     |

Reason: In real projects, accuracy is the most important metric. Howerver, due to the limited computational resouce, 
we may have some constraints, two most common constraints are storage and inference time.

## Notes:

In this project, I use the fashion mnist test set as the validation set because this is a "toy" problem. However, in real world applications the validation set is necessary.



## Step 1: Check data distribution

Training set:

![alt text](https://github.com/QtSignalProcessing/fashion_mnist/blob/master/resource/train_dist.png)

Test set:
![alt text](https://github.com/QtSignalProcessing/fashion_mnist/blob/master/resource/test_dist.png)

Both the training and test set have balanced samples in each class. 



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


## Step 3: Try a more complex model

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



## Step 4: Model compression

CNN architecture: Resnet-14 like

input -> conv1 -> 3 x residual blocks (4) -> fc

         3x3 ,32       [ 32, 64, 128 ]       10


Optimizer: SGD + Momentum with learning rate decay

learning rate: lr = 0.001  

Momentum = 0.9

learning rate decay: at epchoch 20, 40 with parameter gamma = 0.1

| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | **Batch size**|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--      |
| Resnet-14 like      | 2.81M      | 0.995             | 0.936         | 42    |    98.7s                |    0.002s              | 4   |


Analysis:

1. Resnet-14 like also can learn the training set well (0.99+ training accurcary).

2. Compared to previous Resnet-14, this model has less parameters and is faster (training).

3. The model is overfitted.

Next step:

Try some methods to reduce overfitting.


## Step 5: Error analysis

Before trying some methods to reduce overfitting, it is neccesary to analyze some mis-classified images in order to find a direction.

![alt text](https://github.com/QtSignalProcessing/fashion_mnist/blob/master/resource/error_dist.png)

The image above shows the distribution of mis-classified images for each class. Around 17.5% shirt images and 12.5% images of T-shirts are mislabeled, these two types are dominant. 

![alt text](https://github.com/QtSignalProcessing/fashion_mnist/blob/master/resource/T-shirt_error.png)

Taking a detailed view of mis-labeled shirt images, images of shirt are mostly mis-classified as T-shirt, pullover, dress and coat.

It is reasonable to think that adding some more training images could boost the model performace.

Next step:

Use data augmentation.


## Step 6: Data augmentation

There are tones of data augmentation methods, howver, our choice of methods should not add too much unwanted noise into the training process. For example, there is no rotated image in the test set, rotation is not a good choice to augment training images.
My choice of data augmentation method is horizontal flipping the training images with a probability of 0.5. 


CNN architecture: Resnet-14 like

input -> conv1 -> 3 x residual blocks (4) -> fc

         3x3 ,32       [ 32, 64, 128 ]       10


Optimizer: SGD + Momentum with learning rate decay

learning rate: lr = 0.001  

Momentum = 0.9

learning rate decay: at epchoch 20, 40 with parameter gamma = 0.1

| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | **Batch size**|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--      |
| Resnet-14 like      | 2.81M      | 0.999             | 0.9435         | 106    |    98.7s                |    0.002s              | 4   |


Analysis:

1. The model is still overfitted.

2. Simple data augmentaion cannot help the model to perfectly distinguish T-shirt and shirt.


## TODO:

Siamese network or triplet loss could be the next direction since these two ideas are designed for distinguishing "hard" examples.












