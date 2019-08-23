# fashion_mnist

Strategy 1:

Optimizer: SGD + Momentum

learning rate: lr = 0.001

| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | **Batch size**|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--
| Resnet-18           | 11.1M      | 0.992             | 0.941         | 43    |    -                    | -              | 4            |


---
| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | **Batch size**|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--
| Resnet-18           | 11.1M      | 0.999             | 0.91         | -    |    -                    | -              |  128           |

---



Strategy 2:

Optimizer: SGD + Momentum

learning rate: lr = 0.001

| Network             | #Params    | Training accuracy | Test accuracy | Epoch | Training time per epoch | Inference time | **Batch size**|
| :---                | :---       | :---              | :---          | :---  | :---                    |   :--           | :--      |
| Resnet-18 like      | 2.81M      | 0.995             | 0.935         | 42    |    -                    | -              | 4   |



Structure 1:       

input ->  conv2d_relu->max_pool -> conv2d_relu->max_pool->conv2d_relu->max_pool->fc1->fc2

28x28x1   28x28x32     14x14x32    14x14x64     7x7x64    7x7x128      3x3x128-> 512-> 10

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


