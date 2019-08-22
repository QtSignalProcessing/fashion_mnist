# fashion_mnist
lr = 0.001

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

Resnet 18

training accuracy: 99.95%

test accuracy: 94.08%


