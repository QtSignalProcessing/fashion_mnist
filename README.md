# fashion_mnist
Structure 1: 
input ->  conv2d_relu->max_pool -> conv2d_relu->max_pool->conv2d_relu->max_pool->fc1->fc2
28x28x1   28x28x32     14x14x32    14x14x64     7x7x64    7x7x128      3x3x128-> 512-> 10
epoch = 10
training accuracy: 95%
test accuracy: 92.11%
