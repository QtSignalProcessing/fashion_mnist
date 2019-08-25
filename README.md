# fashion_mnist

### Training:

```
python train.py

        --model         # specify model, default = toy (toy, resnet14, resnet14s)
        --lr      # learning rate, default =  0.001
        --data_aug     # data augmenation, default = False
        --batch_size    # batch size, default = 4
        --nepochs       # max epochs, default = 50
        --nworkers      # number of workers, default = 4
        --seed          # random seed, default = 1
        --model_path    # directory to store trained model, default = ./model
```

Examples:

1. Train the toy model 
```
python train.py

```

2. Train resnet-14

```

python train.py --model resnet14

```

3. Train the simplified resnet 14 with data augmentation to acheive the test accuracy of 0.9435

```
python train.py --model resnet14s --data_aug True --nepochs 110

```

### Other

Data distribution can be checked in the jupyter notebook visualize.ipynb. I uploaded two trained models: 1) resnet14 simplified without data augmentaion, which is named as fashin_mnist_resnet14_Le.pt, 2) resnet14 simplified with data augmentaion, which is named as fashin_mnist_resnet14s_hf.pt. Error analysis of the last model is available in visualize.ipynb.  



