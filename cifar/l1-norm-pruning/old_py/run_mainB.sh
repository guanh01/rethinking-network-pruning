
# python main_B.py --scratch ./logs/vgg16/cifar10/pruneA/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16
python main_B.py --scratch ./logs/vgg16/cifar10/pruneB/pruned.pth.tar --dataset cifar10 --arch vgg --depth 16


# python main_B.py --scratch ./logs/resnet56/cifar10/pruneA/pruned.pth.tar --dataset cifar10 --arch resnet --depth 56
# python main_B.py --scratch ./logs/resnet56/cifar10/pruneB/pruned.pth.tar --dataset cifar10 --arch resnet --depth 56
# python main_B.py --scratch ./logs/resnet56/cifar10/pruneC/pruned.pth.tar --dataset cifar10 --arch resnet --depth 56
# python main_B.py --scratch ./logs/resnet56/cifar10/pruneD/pruned.pth.tar --dataset cifar10 --arch resnet --depth 56

# python main_B.py --scratch [PATH TO THE PRUNED MODEL] --dataset cifar10 --arch resnet --depth 110
