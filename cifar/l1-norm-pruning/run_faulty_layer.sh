

## vgg16 full 
python faulty_layer.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/model_best.pth.tar 
python faulty_layer.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/prune/scratchB/model_best.pth.tar
python faulty_layer.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/pruneB/scratchB/model_best.pth.tar

## vgg16 full quantized int8
python faulty_layer.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/model_best.pth.tar  --data-type int8

## vgg16 full quantized int8 with masking
python faulty_layer.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/model_best.pth.tar  --data-type int8 --fault-type faults_layer_masking

## resnet56 full
python faulty_layer.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/model_best.pth.tar 

# pruneA
python faulty_layer.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneA/scratchB/model_best.pth.tar 

# pruneB
python faulty_layer.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneB/scratchB/model_best.pth.tar 

# pruneC
python faulty_layer.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneC/scratchB/model_best.pth.tar 

# pruneD
python faulty_layer.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneD/scratchB/model_best.pth.tar 



