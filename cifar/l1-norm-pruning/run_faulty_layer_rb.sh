

## vgg16 full quantized int8
python faulty_layer_rb.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/model_best.pth.tar  --data-type int8 

# ## vgg16 full quantized int8 with masking
# python faulty_layer.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/model_best.pth.tar  --data-type int8 --fault-type faults_layer_masking

# ## resnet56 full
# python faulty_layer.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/model_best.pth.tar 

## resnet56 full quantized int 8
# python faulty_layer.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/model_best.pth.tar --data-type int8




