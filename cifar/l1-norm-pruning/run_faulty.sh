
for fault_rate in 1e-4 1e-5 5e-5 1e-6 5e-6 1e-7 5e-7 1e-8 5e-8 
do 

## vgg16 full 
# python faulty.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/model_best.pth.tar --fault-rate ${fault_rate} --fault-type faults_param
# python faulty.py --dataset cifar10  --model ./logs/vgg16/cifar10/model_best.pth.tar --fault-rate ${fault_rate} --fault-type zero_bit_masking

# pruneA
# python faulty.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/prune/scratchB/model_best.pth.tar --fault-rate ${fault_rate} --fault-type faults_param_mimic --mimic-stats ./logs/vgg16/cifar10/faults_param
# python faulty.py --dataset cifar10  --model ./logs/vgg16/cifar10/prune/scratchB/model_best.pth.tar --fault-rate ${fault_rate} --fault-type zero_bit_masking

# pruneB
python faulty.py --dataset cifar10  --arch vgg --depth 16 --model ./logs/vgg16/cifar10/pruneB/scratchB/model_best.pth.tar --fault-rate ${fault_rate} --fault-type faults_param_mimic --mimic-stats ./logs/vgg16/cifar10/faults_param




## resnet56 full
# python faulty.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/model_best.pth.tar --fault-rate ${fault_rate} --fault-type faults_param
# python faulty.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/model_best.pth.tar --fault-rate ${fault_rate} --fault-type zero_bit_masking

# pruneA
# python faulty.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneA/scratchB/model_best.pth.tar --fault-rate ${fault_rate} --fault-type faults_param_mimic --mimic-stats ./logs/resnet56/cifar10/faults_param
# python faulty.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneA/scratchB/model_best.pth.tar --fault-rate ${fault_rate} --fault-type zero_bit_masking

# pruneB
# python faulty.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneB/scratchB/model_best.pth.tar --fault-rate ${fault_rate} --fault-type faults_param_mimic --mimic-stats ./logs/resnet56/cifar10/faults_param
# python faulty.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneB/scratchB/model_best.pth.tar --fault-rate ${fault_rate} --fault-type zero_bit_masking

# pruneD
# python faulty.py --dataset cifar10  --arch resnet --depth 56 --model ./logs/resnet56/cifar10/pruneD/scratchB/model_best.pth.tar --fault-rate ${fault_rate} --fault-type faults_param_mimic --mimic-stats ./logs/resnet56/cifar10/faults_param

done 

