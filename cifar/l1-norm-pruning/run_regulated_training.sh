date=5.19.10
model=vgg16_cifar10
lambda=100.0
lr=0.1

python regulated_training.py --dataset cifar10 --arch vgg --depth 16 --warmstart /home/hguan2/workspace/fault-tolerance/rethinking-network-pruning/cifar/l1-norm-pruning/logs/vgg16/cifar10/model_best.pth.tar --resume /home/hguan2/workspace/fault-tolerance/rethinking-network-pruning/cifar/l1-norm-pruning/logs/vgg16/cifar10/regulated_training/l2_${lambda}_lr_${lr}/checkpoint.pth.tar --lr ${lr} --epochs 160 --l2 ${lambda} |& tee ./tmp/${date}_regulated_training_${model}_l2_${lambda}_lr_${lr}.txt


