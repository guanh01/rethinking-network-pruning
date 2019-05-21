# python main.py --dataset cifar10 --arch vgg --depth 11
# python main.py --dataset cifar10 --arch resnet --depth 56
# python main.py --dataset cifar10 --arch resnet --depth 110


date=5.20.10
model=squeezenet
python main.py --dataset cifar10 --arch ${model} --lr 0.001 --epochs 500 |& tee ./tmp/${date}_train_${model}.txt
