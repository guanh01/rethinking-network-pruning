# python vggprune.py --dataset cifar10  --model ./logs/vgg16/cifar10/model_best.pth.tar
python vggprune.py --dataset cifar10  -v B --model ./logs/vgg16/cifar10/model_best.pth.tar

# python res56prune.py --dataset cifar10 -v A --model ./logs/resnet56/cifar10/model_best.pth.tar
# python res56prune.py --dataset cifar10 -v B --model ./logs/resnet56/cifar10/model_best.pth.tar
# python res56prune.py --dataset cifar10 -v C --model ./logs/resnet56/cifar10/model_best.pth.tar
# python res56prune.py --dataset cifar10 -v D --model ./logs/resnet56/cifar10/model_best.pth.tar


# python res110prune.py --dataset cifar10 -v A --model [PATH TO THE MODEL] --save [DIRECTORY TO STORE RESULT]
