
date=5.3.22

# model=vgg16 
# python network_exploration.py --model-name ${model} --quantize --gradual-encode-absolute |& tee ./tmp/${date}_quantized_${model}_exploration.txt 


# model=resnet18 
# python network_exploration.py --model-name ${model} --quantize --gradual-encode-absolute |& tee ./tmp/${date}_quantized_${model}_exploration.txt 


# model=vgg16_bn 
# python network_exploration.py --model-name ${model} --quantize --gradual-encode-absolute |& tee ./tmp/${date}_quantized_${model}_exploration.txt 


# model=resnet34 
# python network_exploration.py --model-name ${model} --quantize --gradual-encode-absolute |& tee ./tmp/${date}_quantized_${model}_exploration.txt 


model=squeezenet 
python network_exploration.py --model-name ${model} |& tee ./tmp/${date}_float32_${model}_exploration.txt 


model=densenet 
python network_exploration.py --model-name ${model} |& tee ./tmp/${date}_float32_${model}_exploration.txt 


model=inception_v3 
python network_exploration.py --model-name ${model} |& tee ./tmp/${date}_float32_${model}_exploration.txt 


model=alexnet 
python network_exploration.py --model-name ${model} |& tee ./tmp/${date}_float32_${model}_exploration.txt