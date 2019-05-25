
date=5.23.12

# model=vgg16 
# python network_exploration.py --model-name ${model} --test-inference-speed |& tee ./tmp/${date}_quantized_${model}_exploration.txt
# model=vgg16_bn
# python network_exploration.py --model-name ${model} --test-inference-speed |& tee ./tmp/${date}_quantized_${model}_exploration.txt


#model=resnet18 
#python network_exploration.py --model-name ${model} --test-inference-speed |& tee ./tmp/${date}_quantized_${model}_exploration.txt

model=squeezenet
python network_exploration.py --model-name ${model} --quantize --test-accuracy --num-batches 0 |& tee ./tmp/${date}_quantized_${model}_exploration.txt

#model=vgg16 
#python network_exploration.py --model-name ${model} --quantize --test-inference-speed |& tee ./tmp/${date}_quantized_${model}_exploration.txt

#model=resnet18 
#python network_exploration.py --model-name ${model} --quantize --test-inference-speed |& tee ./tmp/${date}_quantized_${model}_exploration.txt

#model=squeezenet
#python network_exploration.py --model-name ${model} --quantize --test-inference-speed |& tee ./tmp/${date}_quantized_${model}_exploration.txt
# python network_exploration.py --model-name ${model} --quantize --save-weights --save-parity |& tee ./tmp/${date}_quantized_${model}_exploration.txt 
# python network_exploration.py --model-name ${model} --quantize --gradual-encode-adaptive |& tee ./tmp/${date}_quantized_${model}_exploration.txt 


# model=vgg16_bn 
# model=resnet34 
# model=inception_v3 
# python network_exploration.py --model-name ${model} |& tee ./tmp/${date}_float32_${model}_exploration.txt 

# model=alexnet 
# python network_exploration.py --model-name ${model} |& tee ./tmp/${date}_float32_${model}_exploration.txt
