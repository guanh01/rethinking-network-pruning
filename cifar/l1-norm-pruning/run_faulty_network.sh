

date=4.28
model=vgg16
# model=resnet18
type=faults_network_rb_parity_zero
# type=faults_network_rb_parity_avg

python faulty_network_rb.py --model-name ${model} --fault-type ${type} --start-trial-id 0 --end-trial-id 10 --clean-dir |& tee  ./tmp/${date}_${model}_${type}.txt


