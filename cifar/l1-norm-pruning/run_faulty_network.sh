

date=5.10.13
# model=vgg16
model=resnet18
# model=squeezenet

# type=faults_network_rb
# type=faults_network_rb_ps1
# type=faults_network_rb_parity_zero
# type=faults_network_rb_parity_avg
# type=faults_network_rb_ecc
type=faults_network_rb_bch

python faulty_network_rb.py --model-name ${model} --fault-type ${type} --start-trial-id 0 --end-trial-id 10 --clean-dir |& tee  ./tmp/${date}_${model}_${type}.txt

# type=faults_network_rb_parity_zero
# python faulty_network_rb.py --model-name ${model} --fault-type ${type} --start-trial-id 0 --end-trial-id 10 --clean-dir |& tee  ./tmp/${date}_${model}_${type}.txt

# type=faults_network_rb_ps1
# python faulty_network_rb.py --model-name ${model} --fault-type ${type} --start-trial-id 0 --end-trial-id 10 --clean-dir |& tee  ./tmp/${date}_${model}_${type}.txt

# type=faults_network_rb_parity_avg
# python faulty_network_rb.py --model-name ${model} --fault-type ${type} --start-trial-id 0 --end-trial-id 10 --clean-dir |& tee  ./tmp/${date}_${model}_${type}.txt

# type=faults_network_rb_adaptive
# python faulty_network_rb_adaptive.py --model-name ${model} --fault-type ${type} --start-trial-id 0 --end-trial-id 10 --configs tier2_BCH_budgets.json |& tee  ./tmp/${date}_${model}_${type}.txt

# python faulty_network_rb_adaptive.py --model-name ${model} --fault-type ${type} --start-trial-id 0 --end-trial-id 10 --configs tier2_ECC_budgets.json |& tee  ./tmp/${date}_${model}_${type}.txt