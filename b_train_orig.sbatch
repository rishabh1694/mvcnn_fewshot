#!/bin/bash
#
# run_train.sbatch
#
#SBATCH --job-name=maml
#SBATCH --output=res_%j.txt           # output file
##SBATCH -n 6                          # Number of cores
#SBATCH -N 1                          # Ensure that all cores are on one machine
##SBATCH -t 0-04:00                    # Runtime in D-HH:MM
##SBATCH -p titanx-long               # Partition to submit to (serial_requeue)
#SBATCH --mem=100GB                     # Memory pool for all cores (see also --mem-per-cpu)
##SBATCH --mem-per-cpu=10240
#SBATCH -o hostname_%j.out            # File to which STDOUT will be written
#SBATCH -e hostname_%j.err            # File to which STDERR will be written
##SBATCH --mail-type=FAIL               # Type of email notification- BEGIN,END,FAIL,ALL
##SBATCH --mail-user=jcsu@cs.umass.edu # Email to which notifications will be sent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task 12
##SBATCH --nodelist=node022
##SBATCH --exclude=node126,node096,node030,node023,node007,node104,node105,node127,node117,node139,node108,node152,node125,node118,node116,node081,node005,node131
##SBATCH --exclude=node036,node131,node127,node151,node119,node111,node142,node149,node141,node154,node147,node106,node138

## Training
## train_unlabel.py train.py
srun -u python train.py --dataset ${dataset} --method ${method} \
	--train_n_way ${train_n_way} --test_n_way ${test_n_way} --n_shot ${n_shot} \
	--jigsaw --lbda ${lbda} --lr ${lr} --train_aug --date ${date}\
	--image_size ${image_size} --n_query ${n_query} --model ${model} --gypsum --stop_epoch ${stop_epoch} --save_freq 200
	# --jigsaw --lbda ${lbda} --lr ${lr} --train_aug --date ${date}\
	# --rotation --lbda ${lbda} --lr ${lr} --train_aug --date ${date}\
	# --json_seed ${json_seed} --jigsaw --grey
exit

# srun -u python train.py --dataset CUB --method baseline \
# 	--lr 0.001 --train_aug --date 0522firstk\
# 	--image_size 224 --num_class 200 --model resnet18 --gypsum --stop_epoch 400 --firstk ${firstk} --tracking
# #--rotation --lbda 0.5
# 	# --tracking --jigsaw --rotation
# exit
