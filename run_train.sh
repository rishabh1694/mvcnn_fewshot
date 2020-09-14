dataset=ModelNet
model=ResNet18
method=protonet
#n_shot=1
#num_views=4
n_query=8

for n_shot in 1 5; do
	for num_views in 1 5 10; do
	#for lr in 0.001 0.0005 0.0001 0.00005; do
		for lr in 0.0005 0.0001; do
			echo "${dataset},${model},${method},${n_shot},${num_views},${n_query},${lr}"
			name=${dataset}_${model}_${method}_${n_shot}_${num_views}_${n_query}_lr${lr}
			#name=${dataset}_${model}_${method}_${n_shot}_None_lr${lr}
        		out_path=slurm_out/${name}
			err_path=slurm_err/${name}
			export dataset model method n_shot num_views n_query lr
			sbatch -p 1080ti-long -o ${out_path}.out -e ${err_path}.err b_train.sbatch
		done
	done
done
