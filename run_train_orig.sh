# # dataset=miniImagenet
# # dataset=CUB
# # json_seed=20
# # dataset=cars
# # dataset=aircrafts
# # dataset=dogs
# # dataset=flowers
# train_n_way=5
# test_n_way=5
# n_shot=5
# n_query=16
# image_size=224
# # date=0514
# # date=0517
# # date=0517grey
# # date=05181shot
# date=0519unlabel
# # date=05201shot
# for dataset in CUB; do
# # for dataset in CUB; do
#   for method in protonet; do
#     for model in resnet18; do
#     # for model in resnet18 resnet18_cbam resnet18_dualcbam; do
#       for lbda in 0.0 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
#       # for lbda in 0.0 0.5 0.75; do
#       # for lbda in 0.5; do
#       # for lbda in 0.0; do
#         # for lr in 0.001 0.0003; do
#         for lr in 0.001; do
#           echo "${dataset},${method},${lbda}"
#           # name=resnet18_${image_size}_query${n_query}_aug_jigsaw_lbda${lbda}_lr${lr}
#           # name=resnet18_${image_size}_query${n_query}_aug_jigsaw_lbda${lbda}_lr${lr}_randomResizeCrop225jigsaw2run
#           # name=${date}_${method}_${dataset}_${model}_${image_size}_query${n_query}_aug_jigsaw_lbda${lbda}_lr${lr}
#           # name=${date}_${method}_${dataset}_${model}_${image_size}_query${n_query}_aug_rotation_lbda${lbda}_lr${lr}
#           # name=${date}_${method}_${dataset}_${json_seed}_${model}_${image_size}_query${n_query}_aug_jigsaw_lbda${lbda}_lr${lr}
#           name=${date}_${method}_${dataset}_${model}_${lbda}
#           # out_path=slurm_out/${name}
#           # err_path=slurm_err/${name}
#           out_path=slurm_out/test_${name}
#           err_path=slurm_err/test_${name}
#           # out_path=slurm_out/random_${dataset}
#           # err_path=slurm_err/random_${dataset}
#           export train_n_way test_n_way n_shot n_query image_size date
#           export lbda lr model dataset method
#           # export lbda lr model dataset method json_seed
#           # sbatch -p titanx-long -o ${out_path}.out -e ${err_path}.err b_train.sbatch
#           # sbatch -p 1080ti-short -o ${out_path}.out -e ${err_path}.err b_save_feature.sbatch
#           sbatch -p 1080ti-short -o ${out_path}.out -e ${err_path}.err b_test.sbatch
#           # sleep 1
#         done
#       done
#     done
#   done
# done

# for firstk in 25 15 20; do
#   echo "${firstk}"
#   name=firstk_${firstk}_softmax
#   out_path=slurm_out/${name}
#   err_path=slurm_err/${name}
#   export firstk
#   sbatch -p 1080ti-long -o ${out_path}.out -e ${err_path}.err b_train.sbatch
#   # sbatch -p 1080ti-short -o ${out_path}.out -e ${err_path}.err b_save_feature.sbatch
#   # sbatch -p titanx-short -o ${out_path}.out -e ${err_path}.err b_test.sbatch
#   # sleep 1
# done

############### 0603 ###################
train_n_way=5
test_n_way=5
n_shot=5
# n_query=16
image_size=224
# date=0514
# date=0517
# date=0517grey
# date=05181shot
date=0922
n_query=10
stop_epoch=400
## for mixed datasets:
# n_query=5
# stop_epoch=600
## for MAML:
# n_query=10
# stop_epoch=400
# for dataset in cars aircrafts dogs flowers; do
for dataset in tieredImagenet; do
# for dataset in CUB_20 cars_20 aircrafts_20 dogs_20 flowers_20; do
# for dataset in aircrafts dogs flowers; do
  for method in maml_approx; do
  # for method in baseline baseline++; do
  # for method in protonet; do
    for model in resnet18; do
      for lr in 0.002; do
        # for lbda in 0.0 0.3 0.4 0.5 0.6 0.7 1.0; do
        # for lbda in 0.0 0.5 0.75; do
        for lbda in 0.3; do
        # for lbda in 0.0 0.5 1.0; do
        # for lr in 0.001 0.0003; do
        # for lr in 0.002 0.001; do
          echo "${dataset},${method},${lbda}"
          # name=resnet18_${image_size}_query${n_query}_aug_jigsaw_lbda${lbda}_lr${lr}
          # name=resnet18_${image_size}_query${n_query}_aug_jigsaw_lbda${lbda}_lr${lr}_randomResizeCrop225jigsaw2run
          name=${date}_${method}_${dataset}_${model}_${image_size}_query${n_query}_aug_jigsaw_lbda${lbda}_lr${lr}
          # name=${date}_${method}_${dataset}_${model}_${image_size}_query${n_query}_aug_rotation_lbda${lbda}_lr${lr}
          # name=${date}_${method}_${dataset}_${model}_${image_size}_query${n_query}_aug_lr${lr}
          # name=${date}_${method}_${dataset}_${json_seed}_${model}_${image_size}_query${n_query}_aug_jigsaw_lbda${lbda}_lr${lr}
          # name=${date}_${method}_${dataset}_${model}_${lbda}
          out_path=slurm_out/${name} 
          err_path=slurm_err/${name}
          # out_path=slurm_out/test_${name}
          # err_path=slurm_err/test_${name}
          # out_path=slurm_out/random_${dataset}
          # err_path=slurm_err/random_${dataset}
          export train_n_way test_n_way n_shot n_query image_size date
          export lbda lr model dataset method stop_epoch
          # export lbda lr model dataset method json_seed
          sbatch -p 1080ti-long -o ${out_path}.out -e ${err_path}.err b_train.sbatch
          # sbatch -p 1080ti-short -o ${out_path}.out -e ${err_path}.err b_save_feature.sbatch
          # sbatch -p 1080ti-short -o ${out_path}.out -e ${err_path}.err b_test.sbatch
          # sleep 1
        done
      done
    done
  done
done
