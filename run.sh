bash_file_name=$(basename $0)
export CUDA_VISIBLE_DEVICES=1
for dataset in "cifar100" 
do
      for seed in 0 1 2
      do
            for tta_method in "RMT" "Source" "BN" "Tent" "SAR" "CoTTA" "RoTTA" "TRIBE" "OURS"
            do
            python CTTA.py \
                  -acfg configs/adapter/${dataset}/${tta_method}.yaml \
                  -dcfg configs/dataset/${dataset}.yaml \
                  -ocfg configs/order/${dataset}/0.yaml \
                  SEED $seed \
                  TEST.BATCH_SIZE 64 \
                  bash_file_name $bash_file_name \
                  CORRUPTION.SEVERITY '[5]'
            done
      done
done
