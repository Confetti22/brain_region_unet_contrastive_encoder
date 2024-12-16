python train_brain_region_constrative_slurm.py  \
                        -gpus 0 \
                        -save_every 50 \
                        -cfg 'config/brain_region_unet.yaml' \
                        -out out \
                        -slurm \
                        -slurm_ngpus 1 \
                        -slurm_nnodes 1 \
                        -slurm_nodelist c003 \
                        -slurm_partition compute \