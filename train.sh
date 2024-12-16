python train_brain_region_constrative_slurm.py  \
                        -gpus 0 \
                        -cfg 'config/brain_region_unet.yaml' \
                        -slurm \
                        -slurm_ngpus 2 \
                        -slurm_nnodes 1 \
                        -slurm_nodelist c003 \
                        -slurm_partition compute \