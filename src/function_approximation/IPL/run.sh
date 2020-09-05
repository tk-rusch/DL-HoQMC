bsub -W 04:00 -J "job_array[1-324]" "python train_ensemble.py \$LSB_JOBINDEX"
