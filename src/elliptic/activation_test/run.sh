bsub -W 04:00 -J "job_array[1-486]" "python train_ensemble.py \$LSB_JOBINDEX"
