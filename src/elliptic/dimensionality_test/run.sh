bsub -W 04:00 -J "job_array[1-972]" "python train_ensemble.py \$LSB_JOBINDEX"
