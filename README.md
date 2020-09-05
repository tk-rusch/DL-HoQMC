# Higher-order Quasi-Monte Carlo Training of Deep Neural Networks
This repository contains the implementation to reproduce the numerical experiments 
of the paper [Higher-order Quasi-Monte Carlo Training of Deep Neural Networks](https://arxiv.org)


## Installation
Please make sure you have installed the requirements before executing the python scripts.

```bash
pip install numpy
pip install matplotlib
pip install pytorch
```

## Data

The data required for each experiment is stored in the *data* folder and handled by Git LFS (Large File Storage), 
as it exceeds the maximum storage capacity of github. 

## Excecuting the code
The code to reproduce each experiment in the paper can be found in the source folder.
 
In each experiment directory run

     python plot_results.py 
     
to plot the stored results of the already trained neural network ensemble. 

To retrain the ensemble, run
    
    python run_script <id>

where \<id> corresponce to the *id* of each neural network in the 
ensemble specified in the correspodning *run_script.py* file.
The new results of the trained network ensemble will be stored in *new_results* directory. 
In order to plot the new results, change the data source path in the corresponding *plot_results.py* file accordingly.

Note, that retraining the ensemble sequentially may require a very long time. 
Instead, if applicable, one may run the whole ensemble in parallel. 

If applicable, simply run

    bash run.sh
    
for each experiment. This will generate and submit a job-array, 
where each element corresponce to one neural network in the ensemble.