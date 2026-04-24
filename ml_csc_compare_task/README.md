# ML_CSC_COMPARE_TASK
 
comparing deep learning model training on a **local machine (CPU)** vs **CSC Puhti Supercomputer (GPU)**. The model is a CNN trained on the MNIST dataset using PyTorch.
 
**[Open Full Report (PDF)](https://drive.google.com/file/d/16sbkVwCxVuezAfe_eDBnJWAFt_cMFprl/view?usp=sharing)**

## Project Structure
 
```
ML_CSC_COMPARE_TASK/
│
├── train.py                  # Main training script (runs on both local and Puhti)
├── job.sh                    # SLURM batch script for submitting job on CSC Puhti
├── README.md                 # This file
│
└── result/
    ├── results_local.json    # Training metrics from local machine (CPU)
    ├── results_puhti.json    # Training metrics from CSC Puhti (GPU V100)
    ├── job_34144029.out      # SLURM job stdout log (Puhti output)
    └── job_34144029.err      # SLURM job stderr log (error log)