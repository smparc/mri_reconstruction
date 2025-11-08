# Accelerated MRI Reconstruction with SwinUNet

This project implements and evaluates deep learning models for accelerated MRI reconstruction from undersampled k-space data, as described in the paper "Accelerated MRI Reconstruction with SwinUNet".

It includes implementations for:
* [cite_start]A `BaselineUNet` [cite: 102]
* [cite_start]A `SwinUNet` [cite: 127]

## 🛠 Installation

### 1. Setup
Create a Python environment and install PyTorch. You **must** install PyTorch first. Visit [pytorch.org](https://pytorch.org/) for the command specific to your system/CUDA version.

### 2. Install Dependencies
Install the `fastmri` library and other requirements.
```bash
pip install fastmri
pip install -r requirements.txt
```

### 3. Download the Data
You must download the fastMRI single-coil knee dataset.
1. visit the fastMRI dataset page: https://fastmri.med.nyu.edu/
2. Download knee_singlecoil_train.zip (training set) and knee_singlecoil_val.zip (validation set)
3. Unzip them to a known location

## How to Run
1. Configure Paths
This is the most important step. Open train.py and update these two variables to point to your unzipped dataset directories:
```bash
DATA_PATH = '/path/to/your/knee_singlecoil_train'
VAL_DATA_PATH = '/path/to/your/knee_singlecoil_val'
```

2. Select Model
In train.py, you can choose which model to train:
```bash
MODEL_CHOICE = 'swinunet'  # or 'unet'
```

3. Run Training
Execute the main training script:
```bash
python train.py
```

The script will train the model, reporting validation PSNR and SSIM after each epoch. The best-performing model will be saved as swinunet_best_model.pth.
