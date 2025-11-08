import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import glob
from tqdm import tqdm
import time

from dataset import MRIDataset, get_mask_func
from models import BaselineUNet, SwinUNet
from ssim import CombinedLoss, psnr, SSIM

# --- !!! ACTION REQUIRED !!! ---
# Update these paths to point to your fastMRI dataset
DATA_PATH = '/path/to/fastmri/knee_singlecoil_train'
VAL_DATA_PATH = '/path/to/fastmri/knee_singlecoil_val'

# --- Configuration ---
MODEL_CHOICE = 'swinunet' # 'unet' or 'swinunet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 60 # [cite: 150]

# Hyperparameters based on best SwinUNet [cite: 202-206, 256]
BATCH_SIZE = 6 # [cite: 206]
LEARNING_RATE = 8e-5 # [cite: 256]
WEIGHT_DECAY = 1e-5 # [cite: 147]
LAMBDA_L1 = 1.0 # 
LAMBDA_SSIM = 1.0 # 

def train_epoch(model, dataloader, optimizer, loss_fn):
    model.train()
    running_loss = 0.0
    
    for input_img, target_img in tqdm(dataloader, desc="Training"):
        input_img = input_img.to(DEVICE, non_blocking=True)
        target_img = target_img.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        prediction = model(input_img)
        loss = loss_fn(prediction, target_img)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    return running_loss / len(dataloader)

def validate_epoch(model, dataloader, loss_fn):
    model.eval()
    running_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    ssim_metric = SSIM(window_size=11).to(DEVICE)
    
    with torch.no_grad():
        for input_img, target_img in tqdm(dataloader, desc="Validating"):
            input_img = input_img.to(DEVICE, non_blocking=True)
            target_img = target_img.to(DEVICE, non_blocking=True)
            
            prediction = model(input_img)
            loss = loss_fn(prediction, target_img)
            running_loss += loss.item()
            
            pred_clamped = torch.clamp(prediction, 0.0, 1.0)
            target_clamped = torch.clamp(target_img, 0.0, 1.0)

            total_psnr += psnr(target_clamped, pred_clamped, max_val=1.0).item()
            total_ssim += ssim_metric(pred_clamped, target_clamped).item()
            
    return (
        running_loss / len(dataloader),
        total_psnr / len(dataloader),
        total_ssim / len(dataloader)
    )

def main():
    start_time = time.time()
    print(f"Using device: {DEVICE}")

    # --- 1. Data ---
    print("Setting up data...")
    mask_func = get_mask_func()
    train_files = sorted(glob.glob(os.path.join(DATA_PATH, '*.h5')))
    val_files = sorted(glob.glob(os.path.join(VAL_DATA_PATH, '*.h5')))
    
    if not train_files or not val_files:
        print(f"Error: No .h5 files found in DATA_PATH or VAL_DATA_PATH.")
        print("Please update the paths at the top of train.py")
        return
        
    print(f"Found {len(train_files)} training files.")
    print(f"Found {len(val_files)} validation files.")

    train_dataset = MRIDataset(train_files, mask_func) # [cite: 90]
    val_dataset = MRIDataset(val_files, mask_func)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- 2. Model ---
    print(f"Initializing model: {MODEL_CHOICE}")
    if MODEL_CHOICE == 'unet':
        model = BaselineUNet(in_ch=1, out_ch=1).to(DEVICE)
    elif MODEL_CHOICE == 'swinunet':
        model = SwinUNet(
            img_size=(320, 320), patch_size=4, in_chans=1, out_chans=1,
            embed_dim=64, depths=[2, 2, 6, 2], num_heads=[2, 4, 8, 16],
            window_size=8 # [cite: 203, 254]
        ).to(DEVICE)
    else:
        raise ValueError("Unknown MODEL_CHOICE")

    # --- 3. Training Setup ---
    loss_fn = CombinedLoss(lambda1=LAMBDA_L1, lambda2=LAMBDA_SSIM).to(DEVICE) # 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # [cite: 145, 147]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) # [cite: 146]

    # --- 4. Training Loop ---
    print("Starting training...")
    best_val_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        epoch_start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_psnr, val_ssim = validate_epoch(model, val_loader, loss_fn)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1} complete in {time.time() - epoch_start_time:.2f}s")
        print(f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"Val PSNR:   {val_psnr:.4f} dB | Val SSIM: {val_ssim:.4f}")
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            save_path = f"{MODEL_CHOICE}_best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")

    print(f"\nTraining finished in {(time.time() - start_time) / 3600:.2f} hours.")
    print(f"Best validation PSNR: {best_val_psnr:.4f} dB")

if __name__ == "__main__":
    main()
