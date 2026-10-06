import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataloader import Train_Dataset
from models.zeroidir_model import ZeroIDIRModel
from models.losses import ZeroIDIRLosses

def train():
    # 1. Hyperparameters and Configuration
    epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading dataset...")
    train_data = Train_Dataset(
        image_dir="/content/dataset/LOL/", 
        filelist="train_list.txt", 
        patch_size=(256, 256)
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # 3. Model and Loss Initialization
    print("Initializing model and losses...")
    model = ZeroIDIRModel().to(device)
    criterion = ZeroIDIRLosses().to(device)
    
    # Dual optimizers to prevent gradient conflicts between stages
    optimizer_stage1 = optim.Adam(model.stage1.parameters(), lr=learning_rate)
    optimizer_unet = optim.Adam(model.unet.parameters(), lr=learning_rate)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_stage1_loss = 0.0
        epoch_diff_loss = 0.0

        for i, batch in enumerate(train_loader):
            low_img = batch["low_img"].to(device)
            
            # --- Forward Pass Stage 1: Illumination Correction ---
            optimizer_stage1.zero_grad()
            
            stage1_enhanced = model.stage1(low_img)
            ill_map = torch.max(low_img, dim=1, keepdim=True)[0]
            
            stage1_loss, l_spa, l_exp, l_col, l_tv = criterion.compute_stage1_loss(
                stage1_enhanced, low_img, ill_map
            )
            
            stage1_loss.backward()
            optimizer_stage1.step()
            
            # --- Forward Pass Stage 2: Diffusion ---
            optimizer_unet.zero_grad()
            
            # Detach to prevent diffusion gradients from corrupting Stage 1's zero-reference weights
            pseudo_clean = stage1_enhanced.detach() 

            t = torch.randint(0, model.diffusion.num_timesteps, (low_img.shape[0],), device=device).long()
            noise = torch.randn_like(pseudo_clean)
            
            # Forward diffusion on the pseudo-clean image
            x_t = model.diffusion.q_sample(x_start=pseudo_clean, t=t, noise=noise)
            
            # Predict noise using the UNet conditioned on Stage 1
            pred_noise, pred_x_start = model.diffusion.model_predictions(x_t, t, pseudo_clean * 2 - 1)
            
            # Compute diffusion consistency loss
            diffusion_loss = criterion.compute_diffusion_consistency_loss(
                pred_noise, noise, pred_x_start, pseudo_clean
            )
            
            diffusion_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.unet.parameters(), 1.0)
            optimizer_unet.step()

            # Tracking
            epoch_stage1_loss += stage1_loss.item()
            epoch_diff_loss += diffusion_loss.item()

        # Print epoch statistics
        avg_stage1 = epoch_stage1_loss / len(train_loader)
        avg_diff = epoch_diff_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Stage 1 Loss: {avg_stage1:.4f} | Diff Loss: {avg_diff:.4f}")

        # 5. Save Checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"zeroidir_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    train()
