import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import the custom modules you created
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
    # Update these paths if your Colab structure differs
    train_data = Train_Dataset(
        image_dir="/content/dataset/LOL/", 
        filelist="train_list.txt", 
        patch_size=(512, 512)
    )
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # 3. Model and Loss Initialization
    print("Initializing model and losses...")
    model = ZeroIDIRModel().to(device)
    criterion = ZeroIDIRLosses().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_loader):
            low_img = batch["low_img"].to(device)
            
            optimizer.zero_grad()

            # --- Forward Pass Stage 1: Illumination Correction ---
            stage1_enhanced = model.stage1(low_img)
            
            # Reconstruct the illumination map for the smoothness loss calculation
            ill_map = torch.max(low_img, dim=1, keepdim=True)[0]
            
            # Compute Stage 1 Zero-Reference Losses
            stage1_loss, l_spa, l_exp, l_col, l_tv = criterion.compute_stage1_loss(
                stage1_enhanced, low_img, ill_map
            )

            # --- Forward Pass Stage 2: Diffusion ---
            # Sample random timesteps for the batch
            t = torch.randint(0, model.diffusion.num_timesteps, (low_img.shape[0],), device=device).long()
            
            # Generate target noise and add it to the stage 1 output (forward diffusion)
            noise = torch.randn_like(stage1_enhanced)
            x_t = model.diffusion.q_sample(x_start=stage1_enhanced, t=t, noise=noise)
            
            # Predict the noise using the Unet conditioned on Stage 1
            pred_noise, _ = model.diffusion.model_predictions(x_t, t, stage1_enhanced * 2 - 1)
            
            # Compute Stage 2 Diffusion Loss (MSE between predicted noise and actual noise)
            diffusion_loss = torch.nn.functional.mse_loss(pred_noise, noise)

            # --- Backpropagation ---
            total_loss = stage1_loss + diffusion_loss
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average Total Loss: {avg_loss:.4f}")

        # 5. Save Checkpoints
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"zeroidir_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print("Training complete!")

if __name__ == "__main__":
    train()
