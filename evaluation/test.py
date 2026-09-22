import os
import argparse
import glob
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as TF

from models.zeroidir_model import ZeroIDIRModel

def parse_args():
    parser = argparse.ArgumentParser(description="ZeroIDIR Evaluation and Inference")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="./checkpoints/zeroidir_epoch_50.pth", 
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        default="/content/dataset/LOL/eval15/low", 
        help="Path to input test directory or single image"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="./results", 
        help="Directory to save enhanced images"
    )
    parser.add_argument(
        "--sampling_steps", 
        type=int, 
        default=20, 
        help="DDIM sampling timesteps"
    )
    return parser.parse_args()

def run_inference():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Loading checkpoint: {args.weights}")

    # 1. Initialize and load model weights
    model = ZeroIDIRModel(sampling_timesteps=args.sampling_steps).to(device)
    if os.path.exists(args.weights):
        state_dict = torch.load(args.weights, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint loaded successfully.")
    else:
        print(f"Warning: Checkpoint not found at {args.weights}. Running with initialized weights.")

    model.eval()

    # 2. Collect image paths
    if os.path.isdir(args.input):
        image_paths = sorted(
            glob.glob(os.path.join(args.input, "*.png")) + 
            glob.glob(os.path.join(args.input, "*.jpg"))
        )
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        raise FileNotFoundError(f"Input path does not exist: {args.input}")

    print(f"Processing {len(image_paths)} images...")

    with torch.no_grad():
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            raw_img = Image.open(img_path).convert("RGB")
            
            # Pad dimensions to be divisible by 16 (downsample factor requirement)
            w, h = raw_img.size
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16

            input_tensor = TF.to_tensor(raw_img).unsqueeze(0).to(device)
            if pad_h > 0 or pad_w > 0:
                input_tensor = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")

            # Stage 1: Illumination enhancement
            stage1_out = model.stage1(input_tensor)

            # Stage 2: Perturbed consistency diffusion inference via DDIM
            x_cond = stage1_out * 2.0 - 1.0
            enhanced_output = model.diffusion.ddim_sample(x_cond)

            # Rescale output to [0, 1] range
            enhanced_output = (enhanced_output + 1.0) / 2.0
            enhanced_output = torch.clamp(enhanced_output, 0.0, 1.0)

            # Remove padding to restore original image aspect
            if pad_h > 0 or pad_w > 0:
                input_tensor = input_tensor[:, :, :h, :w]
                enhanced_output = enhanced_output[:, :, :h, :w]

            # Save restored image and a side-by-side comparison
            save_path = os.path.join(args.output, img_name)
            vutils.save_image(enhanced_output, save_path)

            comparison_path = os.path.join(args.output, f"compare_{img_name}")
            comparison_grid = torch.cat([input_tensor, enhanced_output], dim=3)
            vutils.save_image(comparison_grid, comparison_path)

            print(f"Saved: {img_name} -> {args.output}")

    print(f"Inference complete. Results stored in {args.output}")

if __name__ == "__main__":
    run_inference()
