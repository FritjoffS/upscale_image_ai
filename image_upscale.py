import os
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
import numpy as np
from PIL import Image

def download_weights(url, save_path):
    if not os.path.exists(save_path):
        print(f"Downloading model weights to {save_path}...")
        response = requests.get(url)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print("Download completed.")
    else:
        print("Model weights already exist.")

def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def save_image(image, path):
    try:
        image.save(path)
    except Exception as e:
        print(f"Error saving image: {e}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download weights if not present
    weights_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    weights_path = "RealESRGAN_x4plus.pth"
    download_weights(weights_url, weights_path)

    # Initialize the RRDBNet model
    print('Initializing RDBNet model...')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    
    # Initialize the RealESRGANer
    print('Initializing RealESRGANer...')
    upsampler = RealESRGANer(
        scale=4,
        model_path=weights_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )

    # Ensure input and output directories exist
    input_dir = "input"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Process each image in the input directory
    for filename in os.listdir(input_dir):
        input_image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_dir, filename)

        if os.path.isfile(input_image_path):
            print(f'Processing {input_image_path}...')
            low_res_image = load_image(input_image_path)
            if low_res_image is None:
                continue

            # Upscale the image
            print('Upscaling image...' + filename)
            low_res_np = np.array(low_res_image)
            high_res_np, _ = upsampler.enhance(low_res_np)

            # Convert numpy array to PIL Image
            print('Converting numpy array to PIL Image...')
            high_res_image = Image.fromarray(high_res_np.astype('uint8'))

            # Save the high-resolution image
            print(f'Saving high-resolution image to {output_image_path}...')
            save_image(high_res_image, output_image_path)
            print(f"High-resolution image saved to {output_image_path}")

if __name__ == "__main__":
    main()
