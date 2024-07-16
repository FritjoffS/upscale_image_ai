import os
import requests
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import torch
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox

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

def process_images(input_dir, output_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Download weights if not present
    weights_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    weights_path = "RealESRGAN_x4plus.pth"
    download_weights(weights_url, weights_path)

    # Initialize the RRDBNet model
    print('Initializing RRDBNet model...')
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

    # Ensure output directory exists
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

    messagebox.showinfo("Info", "Image processing complete")

def select_input_dir():
    input_dir = filedialog.askdirectory()
    if input_dir:
        input_dir_var.set(input_dir)

def start_processing():
    input_dir = input_dir_var.get()
    output_dir = output_dir_var.get()
    if not input_dir:
        messagebox.showwarning("Warning", "Please select an input directory")
        return
    if not output_dir:
        messagebox.showwarning("Warning", "Please select an output directory")
        return
    process_images(input_dir, output_dir)

def select_output_dir():
    output_dir = filedialog.askdirectory()
    if output_dir:
        output_dir_var.set(output_dir)

# Initialize the GUI application
app = tk.Tk()
app.title("Image Super-Resolution with RealESRGAN")

input_dir_var = tk.StringVar()
output_dir_var = tk.StringVar()

tk.Label(app, text="Input Directory:").grid(row=0, column=0, padx=10, pady=10)
tk.Entry(app, textvariable=input_dir_var, width=50).grid(row=0, column=1, padx=10, pady=10)
tk.Button(app, text="Browse", command=select_input_dir).grid(row=0, column=2, padx=10, pady=10)

tk.Label(app, text="Output Directory:").grid(row=1, column=0, padx=10, pady=10)
tk.Entry(app, textvariable=output_dir_var, width=50).grid(row=1, column=1, padx=10, pady=10)
tk.Button(app, text="Browse", command=select_output_dir).grid(row=1, column=2, padx=10, pady=10)

tk.Button(app, text="Start Processing", command=start_processing).grid(row=2, column=0, columnspan=3, pady=20)

app.mainloop()
