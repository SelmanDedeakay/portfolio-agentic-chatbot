# app.py - Gradio app for MNIST Digit Generator

import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# Define the same Generator architecture as in training
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_size = 28

        # Label embedding
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Generator layers
        self.main = nn.Sequential(
            # Input: latent_dim + num_classes
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, self.img_size * self.img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        label_emb = self.label_emb(labels)
        gen_input = torch.cat([noise, label_emb], dim=1)

        # Generate image
        img = self.main(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

# Load the model
device = torch.device('cpu')
generator = Generator(latent_dim=100, num_classes=10).to(device)

try:
    generator.load_state_dict(torch.load('generator.pth', map_location=device))
    generator.eval()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print("Warning: generator.pth not found!")

# Generate function for single digit
def generate_single_digit(digit, seed):
    if not model_loaded:
        return None
    
    # Set seed if provided
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        noise = torch.randn(1, 100).to(device)
        label = torch.tensor([digit], dtype=torch.long).to(device)
        generated_img = generator(noise, label)
        
        # Denormalize from [-1, 1] to [0, 1]
        generated_img = generated_img * 0.5 + 0.5
        generated_img = torch.clamp(generated_img, 0, 1)
        
        # Convert to numpy and then PIL
        img_numpy = generated_img.cpu().numpy()[0, 0]
        img_pil = Image.fromarray((img_numpy * 255).astype(np.uint8), mode='L')
        
        # Resize for better display
        img_pil = img_pil.resize((256, 256), Image.Resampling.NEAREST)
        
        return img_pil

# Generate function for multiple digits
def generate_multiple_digits(digit, num_samples, seed):
    if not model_loaded:
        return None
    
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        noise = torch.randn(num_samples, 100).to(device)
        labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
        generated_imgs = generator(noise, labels)
        
        # Denormalize
        generated_imgs = generated_imgs * 0.5 + 0.5
        generated_imgs = torch.clamp(generated_imgs, 0, 1)
        
        # Create grid
        cols = min(5, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]
        
        for idx in range(num_samples):
            row = idx // cols
            col = idx % cols
            axes[row][col].imshow(generated_imgs[idx].cpu().numpy()[0], cmap='gray')
            axes[row][col].axis('off')
            axes[row][col].set_title(f'Sample {idx + 1}')
        
        # Remove empty subplots
        for idx in range(num_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            fig.delaxes(axes[row][col])
        
        plt.tight_layout()
        
        # Convert to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        img = Image.open(buf)
        
        return img

# Generate all digits
def generate_all_digits(seed):
    if not model_loaded:
        return None
    
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    
    with torch.no_grad():
        for digit in range(10):
            noise = torch.randn(1, 100).to(device)
            label = torch.tensor([digit], dtype=torch.long).to(device)
            generated_img = generator(noise, label)
            
            # Denormalize
            generated_img = generated_img * 0.5 + 0.5
            generated_img = torch.clamp(generated_img, 0, 1)
            
            row = digit // 5
            col = digit % 5
            axes[row, col].imshow(generated_img.cpu().numpy()[0, 0], cmap='gray')
            axes[row, col].set_title(f'Digit {digit}')
            axes[row, col].axis('off')
    
    plt.suptitle('Generated Digits 0-9', fontsize=16)
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img = Image.open(buf)
    
    return img

# Create Gradio interface
with gr.Blocks(title="MNIST Digit Generator") as demo:
    gr.Markdown("# 🔢 MNIST Digit Generator")
    gr.Markdown("Generate handwritten digits using a trained GAN model")
    
    with gr.Tab("Single Digit"):
        with gr.Row():
            with gr.Column():
                single_digit = gr.Slider(
                    minimum=0, 
                    maximum=9, 
                    step=1, 
                    value=0, 
                    label="Select Digit"
                )
                single_seed = gr.Slider(
                    minimum=-1, 
                    maximum=9999, 
                    step=1, 
                    value=-1, 
                    label="Seed (-1 for random)"
                )
                single_generate_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column():
                single_output = gr.Image(
                    label="Generated Digit",
                    type="pil"
                )
        
        single_generate_btn.click(
            fn=generate_single_digit,
            inputs=[single_digit, single_seed],
            outputs=single_output
        )
    
    with gr.Tab("Multiple Samples"):
        with gr.Row():
            with gr.Column():
                multi_digit = gr.Slider(
                    minimum=0, 
                    maximum=9, 
                    step=1, 
                    value=0, 
                    label="Select Digit"
                )
                num_samples = gr.Slider(
                    minimum=1, 
                    maximum=25, 
                    step=1, 
                    value=5, 
                    label="Number of Samples"
                )
                multi_seed = gr.Slider(
                    minimum=-1, 
                    maximum=9999, 
                    step=1, 
                    value=-1, 
                    label="Seed (-1 for random)"
                )
                multi_generate_btn = gr.Button("Generate Multiple", variant="primary")
            
            with gr.Column():
                multi_output = gr.Image(
                    label="Generated Samples",
                    type="pil"
                )
        
        multi_generate_btn.click(
            fn=generate_multiple_digits,
            inputs=[multi_digit, num_samples, multi_seed],
            outputs=multi_output
        )
    
    with gr.Tab("All Digits"):
        with gr.Row():
            with gr.Column():
                all_seed = gr.Slider(
                    minimum=-1, 
                    maximum=9999, 
                    step=1, 
                    value=-1, 
                    label="Seed (-1 for random)"
                )
                all_generate_btn = gr.Button("Generate All Digits (0-9)", variant="primary")
            
            with gr.Column():
                all_output = gr.Image(
                    label="All Generated Digits",
                    type="pil"
                )
        
        all_generate_btn.click(
            fn=generate_all_digits,
            inputs=[all_seed],
            outputs=all_output
        )
    
    with gr.Tab("About"):
        gr.Markdown("""
        ## About this Model
        
        This app uses a **Generative Adversarial Network (GAN)** trained on the MNIST dataset to generate 
        handwritten digit images.
        
        ### How it works:
        - The model takes random noise and a digit label as input
        - It generates a 28x28 pixel grayscale image of the specified digit
        - Each generation with the same seed will produce the same image
        
        ### Model Architecture:
        - **Generator**: 4-layer neural network with LeakyReLU activation
        - **Training**: 100 epochs on MNIST dataset
        - **Approach**: Conditional GAN for digit-specific generation
        
        ### Features:
        - Generate any digit from 0-9
        - Control randomness with seed values
        - Generate multiple samples at once
        - View all digits in a single grid
        """)
    
    # Examples
    gr.Examples(
        examples=[
            [7, 42],
            [3, 123],
            [9, 999],
            [0, 2024],
        ],
        inputs=[single_digit, single_seed],
        outputs=single_output,
        fn=generate_single_digit,
        cache_examples=True,
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()