# app.py - Streamlit app for MNIST Digit Generator

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

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

# Initialize session state for random seed
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42

# Load the model
@st.cache_resource
def load_generator():
    device = torch.device('cpu')  # Use CPU for deployment
    generator = Generator(latent_dim=100, num_classes=10).to(device)
    
    # Load the trained weights
    try:
        generator.load_state_dict(torch.load('generator.pth', map_location=device))
        generator.eval()
        return generator, device
    except FileNotFoundError:
        st.error("⚠️ Model file 'generator.pth' not found! Please ensure it's in the same directory as this app.")
        return None, device

# Generate digit images
def generate_digit_images(generator, device, digit, num_images=5, seed=None):
    """Generate multiple images of a specific digit"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    with torch.no_grad():
        noise = torch.randn(num_images, 100).to(device)
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        fake_imgs = generator(noise, labels)
        
        # Denormalize images from [-1, 1] to [0, 1]
        fake_imgs = fake_imgs * 0.5 + 0.5
        fake_imgs = torch.clamp(fake_imgs, 0, 1)
        
        return fake_imgs.cpu().numpy()

# Convert numpy array to PIL Image
def numpy_to_pil(img_array):
    """Convert numpy array to PIL Image"""
    img = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img.squeeze(), mode='L')

# Streamlit UI
st.set_page_config(page_title="MNIST Digit Generator", page_icon="🔢", layout="wide")

st.title("🔢 MNIST Digit Generator")
st.markdown("Generate handwritten digits using a trained GAN model")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Digit selection
    selected_digit = st.selectbox(
        "Select digit to generate:",
        options=list(range(10)),
        index=0
    )
    
    # Number of images
    num_images = st.slider(
        "Number of images to generate:",
        min_value=1,
        max_value=25,
        value=5
    )
    
    # Random seed
    use_random_seed = st.checkbox("Use random seed", value=False)
    if use_random_seed:
        st.session_state.random_seed = None
    else:
        st.session_state.random_seed = st.number_input(
            "Seed value:",
            min_value=0,
            max_value=99999,
            value=42
        )
    
    # Generate button
    generate_button = st.button("🎨 Generate Digits", type="primary", use_container_width=True)
    
    # Batch generation
    st.markdown("---")
    st.header("📦 Batch Generation")
    generate_all = st.button("Generate All Digits (0-9)", use_container_width=True)

# Load model
generator, device = load_generator()

# Main content area
if generator is not None:
    if generate_button:
        st.subheader(f"Generated Images of Digit: {selected_digit}")
        
        # Generate images
        with st.spinner('Generating images...'):
            images = generate_digit_images(
                generator, 
                device, 
                selected_digit, 
                num_images,
                seed=st.session_state.random_seed
            )
        
        # Display images in a grid
        cols_per_row = min(5, num_images)
        rows = (num_images + cols_per_row - 1) // cols_per_row
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                if img_idx < num_images:
                    with cols[col_idx]:
                        img = numpy_to_pil(images[img_idx])
                        # Resize for better display
                        img_resized = img.resize((150, 150), Image.Resampling.NEAREST)
                        st.image(img_resized, caption=f"Sample {img_idx + 1}", use_column_width=True)
        
        # Download button for generated images
        if num_images == 1:
            # Single image download
            img = numpy_to_pil(images[0])
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            btn = st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name=f"digit_{selected_digit}.png",
                mime="image/png"
            )
        else:
            # Multiple images - create a grid image
            st.markdown("---")
            st.subheader("Download Options")
            
            # Create a combined image
            fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 2, rows * 2))
            if rows == 1:
                axes = axes.reshape(1, -1)
            elif cols_per_row == 1:
                axes = axes.reshape(-1, 1)
            
            for idx in range(num_images):
                row = idx // cols_per_row
                col = idx % cols_per_row
                axes[row, col].imshow(images[idx].squeeze(), cmap='gray')
                axes[row, col].axis('off')
            
            # Remove empty subplots
            for idx in range(num_images, rows * cols_per_row):
                row = idx // cols_per_row
                col = idx % cols_per_row
                fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            st.download_button(
                label="Download All Images as Grid",
                data=buf.getvalue(),
                file_name=f"digit_{selected_digit}_grid.png",
                mime="image/png"
            )
    
    if generate_all:
        st.subheader("Generated Samples for All Digits (0-9)")
        
        with st.spinner('Generating all digits...'):
            fig, axes = plt.subplots(2, 5, figsize=(10, 4))
            
            for digit in range(10):
                images = generate_digit_images(
                    generator, 
                    device, 
                    digit, 
                    num_images=1,
                    seed=st.session_state.random_seed
                )
                
                row = digit // 5
                col = digit % 5
                axes[row, col].imshow(images[0].squeeze(), cmap='gray')
                axes[row, col].set_title(f'Digit {digit}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Download button
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close()
            
            st.download_button(
                label="Download All Digits",
                data=buf.getvalue(),
                file_name="all_digits_0-9.png",
                mime="image/png"
            )
    
    # Info section
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        This app uses a **Generative Adversarial Network (GAN)** trained on the MNIST dataset to generate 
        handwritten digit images.
        
        **How it works:**
        - The model takes random noise and a digit label as input
        - It generates a 28x28 pixel grayscale image of the specified digit
        - Each generation with the same seed will produce the same image
        
        **Features:**
        - Generate any digit from 0-9
        - Control the number of generated samples
        - Set a random seed for reproducible results
        - Download generated images
        - Batch generate all digits at once
        
        **Model Architecture:**
        - Generator: 4-layer neural network with LeakyReLU activation
        - Trained for 100 epochs on MNIST dataset
        - Uses conditional GAN approach for digit-specific generation
        """)

else:
    st.error("Unable to load the model. Please check if 'generator.pth' is in the correct location.")