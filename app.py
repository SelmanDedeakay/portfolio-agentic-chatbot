# Add this to your app.py if your model file is large (>100MB)

import streamlit as st
import torch
import requests
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Generator Network
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

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

            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        label_emb = self.label_emb(labels)
        gen_input = torch.cat([noise, label_emb], dim=1)

        # Generate image
        img = self.main(gen_input)
        img = img.view(img.size(0), 1, img_size, img_size)
        return img

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        # Label embedding
        self.label_emb = nn.Embedding(num_classes, num_classes)

        # Discriminator layers
        self.main = nn.Sequential(
            # Input: img_size*img_size + num_classes
            nn.Linear(img_size * img_size + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Flatten image and embed labels
        img_flat = img.view(img.size(0), -1)
        label_emb = self.label_emb(labels)

        # Concatenate image and label embeddings
        disc_input = torch.cat([img_flat, label_emb], dim=1)

        # Get validity score
        validity = self.main(disc_input)
        return validity

@st.cache_resource
def download_model():
    """Download model from external source if not present"""
    model_path = "generator.pth"
    
    if not os.path.exists(model_path):
        st.info("📥 Downloading model file... (This may take a moment on first load)")
        
        # Option 1: Google Drive link (replace with your file ID)
        # file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
        # url = f"https://drive.google.com/uc?id={file_id}"
        
        # Option 2: Dropbox link (replace with your link)
        # url = "https://dropbox.com/s/YOUR_FILE_LINK/generator.pth?dl=1"
        
        # Option 3: GitHub Releases (recommended for <2GB files)
        url = "https://github.com/SelmanDedeakay/MSTI-AI-Contest/releases/download/v1.0/generator.pth"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {str(e)}")
            return None
    
    return model_path

# Modify your load_model function to use this:
@st.cache_resource
def load_model():
    """Load the trained generator model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Download model if needed
    model_path = download_model()
    if model_path is None:
        return None, device
    
    # Initialize model
    generator = Generator(latent_dim=100, num_classes=10)
    
    try:
        # Load trained weights
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.to(device)
        generator.eval()
        return generator, device
    except Exception as e:
        st.error(f"⚠️ Error loading model: {str(e)}")
        return None, device