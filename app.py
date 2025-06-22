import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision.utils import make_grid

# --- Define Model Architecture (MUST BE IDENTICAL TO TRAINING SCRIPT) ---
latent_dim = 100
n_classes = 10
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        img = self.model(x)
        return img.view(img.size(0), *img_shape)

# --- Load Trained Model ---
@st.cache_resource # Cache the model to avoid reloading on every interaction
def load_model():
    # Use CPU for deployment, as Streamlit Cloud has no GPU
    device = torch.device('cpu') 
    model = Generator()
    # Load the state dictionary. map_location ensures it loads to CPU.
    model.load_state_dict(torch.load('generator.pth', map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    return model

generator = load_model()

# --- Streamlit Web App Interface ---
st.set_page_config(layout="wide")
st.title("✒️ Handwritten Digit Generator")
st.write("Select a digit from the dropdown menu and click 'Generate' to create five new images of that digit using a Conditional GAN trained on MNIST.")

col1, col2 = st.columns([1, 3])

with col1:
    digit_to_generate = st.selectbox(
        "Select a digit (0-9):",
        options=list(range(10))
    )
    
    generate_button = st.button("Generate Images", type="primary")

with col2:
    if generate_button:
        st.subheader(f"Generated Images for Digit: {digit_to_generate}")
        
        # Number of images to generate
        num_images = 5
        
        # Prepare for generation
        device = torch.device('cpu')
        z = torch.randn(num_images, latent_dim, device=device)
        labels = torch.LongTensor([digit_to_generate] * num_images).to(device)
        
        # Generate images
        with torch.no_grad():
            generated_imgs = generator(z, labels)
        
        # Post-process for display (move from [-1, 1] to [0, 1])
        generated_imgs = (generated_imgs + 1) / 2.0

        # Display images in a grid
        grid = make_grid(generated_imgs, nrow=5, normalize=True)
        st.image(grid.permute(1, 2, 0).cpu().numpy())
    else:
        st.info("Please select a digit and click the generate button.")

