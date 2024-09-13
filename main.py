# main.py
import streamlit as st
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from pathlib import Path
from tqdm.auto import tqdm

st.set_page_config(page_title="StreamBooth", layout="wide")

st.title("StreamBooth: DreamBooth Fine-Tuner for Stable Diffusion")

st.sidebar.header("Configuration")

# Sidebar inputs
dataset_dir = st.sidebar.text_input("Dataset Directory", value="/app/data")
model_dir = st.sidebar.text_input(
    "Pre-trained Model Directory", value="/app/models")
output_dir = st.sidebar.text_input("Output Directory", value="/app/output")
training_prompt = st.sidebar.text_input(
    "Training Prompt", value="a photo of [YOUR_TOKEN]")
num_train_epochs = st.sidebar.number_input(
    "Number of Training Epochs", min_value=1, max_value=100, value=1)
use_image_captioning = st.sidebar.checkbox(
    "Enable Image Captioning", value=False)
use_encoder_training = st.sidebar.checkbox(
    "Enable Encoder Training", value=False)

# Function to load images


def load_images(data_dir):
    image_extensions = [".png", ".jpg", ".jpeg"]
    images = []
    for file in os.listdir(data_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            img_path = os.path.join(data_dir, file)
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                st.warning(f"Could not open image {img_path}: {e}")
    return images

# Custom dataset class


class DreamBoothDataset(Dataset):
    def __init__(self, images, tokenizer, prompt):
        self.images = images
        self.tokenizer = tokenizer
        self.prompt = prompt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        encoding = self.tokenizer(self.prompt, return_tensors="pt")
        return {"input_ids": encoding.input_ids.squeeze(), "pixel_values": image}

# Training function


def train_dreambooth(dataset_dir, model_dir, output_dir, prompt, num_epochs, image_captioning=False, encoder_training=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.write(f"Using device: {device}")

    # Load the pre-trained Stable Diffusion model
    with st.spinner("Loading pre-trained model..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir, torch_dtype=torch.float16)
        pipe = pipe.to(device)

    # Load tokenizer and text model
    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    text_encoder = CLIPTextModel.from_pretrained(model_dir)

    # Load images
    images = load_images(dataset_dir)
    if not images:
        st.error("No images found in the dataset directory.")
        return

    st.write(f"Loaded {len(images)} images for training.")

    # Create dataset
    dataset = DreamBoothDataset(images, tokenizer, prompt)

    # Simple training loop
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=5e-6)

    for epoch in range(num_epochs):
        st.write(f"Starting epoch {epoch+1}/{num_epochs}")
        progress_bar = st.progress(0)
        for i, batch in enumerate(tqdm(dataset, desc=f"Epoch {epoch+1}")):
            pixel_values = batch["pixel_values"].unsqueeze(0).to(device)
            input_ids = batch["input_ids"].unsqueeze(0).to(device)

            # Forward pass
            outputs = pipe.unet(pixel_values=pixel_values, encoder_hidden_states=text_encoder(
                input_ids).last_hidden_state)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Update progress bar
            progress = (i + 1) / len(dataset)
            progress_bar.progress(progress)

        st.write(f"Epoch {epoch+1} completed.")

    # Save the fine-tuned model
    os.makedirs(output_dir, exist_ok=True)
    with st.spinner("Saving the fine-tuned model..."):
        pipe.save_pretrained(output_dir)
    st.success(f"Training completed. Model saved to `{output_dir}`.")

# Image generation function


def generate_images(model_path, prompt, num_images, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    st.write(f"Using device: {device}")

    # Load the fine-tuned model
    with st.spinner("Loading the fine-tuned model..."):
        pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16)
        pipe = pipe.to(device)

    # Generate images
    os.makedirs(output_dir, exist_ok=True)
    generated_images = []
    for i in range(num_images):
        with torch.autocast(device):
            image = pipe(prompt).images[0]
        image_path = os.path.join(output_dir, f"generated_{i+1}.png")
        image.save(image_path)
        generated_images.append(image)
        st.image(
            image, caption=f"Generated Image {i+1}", use_column_width=True)

    st.success(f"Generated {num_images} image(s) in `{output_dir}`.")


# Main application
if st.sidebar.button("Start Fine-Tuning"):
    if not all([os.path.isdir(dataset_dir), os.path.isdir(model_dir)]):
        st.error("Please ensure that the dataset and model directories exist.")
    else:
        with st.spinner("Training in progress..."):
            train_dreambooth(
                dataset_dir=dataset_dir,
                model_dir=model_dir,
                output_dir=output_dir,
                prompt=training_prompt,
                num_epochs=num_train_epochs,
                image_captioning=use_image_captioning,
                encoder_training=use_encoder_training
            )

st.header("Generate Images")

gen_prompt = st.text_input(
    "Generate Image with Prompt", value="a photo of [YOUR_TOKEN] in a sunny park")
gen_num_images = st.number_input(
    "Number of Images to Generate", min_value=1, max_value=10, value=1)

if st.button("Generate"):
    if not os.path.isdir(output_dir):
        st.error(
            "Please ensure the output directory exists and contains the fine-tuned model.")
    else:
        with st.spinner("Generating images..."):
            generate_images(
                model_path=output_dir,
                prompt=gen_prompt,
                num_images=gen_num_images,
                output_dir=os.path.join(output_dir, "generated")
            )
