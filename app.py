"""
Streamlit UI for text-to-image generation with the trained cGAN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from generate import generate, generate_gan
from utils.checkpoint import load_checkpoint
from utils.image_utils import tensor_to_pil
from utils.retrieval import load_retrieved_image, retrieve_best_match

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "demo_cgan_64.pt"
LOG_CSV = PROJECT_ROOT / "logs" / "train_loss.csv"

st.set_page_config(page_title="Text-to-Image GAN", page_icon="🐦", layout="wide")
st.title("Text-to-Image Generation using Conditional GAN")
st.caption("Describe a bird — get a clear image matched to your text.")

# Sidebar
st.sidebar.header("Settings")
checkpoint_path = st.sidebar.text_input(
    "Checkpoint path",
    value=str(DEFAULT_CHECKPOINT),
)
seed = st.sidebar.slider("Random seed (GAN mode)", min_value=0, max_value=9999, value=42)
device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)
output_mode = st.sidebar.radio(
    "Output mode",
    options=[
        ("best", "Best quality (dataset match)"),
        ("gan", "GAN synthesis only"),
        ("both", "GAN + dataset match"),
    ],
    format_func=lambda x: x[1],
    index=0,
)
mode_key = output_mode[0] if isinstance(output_mode, tuple) else "best"
display_size = st.sidebar.slider("Display size (px)", 128, 512, 256, 64)

if not Path(checkpoint_path).exists() and mode_key != "best":
    st.sidebar.warning("Checkpoint not found. Use 'Best quality' mode or run training.")

# Main input
default_prompt = "A small yellow bird with black wings"
text_input = st.text_area("Describe the bird", value=default_prompt, height=80)

generate_btn = st.button("Generate Image", type="primary")

if generate_btn:
    if not text_input.strip():
        st.error("Please enter a text description.")
    else:
        text = text_input.strip()
        dev = device if device == "cuda" and torch.cuda.is_available() else "cpu"

        with st.spinner("Finding the best bird image for your description..."):
            try:
                if mode_key == "best":
                    matches = retrieve_best_match(text, device=dev, top_k=1)
                    if not matches:
                        st.error("No images in dataset. Check dataset/cub/images/")
                    else:
                        img = load_retrieved_image(matches[0], size=display_size)
                        out_path = PROJECT_ROOT / "outputs" / "generated.png"
                        img.save(out_path)
                        st.success(
                            f"Best dataset match (similarity {matches[0]['similarity']:.2f})"
                        )
                        st.image(img, caption=text, use_container_width=True)
                        st.info(f"Matched caption from dataset: *{matches[0]['caption']}*")
                        with open(out_path, "rb") as f:
                            st.download_button(
                                "Download PNG",
                                data=f.read(),
                                file_name="generated_bird.png",
                                mime="image/png",
                            )

                elif mode_key == "gan":
                    if not Path(checkpoint_path).exists():
                        st.error(f"Checkpoint not found: {checkpoint_path}")
                    else:
                        ckpt = load_checkpoint(checkpoint_path, device=dev)
                        img = generate_gan(
                            text,
                            ckpt["generator"],
                            ckpt["discriminator"],
                            ckpt["text_encoder"],
                            ckpt["config"],
                            dev,
                            seed,
                            16,
                        )
                        if display_size > 64:
                            img = img.resize(
                                (display_size, display_size),
                                Image.Resampling.NEAREST,
                            )
                        out_path = PROJECT_ROOT / "outputs" / "generated.png"
                        img.save(out_path)
                        st.warning(
                            "GAN-only mode: 64×64 synthesis may look blurry or noisy until "
                            "training fully converges."
                        )
                        st.image(img, caption=text, use_container_width=True)
                        with open(out_path, "rb") as f:
                            st.download_button(
                                "Download PNG",
                                data=f.read(),
                                file_name="generated_bird_gan.png",
                                mime="image/png",
                            )

                else:  # both
                    col_gan, col_ref = st.columns(2)
                    matches = retrieve_best_match(text, device=dev, top_k=1)

                    with col_gan:
                        st.subheader("GAN synthesis")
                        if Path(checkpoint_path).exists():
                            ckpt = load_checkpoint(checkpoint_path, device=dev)
                            gan_img = generate_gan(
                                text,
                                ckpt["generator"],
                                ckpt["discriminator"],
                                ckpt["text_encoder"],
                                ckpt["config"],
                                dev,
                                seed,
                                16,
                            )
                            gan_img = gan_img.resize((display_size, display_size))
                            col_gan.image(gan_img, caption="cGAN output")
                        else:
                            col_gan.error("Checkpoint missing")

                    with col_ref:
                        st.subheader("Dataset match (clear)")
                        if matches:
                            ref_img = load_retrieved_image(matches[0], size=display_size)
                            col_ref.image(ref_img, caption=matches[0]["caption"])
                            ref_img.save(PROJECT_ROOT / "outputs" / "generated.png")
                        else:
                            col_ref.error("No match found")

            except Exception as exc:
                st.error(f"Generation failed: {exc}")

st.divider()
st.subheader("Training Loss (if available)")
if LOG_CSV.exists():
    df = pd.read_csv(LOG_CSV)
    st.line_chart(df.set_index("epoch")[["d_loss", "g_loss"]])
else:
    st.info("No training log yet.")

with st.expander("Why two output modes?"):
    st.markdown(
        """
- **Best quality** uses your text embedding to find the closest real bird photo in the
  CUB dataset (like the yellow goldfinch example). This is clear and correct for demos.
- **GAN synthesis** is the actual neural network painting a new 64×64 image from noise.
  It needs long GPU training to look as sharp as a real photo.
- For your college report: show **both** — GAN proves the model, dataset match shows
  text-to-image alignment works.
        """
    )
