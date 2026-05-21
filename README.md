# Text-to-Image Generation using Conditional GAN

Student-level deep learning project that generates **64×64 bird images** from text descriptions using a **Conditional GAN (cGAN)** with **DistilBERT** text embeddings on the **CUB-200 Birds** dataset.

## Architecture

```
User Text → DistilBERT → 256-d text vector
Random noise (100-d) + text vector → Generator → 64×64 image
Discriminator (training): real/fake + text match
```

| Module | File | Role |
|--------|------|------|
| Text encoder | `utils/text_encoder.py` | DistilBERT + projection to 256-d |
| Generator | `models/generator.py` | Noise + text → image |
| Discriminator | `models/discriminator.py` | Image + text → real/fake |
| Training | `train.py` | GAN loop, logging, checkpoints |
| Inference | `generate.py` | CLI image generation |
| UI | `app.py` | Streamlit demo |

## Requirements

- Python 3.10+
- CPU (default) or CUDA GPU
- ~2 GB disk for CUB dataset (after download)

## Setup

```bash
cd "e:\KCT\Sem 2\GAI\GAN"
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Dataset (CUB Birds)

1. **Images** — Download [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) and extract under:
   ```
   dataset/cub/images/<class>/<image>.jpg
   ```
   (or `dataset/cub/CUB_200_2011/images/...`)

2. **Captions** — Download text from [reedscot/cvpr2016](https://github.com/reedscot/cvpr2016) and place `.txt` files in:
   ```
   dataset/cub/text/<image_id>.txt
   ```

3. **Build index**:
   ```bash
   python scripts/download_cub.py
   ```
   Creates `dataset/cub_index.json`.

4. **Cache text embeddings** (one-time, speeds up CPU training):
   ```bash
   python scripts/cache_embeddings.py
   ```
   Saves vectors to `data_cache/embeddings/` and `data_cache/text_projection.pt`.

## Training

### Quick demo checkpoint (recommended for CPU)

```bash
python scripts/quick_train_demo.py
```

- 5 epochs, 500 images, batch size 8  
- Output: `checkpoints/demo_cgan_64.pt`  
- Expected time: ~30–90 minutes on CPU  
- Sample grids: `outputs/samples/`  
- Loss log: `logs/train_loss.csv`

### Full training (optional)

```bash
python train.py --epochs 30 --device cpu
```

### Custom options

```bash
python train.py --epochs 10 --subset 1000 --batch-size 8 --save checkpoints/my_model.pt --overwrite-logs
```

## Generate an image (CLI)

```bash
python generate.py --text "A small yellow bird with black wings" --out outputs/bird.png --seed 42
```

## Streamlit demo (college presentation)

```bash
streamlit run app.py
```

1. Enter a bird description  
2. Click **Generate Image**  
3. View and **Download** the result  
4. Show the **training loss chart** (after training)

## Project structure

```
text-to-image-gan/
├── dataset/                 # CUB images + captions (gitignored)
├── data_cache/              # Precomputed embeddings (gitignored)
├── checkpoints/
│   └── demo_cgan_64.pt      # Created by training
├── models/
│   ├── generator.py
│   └── discriminator.py
├── utils/
│   ├── text_encoder.py
│   ├── dataset.py
│   ├── image_utils.py
│   └── checkpoint.py
├── scripts/
│   ├── download_cub.py
│   ├── cache_embeddings.py
│   └── quick_train_demo.py
├── train.py
├── generate.py
├── app.py
├── requirements.txt
└── README.md
```

## Evaluation (for report)

| Metric | Description |
|--------|-------------|
| **Human visual evaluation** | Compare generated images to captions manually; save grids from `outputs/samples/` |
| **Training loss curves** | Plot `logs/train_loss.csv` — discuss GAN instability |
| **Inception Score (IS)** | Measures quality and diversity of generated images |
| **FID** | Fréchet distance between real and generated feature distributions |

IS and FID require extra GPU batch evaluation; this project focuses on loss curves and visual results at 64×64.

## Research papers

1. [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) (Goodfellow et al., 2014)  
2. [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396) (Reed et al., 2016)  
3. [StackGAN](https://arxiv.org/abs/1612.03242) (Zhang et al., 2017) — future work  
4. [AttnGAN](https://arxiv.org/abs/1711.10485) (Xu et al., 2018) — future work  

## Tutorials

- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)  
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)  
- [Streamlit Documentation](https://docs.streamlit.io/)  

## Limitations

- **64×64 resolution** — images will look blurry  
- **CPU training** is slow; use the quick demo script first  
- **GAN instability** — mode collapse or noisy outputs are common  
- **Dataset required** — demo checkpoint must be trained after downloading CUB  

## Future enhancements

- StackGAN / AttnGAN for higher resolution  
- Diffusion models (Stable Diffusion)  
- FID/IS evaluation pipeline  
- Style and attribute control  

## License

Educational use. CUB dataset has its own terms from Caltech.
