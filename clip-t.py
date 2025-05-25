import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import torch

from model.FAREEncoder import FAREEncoder


def load_image_safely(image_path):
    image_path = image_path.replace('\\', '/')
    if not os.path.isfile(image_path):
        print(f"Warning: Image not found - {image_path}")
        return None
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Failed to load image - {image_path}, error: {e}")
        return None


def calculate_clip_t_similarity(encoder, text, image):
    text_feat = encoder.encode_text(text)
    image_feat = encoder.encode_image(image)

    if isinstance(text_feat, torch.Tensor):
        text_feat = text_feat.detach().cpu()
    if isinstance(image_feat, torch.Tensor):
        image_feat = image_feat.detach().cpu()

    sim = torch.nn.functional.cosine_similarity(text_feat.unsqueeze(0), image_feat.unsqueeze(0)).item()
    return sim


def process_clip_t_similarity(csv_path, output_path):
    encoder = FAREEncoder()

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Description', 'image_path'])
    df['image_path'] = df['image_path'].astype(str).apply(lambda x: x.strip())

    similarities = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        desc = row['Description']
        image_path = row['image_path']
        image = load_image_safely(image_path)
        if image is None:
            sim_val = None
        else:
            try:
                sim_val = calculate_clip_t_similarity(encoder, desc, image)
            except Exception as e:
                print(f"Failed to compute similarity - Row {idx}, Error: {e}")
                sim_val = None

        similarities.append(sim_val)

    df['CLIP-T Score'] = similarities
    df.to_csv(output_path, index=False)
    print(f"CLIP-T similarity computed for {len(df)} rows and saved to {output_path}")
