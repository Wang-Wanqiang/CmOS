import os
import json
import ast
import requests
import pandas as pd
import numpy as np
from pathlib import PurePosixPath
from urllib.parse import urlparse, unquote
from http import HTTPStatus
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from dashscope import ImageSynthesis
from tqdm import tqdm

from model.FAREEncoder import FAREEncoder


class ImageGeneration:
    def __init__(self, image_dataset_path: str, api_key: str,
                 model_name: str = "qwen2.5-vl-instruct",
                 save_dir: str = "./enhanced_images"):
        self.image_dataset = pd.read_csv(image_dataset_path)
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key,
                             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.encoder = FAREEncoder()
        self.model_name = model_name
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _parse_embedding(self, embedding_str: str) -> np.ndarray:
        if isinstance(embedding_str, str):
            return np.array(ast.literal_eval(embedding_str))
        return np.array(embedding_str)

    def calculate_similarity(self, text: str, embedding_str: str) -> float:
        text_embedding = self.encoder.encode_text(text).cpu().numpy()
        image_embedding = self._parse_embedding(embedding_str)
        return cosine_similarity([text_embedding], [image_embedding])[0][0]

    def retrieve_most_similar_image(self, option: str) -> dict:
        similarities = [
            self.calculate_similarity(option, row['caption_embedding'])
            for _, row in self.image_dataset.iterrows()
        ]
        idx = int(np.argmax(similarities))
        row = self.image_dataset.iloc[idx]
        return {"path": row['path'], "caption": row['caption']}

    def generate_prompt(self, option_description: str, image_info: dict) -> str:
        return f"""You are an expert in matching images with descriptions. Please analyze the matching and similarity between the option description and the image, then output your judgment in JSON format.

Option description:
"{option_description}"

Image information:
Path: {image_info["path"]}
Caption: "{image_info["caption"]}"

Please output a structured JSON in the format:
{{
  "match": true/false,  // true if similarity > 0.85
  "similarity": float,  // similarity score
  "reason": "",         // describe similarities and differences
  "suggestion": ""      // prompt for image generation model including what the image contains
}}

Strictly output only the above JSON without any extra text.
"""

    def _clean_json_block(self, text: str) -> str:
        import re
        code_block = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        return code_block.group(1).strip() if code_block else text.strip()

    def analyze_with_model(self, option_description: str, image_info: dict) -> dict:
        prompt = self.generate_prompt(option_description, image_info)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_content = response.choices[0].message.content.strip()
        cleaned_text = self._clean_json_block(raw_content)
        try:
            return json.loads(cleaned_text)
        except Exception as e:
            print("[Error] Failed to parse model output:", e)
            print("[Raw response]", raw_content)
            print("[Cleaned text]", cleaned_text)
            raise

    def edit_image_with_prompt(self, base_image_path: str, prompt: str) -> str:
        print(f"[Image Enhancement] Generating image with prompt: {prompt} based on image: {base_image_path}")
        try:
            rsp = ImageSynthesis.call(
                api_key=self.api_key,
                model="wanx2.1-t2i-turbo",
                prompt=prompt,
                n=1,
                size='1024*1024'
            )
        except Exception as e:
            print("[Error] DashScope API call failed:", e)
            raise RuntimeError("Image generation failed")

        if rsp.status_code == HTTPStatus.OK and rsp.output and rsp.output.results:
            result_url = rsp.output.results[0].url
            file_name = PurePosixPath(unquote(urlparse(result_url).path)).parts[-1]
            new_path = os.path.join(self.save_dir, file_name)

            try:
                with open(new_path, 'wb') as f:
                    f.write(requests.get(result_url).content)
                print(f"[Success] Image saved to: {new_path}")
                return new_path
            except Exception as e:
                print("[Error] Failed to download image:", e)
                raise RuntimeError("Image download failed")
        else:
            print(f"[Error] Image generation failed: {rsp.status_code} | {rsp.code}: {rsp.message}")
            raise RuntimeError("Image generation failed with no valid result")

    def optimize_image_by_rounds(self, option_description: str, max_rounds: int = 3) -> dict:
        if self.image_dataset.empty:
            print("[Info] Image dataset empty, generating image from scratch")
            return self.optimize_image_from_scratch(option_description, max_rounds)

        # Retrieve the most similar image from dataset
        image_info = self.retrieve_most_similar_image(option_description)

        for i in range(max_rounds):
            print(f"\n-- Optimization Round {i + 1} --")
            print(f"[Current image] {image_info['path']}")
            print(f"[Current caption] {image_info['caption']}")

            result = self.analyze_with_model(option_description, image_info)

            if result["match"] and result["similarity"] >= 0.85:
                print("[Analysis] Good match found, no further optimization needed")
                return {
                    "final_image": image_info["path"],
                    "match": result["match"],
                    "similarity": result["similarity"],
                    "suggestion": result["suggestion"]
                }

            print(f"[Suggestion] {result['suggestion']}")
            new_image_path = self.edit_image_with_prompt(image_info["path"], result["suggestion"])
            image_info = {
                "path": new_image_path,
                "caption": result["suggestion"]
            }

        print(f"\n[Terminated] Max optimization rounds reached, final similarity: {result['similarity']}")
        final_result = self.analyze_with_model(option_description, image_info)
        return {
            "final_image": image_info["path"],
            "match": final_result["match"],
            "similarity": final_result["similarity"],
            "suggestion": final_result["suggestion"]
        }

    def optimize_image_from_scratch(self, option_description: str, max_rounds: int = 3) -> dict:
        """
        Generate an initial image directly from the option description and optimize via MLLM feedback.
        """
        print(f"\n[Initial generation] Creating initial image from description: {option_description}")
        image_path = self.edit_image_with_prompt(base_image_path="", prompt=option_description)
        image_info = {
            "path": image_path,
            "caption": option_description
        }

        for i in range(max_rounds):
            print(f"\n-- Optimization Round {i + 1} --")
            print(f"[Current image] {image_info['path']}")
            print(f"[Current caption] {image_info['caption']}")

            result = self.analyze_with_model(option_description, image_info)

            if result["match"] and result["similarity"] >= 0.85:
                print("[Analysis] Good match found, no further optimization needed")
                return {
                    "final_image": image_info["path"],
                    "match": result["match"],
                    "similarity": result["similarity"],
                    "suggestion": result["suggestion"]
                }

            print(f"[Suggestion] {result['suggestion']}")
            new_image_path = self.edit_image_with_prompt(image_info["path"], result["suggestion"])
            image_info = {
                "path": new_image_path,
                "caption": result["suggestion"]
            }

        print(f"\n[Terminated] Max optimization rounds reached, final similarity: {result['similarity']}")
        final_result = self.analyze_with_model(option_description, image_info)
        return {
            "final_image": image_info["path"],
            "match": final_result["match"],
            "similarity": final_result["similarity"],
            "suggestion": final_result["suggestion"]
        }

    def generate_images_for_all(self, option_csv: str, output_csv: str, start_idx: int = 0, end_idx: int = None):
        # Load existing output if any
        if os.path.exists(output_csv):
            df = pd.read_csv(output_csv)
            print(f"[Info] Loaded existing output file: {output_csv}")
        else:
            df = pd.read_csv(option_csv)
            df["option_image_path"] = ""  # Initialize new column
            print(f"[Info] Loaded input file: {option_csv}")

        total_options = len(df)
        end_idx = min(end_idx, total_options) if end_idx is not None else total_options

        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]

            try:
                option_desc = row["Description"]
            except KeyError as e:
                print(f"[Error] Row {idx + 1} missing 'Description' column: {e}")
                continue

            if isinstance(row.get("option_image_path", ""), str) and row["option_image_path"].strip():
                print(f"\n[Skip] Row {idx + 1} already has image path, skipping.")
                continue

            print(f"\n==== Processing row {idx + 1}/{total_options} [{row.get('Generated Option ID', '')}] ====")
            try:
                # Use retrieval + MLLM optimization if dataset is not empty
                if self.image_dataset.empty:
                    result = self.optimize_image_from_scratch(option_desc)
                else:
                    result = self.optimize_image_by_rounds(option_desc)
                df.at[idx, "option_image_path"] = result["final_image"]
                print(f"[Success] Saved image path: {result['final_image']}")
            except Exception as e:
                print(f"[Error] Failed to generate image for row {idx + 1}: {e}")
                df.at[idx, "option_image_path"] = ""

            # Save periodically
            if (idx + 1) % 10 == 0 or idx + 1 == end_idx:
                df.to_csv(output_csv, index=False)
                print(f"[Info] Progress saved at row {idx + 1}")

        print(f"\n[Completed] Image generation for rows {start_idx} to {end_idx} done.")

