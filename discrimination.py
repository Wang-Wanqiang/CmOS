import base64
import csv
import json
import mimetypes
import os
import re

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from data.Retriever import Retriever
from model.FAREEncoder import FAREEncoder


class ContentDiscrimination:
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.encoder = FAREEncoder()
        self.retriever = Retriever()
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _image_path_to_qwen_format(self, image):
        if not os.path.exists(image):
            raise FileNotFoundError
        mime_type, _ = mimetypes.guess_type(image)
        if not mime_type:
            mime_type = "image/jpeg"
        try:
            with open(image, "rb") as f:
                base64_str = base64.b64encode(f.read()).decode()
        except Exception as e:
            raise ValueError

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_str}"
            }
        }

    def build_prompt_with_exemplar(self, question, answer, image=None, exemplar_csv_path=None):
        print("Searching for the best matching exemplar question...")
        best_match, _ = self.retriever.retrieve_similar_exemplar(
            question, image, answer, exemplar_csv_path
        )
        if best_match is None or not best_match.get("question", "").strip():
            raise ValueError

        prompt_text = f"""
        We are conducting a reasoning task on whether the original question can be converted into image-based options.
        Please read the following background information of the original question and the exemplar question, then judge whether the original question is suitable to be converted into image options.
        Note that you should first reason and then make a judgment, and also specify which one.
        [Original Question Background]
        Original Question: {question}
        Original Answer: {answer}

        [Exemplar Question]
        Question: {best_match["question"]}
        Answer: {best_match["answer"]}
        Reasoning: {best_match.get("reason", "(No reasoning information)")}
        Judgment: {best_match.get("convertible")}

        Please combine the above information and images to judge whether the original question can be converted into image options and explain the reason.
        [Please strictly follow the output format]:
        {{
          "reason": string, explaining the basis for judgment, no less than 20 characters.
          "convertible": boolean, true means can be converted into image options, false means not suitable;
        }}
        """

        chat_messages = [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are a rigorous educational technology researcher analyzing the rationality of question design using images as options."
                        "Please consider the images in your judgment, but do not describe the images themselves, focus on reasoning and judgment."
                    )
                }],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}]
            }
        ]

        if image:

            chat_messages[1]["content"].append({
                "type": "text",
                "text": "The following image is from the background information of the original question:"
            })
            chat_messages[1]["content"].append(self._image_path_to_qwen_format(image))

        if "image" in best_match and best_match["image"] and isinstance(best_match["image"], str) and os.path.exists(best_match["image"]):

            chat_messages[1]["content"].append({
                "type": "text",
                "text": "The following image is the reference image of the exemplar question:"
            })
            chat_messages[1]["content"].append(self._image_path_to_qwen_format(best_match["image"]))


        return chat_messages, best_match

    def save_result_to_csv(self, id, question, image, answer, result, exemplar_info, csv_path="conversion_results.csv"):

        file_exists = os.path.exists(csv_path)
        with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "id", "question", "image", "answer", "convertible", "reason", "exemplar_question", "exemplar_answer",
                "exemplar_reason"
            ])
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "id": id,
                "question": question,
                "image": image,
                "answer": answer,
                "convertible": result["convertible"],
                "reason": result["reason"],
                "exemplar_question": exemplar_info.get("question"),
                "exemplar_answer": exemplar_info.get("answer"),
                "exemplar_reason": exemplar_info.get("reason") or exemplar_info.get("reasoning", "(missing)")
            })

    def get_llm_output(self, messages, model_name="qwen2.5-vl-32b-instruct"):
        print("Getting output from large language model...")
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        output = response.choices[0].message.content
        match = re.search(r"\{[\s\S]*\}", output)
        if not match:
            raise ValueError
        json_str = match.group(0)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError
        if "convertible" not in result or "reason" not in result:
            raise ValueError
        return result

    def evaluate_question_convertibility(self, id, question, answer, image=None, exemplar_csv_path=None):
        # if not image or pd.isna(image) or not isinstance(image, str) or not image.strip():
        #     print(f"⚠️ No image path, skipping question ID: {id}")
        #     image = None
        # elif not os.path.exists(image):
        #     raise FileNotFoundError(f"❌ Image file not found (question ID: {id}): {image}")
        messages, exemplar_info = self.build_prompt_with_exemplar(question, answer, image, exemplar_csv_path)
        print(f"Prompt construction successful (question ID: {id}), getting LLM output...")
        result = self.get_llm_output(messages)
        print(f"LLM output completed (question ID: {id})")
        return result, exemplar_info

    def generate_convertible_for_all(self, input_csv_path, output_csv_path, exemplar_csv_path=None,
                                     start_index: int = 0, end_index: int = None):
        print(
            f"📊 Reading output file {output_csv_path if os.path.exists(output_csv_path) else input_csv_path}, processing index range [{start_index}, {end_index})...")

        if os.path.exists(output_csv_path):
            df = pd.read_csv(output_csv_path)
        else:
            df = pd.read_csv(input_csv_path)

        if end_index is None:
            end_index = len(df)

        cache = []
        for i in tqdm(range(start_index, end_index), desc="Processing questions"):
            row = df.iloc[i]
            id = row.get("id")
            question = row.get("question")
            answer = row.get("answer")
            image = row.get("image")
            if pd.isna(image) or not isinstance(image, str) or not image.strip():
                image = None

            print(f"Question ID: {id}, Image path: {image}")

            try:
                result, exemplar_info = self.evaluate_question_convertibility(id, question, answer, image,
                                                                              exemplar_csv_path)
                cache.append((id, question, image, answer, result, exemplar_info))
            except Exception as e:
                print(f"❌ Failed to process (question ID: {id}): {e}")
                cache.append((id, question, image, answer, {"convertible": None, "reason": f"Error: {str(e)}"}, None))

            if len(cache) == 50:
                for id, question, image, answer, result, exemplar_info in cache:
                    self.save_result_to_csv(id, question, image, answer, result, exemplar_info, output_csv_path)
                cache = []

        if cache:
            for id, question, image, answer, result, exemplar_info in cache:
                self.save_result_to_csv(id, question, image, answer, result, exemplar_info, output_csv_path)

        print(f"✅ Processing complete (index range {start_index}-{end_index}), results saved to {output_csv_path}")


if __name__ == '__main__':
    cd = ContentDiscrimination(api_key=your_api)
    input = "your_test.csv"
    output = "your_ouput.csv"
    db_path = "your_exemplar.csv"
    start_index: int = 0
    end_index: int = 500
    cd.generate_convertible_for_all(input, output, db_path, start_index=start_index, end_index=end_index)
