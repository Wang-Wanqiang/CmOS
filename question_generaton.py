import os
import json
import base64
import pandas as pd
import re
from openai import OpenAI

PROMPT_TEMPLATE = """
You are an expert in creating multimodal educational questions.

Based on the core concept of the original question and its answer, generate a question from different perspectives that are all visual multiple-choice questions. Each question must be based on visual options, and the correct answer can be an image representing the original answer.
Original Question: {question}
Original Answer: {answer}
Below are two examples for your reference:
[Example 1]
Original Question: Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?
Pay attention： The quality of the original question is high, so please give priority to the original question regarding the problem of visual options. If modification of the original question is needed, prefer the format: 'Which of the following pictures shows ...'
Original Answer: sample B
Original Image: A diagram showing Sample A and Sample B, each as a bottle with particles. Sample A: Mass of each particle: 30 u; Average speed: 1,100 m/s. Sample B: Mass of each particle: 46 u; Average speed: 1,100 m/s.
Reasoning:
(1) Core concept: The question involves average kinetic energy and temperature comparison—abstract concepts.
(2) Options analysis: The image displays mass and speed intuitively, which helps visualize the energy comparison.
(3) Strategy: Since the visual options can clearly convey the concept, the original question is reused as the new visual question.
New Question: Compare the average kinetic energies of the particles in each sample. Which sample has the higher temperature?
New Answer: An image of Sample B
Reasoning: Temperature relates to kinetic energy. Sample B particles have more mass but the same speed, so greater kinetic energy → higher temperature.

[Example 2]
Original Question: Which country is highlighted?
Original Answer: Jamaica
Original Image: A map with Jamaica highlighted.
Reasoning:
(1) Core concept: "Highlighted country" can be directly shown in a map.
(2) Options analysis: Using maps with different countries highlighted makes for clear visual options.
(3) Strategy: Rewrite the question as Which of the following maps highlights Jamaica? to align with visual option format.
New Question: Which of the following maps highlights Jamaica?
New Answer: A map with Jamaica highlighted
Reasoning: The new question directs students to identify the highlighted country from image options. The answer aligns with the original image and allows for intuitive visual recognition.

The output must be in the following JSON array format:
[
  {{
    "Q": "Generated visual question",
    "answer": "A short description of the image corresponding to the correct answer",
    "Reason": "Explanation of the reasoning process used to solve this question"
  }}
]
'''
class QuestionGeneration:
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = "qwen2-vl-7b-instruct"

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode("utf-8")
            return f"data:image/png;base64,{base64_data}"

    def build_prompt(self, raw_question, raw_answer, image=None):
        content_blocks = []
        if image and os.path.exists(image):
            image_data = self.encode_image_to_base64(image)
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": image_data}
            })
        text_prompt = PROMPT_TEMPLATE.format(
            question=raw_question,
            answer=raw_answer
        )
        content_blocks.append({"type": "text", "text": text_prompt.strip()})
        return [{"role": "user", "content": content_blocks}]

    def generate_question(self, raw_question, raw_answer, image=None):
        messages = self.build_prompt(raw_question, raw_answer, image)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True
            )
            full_response = ""
            for chunk in response:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    full_response += delta.content

            # More robust extraction of JSON list string
            start_idx = full_response.find("[")
            end_idx = full_response.rfind("]")
            if start_idx == -1 or end_idx == -1:
                raise ValueError("⚠️ Cannot find complete JSON array structure")

            json_str = full_response[start_idx:end_idx + 1]

            # Attempt to fix invalid backslashes
            json_str = json_str.replace("\\", "\\\\")  # double escaping
            result_data = json.loads(json_str)
            return result_data if isinstance(result_data, list) else [result_data]

        except Exception as e:
            print(f"❌ Error during generate_question: {e}")
            print("⚠️ Original response content:", full_response[:500])
            return None

    def save_to_csv(self, row, generated_results, filename="question_reason.csv", save_dir="./results"):
        if not generated_results:
            print("No generated data, skipping save.")
            return

        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)

        if os.path.exists(filepath):
            df_existing = pd.read_csv(filepath)
            count = df_existing[df_existing["Original Question ID"] == row["id"]].shape[0]
        else:
            df_existing = pd.DataFrame()
            count = 0

        records = []
        for i, result in enumerate(generated_results):
            new_question_id = f"{row['id']}_{count + i + 1:02d}"
            records.append({
                "Original Question ID": row["id"],
                "Generated Question ID": new_question_id,
                "question": row["question"],
                "choices": row["choices"],
                "answer": row["answer"],
                "image": row["image"] if pd.notna(row["image"]) else "",
                "task": row["task"],
                "grade": row["grade"],
                "split": row["split"],
                "convertible": row["convertible"],
                "Ref_Question": row["Ref_Question"],
                "Generated Question": result.get("Q", ""),
                "Correct Answer": result.get("answer", ""),
                "Reasoning Process": result.get("Reason", "")
            })

        df_new = pd.DataFrame(records)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"✅ Saved {len(records)} questions to: {filepath}")

    def generate_multiple_questions(self, row, num_rounds=5, filename="question_reason.csv"):
        print(f"\nGenerating {num_rounds} questions in one call...")
        result_list = self.generate_question(
            raw_question=row["question"],
            raw_answer=row["answer"],
            image=row["image"] if pd.notna(row["image"]) else None
        )
        if result_list:
            self.save_to_csv(row, result_list[:num_rounds], filename=filename)

    def generate_question_for_all(self, input_csv, output_csv, num_rounds=5, start_index=0, end_index=None):
        df = pd.read_csv(input_csv)
        total = len(df)
        if end_index is None:
            end_index = total
        print(f"There are {total} questions in total, preparing to process index range [{start_index}, {end_index})")
        for idx in range(start_index, min(end_index, total)):
            row = df.iloc[idx]
            print(f"Processing item {idx + 1}/{total} - Question ID: {row['id']}")
            self.generate_multiple_questions(
                row=row,
                num_rounds=num_rounds,
                filename=output_csv
            )

if __name__ == "__main__":
    qg = QuestionGeneration(api_key=your_api_key)
    input_path = "your_test.csv"
    output_path = "your_output.csv"
    qg.generate_question_for_all(
        input_csv=input_path,
        output_csv=output_path,
        start_index=0,
        end_index=205,
        num_rounds=5
    )
