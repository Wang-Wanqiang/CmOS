import os
import re
import json
import base64
import pandas as pd
from openai import OpenAI

class VisualOptionDescriptionGenerator:
    def __init__(self, api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model_name="qwen-vl-max"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def build_prompt(self, row):

        question = row['Generated Question']
        answer = row['Correct Answer']
        reasoning = row['Reasoning Process']

        prompt = f"""
        You are a teacher responsible for developing multiple-choice questions with visual options. 
        Please generate a set of options (including the correct answer and distractors) based on the question stem, the correct answer, and the reasoning process provided.
        Given:
        Question stem: {question}
        Correct answer: {answer}
        Reasoning process: {reasoning}
        Please design a set of visual options for this question, including the correct option and 1 to 3 distractors (number of distractors depends on the question). 
        Provide very detailed visual descriptions for the correct option and distractors. 
        Besides describing what objects are present, also describe brightness, contrast, and structure (edges, textures, local structures). 
        If those are not the key distinguishing features among options, keep them as consistent as possible, so these descriptions can be directly used by visual models to generate images. 
        The output format should be as follows:
{{
    "Correct Option": {{
        "Title": "Title of the correct option",
        "Description": "Visual description of the correct option"
    }},
    "Distractors 1": {{
        "Title": "Title of distractor option 1",
        "Description": "Visual description of distractor option 1"
    }},
    "Distractors 2": {{
        "Title": "Title of distractor option 2",
        "Description": "Visual description of distractor option 2"
    }},
    "Distractors 3": {{
        "Title": "Title of distractor option 3",
        "Description": "Visual description of distractor option 3"
    }}
}}
        Please generate the options content according to this format and descriptions.
        """

        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    def parse_response(self, full_response):
        """Extract JSON from model output and return structured options."""
        try:
            # Extract the outermost JSON object
            json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
            if not json_match:
                raise ValueError("Failed to extract valid JSON")

            json_str = json_match.group(0)
            # Fix illegal backslashes
            json_str = re.sub(r'\\(?![\"/bfnrtu])', r'\\\\', json_str)
            data = json.loads(json_str, strict=False)

            result = {
                "Correct Option": {
                    "Title": data["Correct Option"].get("Title", ""),
                    "Description": data["Correct Option"].get("Description", "")
                }
            }

            for i in range(1, 4):
                key = f"Distractors {i}"
                if key in data:
                    result[key] = {
                        "Title": data[key].get("Title", ""),
                        "Description": data[key].get("Description", "")
                    }

            return result

        except Exception as e:
            print(f"❌ JSON parsing failed: {e}")
            print(f"⚠️ Model response:\n{full_response[:500]}")
            return None

    def generate_visual_options(self, row):
        """Call the large model to generate textual descriptions of visual options."""
        messages = self.build_prompt(row)

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

            print("📩 Full model output:", full_response[:300])  # for debugging
            return self.parse_response(full_response)

        except Exception as e:
            print(f"❌ Request failed: {e}")
            return None

    def process_file(self, input_path, output_path, start=0, end=None):
        """Batch process input file, generate visual option descriptions for each row, support index range."""
        try:
            # Attempt to read CSV file with specified encoding
            df = pd.read_csv(input_path, encoding='ISO-8859-1')  # or try 'latin1' encoding
        except UnicodeDecodeError as e:
            print(f"❌ Encoding error while reading file: {e}")
            return

        if end is None:
            end = len(df)

        all_results = []

        for idx in range(start, end):
            row = df.iloc[idx]
            print(f"🔄 Processing question at row {idx}: {row['Generated Question']}")
            result = self.generate_visual_options(row)
            if result:
                all_results.append({**row, **result})
            else:
                all_results.append({**row, "Correct Option": "ERROR", "Distractors 1": "ERROR"})

        pd.DataFrame(all_results).to_csv(output_path, index=False)
        print(f"✅ Saved results from row {start} to {end} into {output_path}")


if __name__ == "__main__":
    api_key="your_api"
    input_file = "your_questions.csv"
    output_file = "your_visual_option_descriptions2.csv"

    generator = VisualOptionDescriptionGenerator(api_key)
    generator.process_file(input_file, output_file, start=0, end=null)
