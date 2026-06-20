import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

MODEL = "gemini-3-flash"

SYSTEM_MSG = """You are an expert text processing assistant specializing in image generation prompt augmentation. Your \
task is to locate subject mentions in text-to-image prompts and add reference markers to them.

**Your Task:**
Given an original text-to-image prompt and a list of subject names, you must:
1. Locate each subject name (or its close variation) in the original prompt
2. Insert "in reference image X" immediately after each located subject
3. Preserve the original prompt's grammar, flow, and meaning
4. Maintain the exact order and wording of the original prompt (except for the insertions)

**Matching Rules:**

1. **Flexible Matching**: Subject names may not match exactly with text in the prompt
    - Match core noun phrases even if descriptive words differ
    - Example: "A slow loris with round eyes" should match "A slow loris with large round eyes"
    - Focus on: main subject + key characteristics

2. **Case Insensitive**: Ignore case differences during matching

3. **Partial Word Matching**: Allow minor variations in descriptive words
    - "big dog" can match "large dog" or "huge dog"
    - "red car" can match "bright red car"

4. **First Occurrence**: If a subject appears multiple times, mark only the FIRST occurrence

5. **Sequential Processing**: Process subjects in list order (1, 2, 3, ...)

**Insertion Format:**
- Add exactly: " in reference image X" (with leading space, X is the number)
- Insert immediately after the matched subject phrase
- Do NOT add any extra punctuation or words

**Edge Cases:**
- If a subject is NOT found in the prompt: Skip it (do not add marker)
- If subjects overlap: Mark the longer/more specific one first
- Preserve all original commas, punctuation, and formatting

**Output Requirements:**
1. Return ONLY the modified prompt text
2. Do NOT add explanations, notes, or comments
3. Do NOT use quotes around the output
4. Keep the exact same structure as the original prompt

**Quality Checks:**
- Ensure natural readability after insertion
- Verify all reference numbers are correct (1, 2, 3...)
- Confirm no original text was removed or altered (except for insertions)

**Examples:**

Example 1:
Original: "A golden retriever running through a field of sunflowers"
Subjects: 
1. golden retriever
2. field of sunflowers
Output: "A golden retriever in reference image 1 running through a field of sunflowers in reference image 2"

Example 2:
Original: "A steampunk robot with brass gears holding an antique pocket watch"
Subjects:
1. steampunk robot with brass gears
2. antique pocket watch
Output: "A steampunk robot with brass gears in reference image 1 holding an antique pocket watch in reference image 2"

Example 3:
Original: "A majestic lion with a flowing mane standing on a rocky cliff at sunset"
Subjects:
1. majestic lion with flowing mane
2. rocky cliff
Output: "A majestic lion with a flowing mane in reference image 1 standing on a rocky cliff in reference image 2 at \
sunset"
"""



USER_TEMPLATE = """Please process the following text-to-image prompt by adding reference image markers.

**Original Prompt:**
{original_prompt}

**Subject List (in order):**
{subject_list}

**Instructions:**
1. Find each subject (or its close variation) in the original prompt
2. Add "in reference image X" after each subject (X = position in list, starting from 1)
3. Return ONLY the modified prompt, no explanations

**Example:**
Original: "A fluffy cat sitting on a red cushion"
Subjects: ["fluffy cat", "red cushion"]
Output: "A fluffy cat in reference image 1 sitting on a red cushion in reference image 2"

Now process the given prompt:"""


class SemanticSubjectImageMatcher:
    """使用语义相似度的匹配器"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic matcher

        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)

    def match(
        self,
        subject_names: List[str],
        image_paths: List[str]
    ) -> Dict[str, str]:
        """
        Match using semantic similarity

        Args:
            subject_names: List of subjects
            image_paths: List of image paths

        Returns:
            Dict mapping subject name to image path
        """
        # Extract filenames for encoding
        filenames = [os.path.splitext(os.path.basename(p))[0] for p in image_paths]

        # Encode subjects and filenames
        subject_embeddings = self.model.encode(subject_names, convert_to_tensor=True)
        filename_embeddings = self.model.encode(filenames, convert_to_tensor=True)

        # Calculate cosine similarity
        similarity_matrix = util.cos_sim(subject_embeddings, filename_embeddings)

        # Convert to numpy for easier manipulation
        similarity_matrix = similarity_matrix.cpu().numpy()

        # Greedy matching
        matched_subjects = set()
        matched_images = set()
        matches = {}

        # Flatten and sort by similarity
        pairs = []
        for i in range(len(subject_names)):
            for j in range(len(image_paths)):
                pairs.append((i, j, similarity_matrix[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)

        # Select best non-overlapping matches
        for i, j, _ in pairs:
            if i not in matched_subjects and j not in matched_images:
                matches[subject_names[i]] = image_paths[j]
                matched_subjects.add(i)
                matched_images.add(j)
                if len(matches) == len(subject_names):
                    break

        return matches


def process_prompts(client: OpenAI, ori_prompt, subject_lists) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user",
        "content": [
            {"type": "text", "text": USER_TEMPLATE.format(
                original_prompt=ori_prompt,
                subject_list=subject_lists
            )}
        ]},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.9,
            max_tokens=4096,
            messages=messages,
            extra_body={"thinking_level": 'low'}
        )
        if resp.choices[0].message.content is None:
            max_retries = 3
            for _ in range(max_retries):
                print("Retrying...")
                resp = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.9,
                    extra_body={"thinking_level": 'low'})
                if resp.choices[0].message.content is not None:
                    break
            if resp.choices[0].message.content is None:
                return 'error'
        return resp.choices[0].message.content.strip()
    except Exception as e:  # pylint: disable=broad-except
        print(f"Request failed - prompt: {ori_prompt}, subjects: {subject_lists}, error: {e}")
        return 'error'


def main():
    parser = argparse.ArgumentParser(description="Process prompts to add subject reference markers.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing subject subdirectories")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--num_subjects", type=int, required=True, help="Number of subjects (e.g. 5)")
    args = parser.parse_args()

    root_dir = Path(args.data_dir)
    output_path = Path(args.output)

    for prompt_dir in tqdm(root_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        scores_path = prompt_dir / "scores_vlm.json"
        if not scores_path.exists():
            continue
        with open(scores_path, "r", encoding="utf-8") as f:
            scores = json.load(f)
        if scores['min_score'] < 3:
            continue
        dic = {}
        target_img_path = f"{args.num_subjects}_subjects/{prompt_dir.name}/{prompt_dir.name}.jpeg"
        dic['image'] = target_img_path
        with open(prompt_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        ori_prompt = metadata['prompt']
        subject_names = metadata['subjects']
        processed_prompt = process_prompts(client, ori_prompt, subject_names)
        dic['prompt'] = processed_prompt
        transfer_subject_paths = [img.name for img in (prompt_dir / "transfer_subjects").iterdir()]

        matcher = SemanticSubjectImageMatcher()
        matches = matcher.match(subject_names, transfer_subject_paths)
        edit_images = []
        for subject in subject_names:
            edit_images.append(f"{args.num_subjects}_subjects/{prompt_dir.name}/transfer_subjects/{matches[subject]}")
        dic['edit_image'] = edit_images

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dic, ensure_ascii=False) + "\n")

    print("Done")


if __name__ == "__main__":
    main()
