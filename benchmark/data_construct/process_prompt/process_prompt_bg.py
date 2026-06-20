import os
import json
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

MODEL = "gemini-3-flash"

SYSTEM_MSG = """You are an expert text processing assistant specializing in image generation prompt optimization. Your \
task is to identify and REPLACE scene/environment descriptions in already subject-marked prompts with a background \
reference marker.

**Your Task:**
Given a text-to-image prompt that ALREADY has subject markers (e.g., "in reference image 1", "in reference image 2"), \
you must:
1. Identify the highest subject reference number in the prompt
2. Identify and REMOVE all scene/environment/atmosphere descriptions
3. INSERT "using the scene of reference image Y as background" to REPLACE the removed descriptions
   - Y = highest subject reference number + 1
4. Preserve all subject markers, actions, relationships, technical details, and artistic style

**Background Reference Number Calculation:**
- Find all existing "in reference image X" markers in the prompt
- Identify the maximum X value (e.g., if you see references image 1, 2, 3, then max = 3)
- Background reference number = max + 1

**Scene Description Identification (REMOVE AND REPLACE THESE):**

**REMOVE these types of descriptions:**

1. **Location/Setting Descriptors:**
   - "in a jungle", "at the park", "inside a room", "on a beach"
   - "tropical jungle backdrop", "urban city setting", "forest environment"
   - "cozy living room", "Victorian workshop", "medieval castle interior"
   - "office environment", "studio setting", "outdoor scene"

2. **Environment/Landscape Elements:**
   - "surrounded by trees", "with mountains in background", "field of flowers"
   - "rocky cliff", "sandy desert", "ocean waves"
   - "backdrop with...", "setting of...", "environment of..."
   - Background objects that set the scene (not subjects)

3. **Atmospheric/Weather/Time Descriptions:**
   - "with dappled sunlight", "at sunset", "at golden hour", "in the rain"
   - "under blue sky", "with clouds", "in fog", "at night", "during storm"
   - "with sunbeams", "in moonlight", "starry night"
   - Time-of-day indicators: "dawn", "dusk", "midday"

4. **Spatial Scene Context:**
   - "in the distance", "far away", "in the background" (when describing environment)
   - "spacious", "cramped" (describing the space itself)

5. **Lighting Style (not time/atmosphere):**
- "dramatic lighting", "soft lighting", "rim lighting", "studio lighting"
- "high contrast", "low key", "volumetric lighting"
- Keep style descriptors like "warm lighting" or "dramatic lighting"
- Remove time indicators like "golden hour" or "sunset lighting"

**KEEP these elements (DO NOT REMOVE):**

1. **Subject Reference Markers:**
   - ALL existing "in reference image X" markers must be preserved
   - Do not modify or remove any subject markers

2. **Subject Actions and Poses:**
   - "holding", "sitting", "running", "standing", "playing", "floating"
   - "with eyes closed", "smiling", "looking at camera"
   - Any verb describing what subjects are doing

3. **Object/Subject Relationships:**
   - "between", "next to", "on top of", "holding", "on", "under"
   - Spatial relationships between subjects themselves

4. **Technical/Camera Details:**
   - "shallow depth of field", "bokeh", "depth of field"
   - "cinematic composition", "rule of thirds"
   - "professional photography", "8k resolution", "highly detailed"
   - "sharp focus", "macro lens"

5. **Artistic Style:**
   - "vibrant colors", "soft bokeh", "cinematic"
   - "playful contrast", "textures", "highly detailed"
   - "photorealistic", "oil painting style", "watercolor"

6. **Quality/Detail Descriptors:**
   - "highly detailed", "intricate", "sharp focus"
   - "professional", "masterpiece", "high quality"

**Background Marker Insertion Rules:**

1. **Placement:**
   - Insert where scene/environment descriptions were removed
   - Usually after subjects and their actions
   - Before technical details and artistic style descriptors

2. **Format:**
   - Add exactly: ", using the scene of reference image Y as background"
   - Include leading comma for proper grammar
   - Y = (highest existing subject reference number + 1)

3. **Replacement Strategy:**
   - Remove entire scene description phrases/clauses
   - Insert the background marker in their place
   - Ensure smooth grammatical flow with proper comma usage

**Output Requirements:**
1. Return ONLY the modified prompt text
2. Do NOT add explanations, notes, or comments
3. Do NOT use quotes around the output
4. The output should be shorter than the original (scene descriptions removed)
5. Maintain all existing subject markers unchanged
6. Ensure natural readability and proper grammar

**Examples:**

Example 1:
Input: "A slow loris with large round eyes in reference image 1 holding a chocolate ice cream bar in reference image \
2, tropical jungle backdrop with dappled sunlight, playful contrast between the primate's soft fur and the melting \
dessert, warm golden hour lighting, shallow depth of field highlighting textures, cinematic composition with vibrant \
colors and soft bokeh."

Analysis:
- Highest subject reference: 2
- Background reference: 3
- Remove: "tropical jungle backdrop with dappled sunlight" (location + atmosphere), "golden hour" (time), "warm \
lighting" (style)
- Keep: subject markers, "holding" (action), "playful contrast" (artistic), technical details

Output: "A slow loris with large round eyes in reference image 1 holding a chocolate ice cream bar in reference image \
2, using the scene of reference image 3 as background, playful contrast between the primate's soft fur and the melting \
dessert, shallow depth of field highlighting textures, cinematic composition with vibrant colors and soft bokeh."

Example 2:
Input: "A golden retriever in reference image 1 running through a field of sunflowers in reference image 2 under blue \
sky with fluffy clouds, professional photography, shallow depth of field"

Analysis:
- Highest subject reference: 2
- Background reference: 3
- Remove: "under blue sky with fluffy clouds" (atmosphere)
- Keep: subject markers, "running through" (action), technical details

Output: "A golden retriever in reference image 1 running through a field of sunflowers in reference image 2, using the \
scene of reference image 3 as background, professional photography, shallow depth of field"


"""



USER_TEMPLATE = """**Instructions:**
- Remove: scene descriptions, locations, weather, atmosphere, lighting, time
- Replace with: background reference marker (auto-calculate number)
- Keep: all subject markers, actions, technical details

**Output format:** Modified prompt only, no explanations

Now process the given prompt: {original_prompt}
"""



def process_prompts(client: OpenAI, ori_prompt) -> str:
    messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",
            "content": [
                {"type": "text", "text": USER_TEMPLATE.format(original_prompt=ori_prompt)}
            ]
            },
        ]
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
    content = resp.choices[0].message.content.strip()
    return content


def main():
    parser = argparse.ArgumentParser(description="Process prompts to add background reference markers.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory (e.g. /path/to/data_2500)")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    with open(args.input, "r", encoding="utf-8") as f:
        subject_pairs = [json.loads(line) for line in f]
    for subject_pair in tqdm(subject_pairs):
        img_dir = '/'.join(subject_pair['image'].split('/')[:-1])
        img_dir_path = base_dir / img_dir
        background_dir = img_dir_path / "background"
        if not background_dir.exists():
            continue
        bg_score_path = background_dir / "vlm_eval.json"
        if not bg_score_path.exists():
            continue
        with open(bg_score_path, "r", encoding="utf-8") as f:
            bg_scores = json.load(f)
        if bg_scores['final_score'] < 8:
            continue
        processed_prompt = process_prompts(client, subject_pair['prompt'])
        subject_pair['prompt'] = processed_prompt
        edit_images = subject_pair['edit_image']
        edit_images.append(f"{img_dir}/background/background.png")
        subject_pair['edit_image'] = edit_images
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(subject_pair, ensure_ascii=False) + "\n")

    print("Done")

if __name__ == "__main__":
    main()
