import os
import json
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

MODEL = "gemini-3-flash"

SYSTEM_MSG = """You are an expert text processing assistant specializing in image generation prompt optimization. Your \
task is to identify and REPLACE visual/artistic style descriptions in already subject-marked prompts with a style \
reference marker.

**Your Task:**
Given a text-to-image prompt that ALREADY has subject markers (e.g., "in reference image 1", "in reference image 2"), \
you must:
1. Identify the highest subject reference number in the prompt
2. Identify and REMOVE all visual/artistic style descriptions
3. INSERT "The overall visual and artistic style of the generated image should resemble the style of reference image \
Y." to REPLACE the removed descriptions
    - Y = highest subject reference number + 1
4. Preserve all subject markers, actions, relationships, background descriptions.

**Style Reference Number Calculation:**
- Find all existing "in reference image X" markers in the prompt
- Identify the maximum X value (e.g., if you see references image 1, 2, 3, then max = 3)
- Style reference number = max + 1

**Style Description Identification (REMOVE AND REPLACE THESE):**

**REMOVE these types of descriptions:**
1. **Artistic/Rendering Style:**
    - Style types: "photorealistic", "oil painting style", "watercolor", "anime style", "cartoon"
    - Art movements: "impressionist", "baroque", "art deco", "minimalist", "surrealist"
    - Rendering: "3D render", "hand-drawn", "digital art", "sketch style"
    - Visual aesthetics: "cinematic", "dreamy", "ethereal", "gritty", "vintage"

2. **Color/Tone Style:**
    - Color schemes: "vibrant colors", "muted colors", "pastel tones", "monochrome"
    - Color temperature: "warm tones", "cool tones", "saturated", "desaturated"
    - Color grading: "color grading", "color palette", "color correction"
    - Mood colors: "dark and moody", "bright and cheerful"

3. **Lighting Style (NOT specific light sources):**
    - Style descriptors: "dramatic lighting", "soft lighting", "natural lighting"
    - Lighting techniques: "rim lighting", "backlighting", "side lighting", "studio lighting"
    - Contrast: "high contrast", "low contrast", "chiaroscuro", "low key", "high key"
    - Atmospheric lighting: "volumetric lighting", "atmospheric lighting", "moody lighting"
    - Photography lighting terms: "golden hour lighting", "blue hour lighting"

4. **Technical/Photography Style:**
    - Focus: "shallow depth of field", "deep depth of field", "bokeh", "soft bokeh"
    - Composition: "cinematic composition", "rule of thirds", "symmetrical composition"
    - Camera/lens style: "macro photography", "wide angle", "telephoto", "fisheye"
    - Photography type: "professional photography", "portrait photography", "landscape photography"
    - Effects: "film grain", "lens flare", "motion blur" (as style effect)

5. **Quality/Detail Style:**
    - Detail level: "highly detailed", "ultra detailed", "intricate details", "fine details"
    - Focus quality: "sharp focus", "tack sharp", "soft focus"
    - Resolution: "8k resolution", "4k", "high resolution", "ultra HD"
    - Quality terms: "professional", "masterpiece", "award-winning", "high quality"

6. **Texture/Material Style (overall visual quality):**
    - Surface quality: "smooth texture", "rough texture", "glossy", "matte"
    - Visual sharpness: "crisp", "soft", "sharp"
    - When describing overall textural style (not specific object textures)

7. **Atmosphere/Mood Style (visual/aesthetic mood):**
    - Visual mood: "moody", "bright", "dark atmosphere"
    - Aesthetic feel: "nostalgic", "futuristic", "retro"
    - Emotional tone: "cheerful", "somber", "energetic"

**KEEP these elements (DO NOT REMOVE):**

1. **Subject Reference Markers:**
    - ALL existing "in reference image X" markers must be preserved
    - Do not modify or remove any subject markers

2. **Background/Scene Reference Markers:**
    - "using the scene of reference image X as background" must be preserved
    - Do not modify this if present

3. **Subject Physical Descriptions:**
    - Size, color, shape of subjects: "large round eyes", "red sports car", "fluffy fur"
    - Subject characteristics: "vintage", "antique", "modern" (when describing an object itself)
    - Specific textures of objects: "furry", "metallic", "wooden" (object properties)

4. **Actions and Poses:**
    - Verbs: "holding", "running", "sitting", "standing", "playing", "floating", "swimming"
    - Expressions: "smiling", "frowning", "looking at camera"
    - States: "with eyes closed", "sleeping", "jumping"

5. **Object/Subject Relationships:**
    - Spatial relations: "next to", "on top of", "between", "under", "above"
    - Interactions: "holding", "touching", "carrying"
    - Relative positions between subjects

6. **Location/Setting/Environment:**
    - Places: "in a jungle", "at the park", "inside a room", "on a beach", "in space"
    - Settings: "tropical jungle backdrop", "urban city", "Victorian workshop"
    - Environment elements: "surrounded by trees", "with mountains", "field of flowers"
    - Background objects: "buildings", "clouds", "stars", "furniture"

7. **Weather/Time as Scene Elements (NOT as style):**
    - Time of day: "at sunset", "at dawn", "at night", "midday"
    - Weather: "in the rain", "with snow", "foggy", "stormy"
    - Natural phenomena: "with sunbeams", "under stars", "rainbow"
    - Note: Keep "sunset" but remove "sunset lighting" or "golden hour lighting"

8. **Specific Light Sources (scene elements, NOT lighting style):**
    - Natural sources: "sunlight", "moonlight", "starlight"
    - Artificial sources: "fireplace", "candle", "neon signs", "streetlights"
    - Note: "sunlight through window" is a scene element; "dramatic lighting" is a style

9. **Spatial Scene Context:**
    - Distance: "in the distance", "far away", "in the background", "in the foreground"
    - Space description: "spacious room", "cramped space"
    - Perspective: "viewed from above", "from the side"



**Style Marker Insertion Rules:**

1. **Placement:**
    - Insert at the END of the prompt after all other descriptions

2. **Format:**
    - Add exactly: "The overall visual and artistic style of the generated image should resemble the style of \
reference image Y."
    - Include leading period for proper grammar
    - Y = (highest existing subject reference number + 1)

3. **Replacement Strategy:**
    - Remove all style descriptions throughout the prompt
    - Add the style marker once at the end
    - Do not repeat the style marker

**Output Requirements:**
1. Return ONLY the modified prompt text
2. Do NOT add explanations, notes, or comments
3. Do NOT use quotes around the output
4. Maintain all existing subject markers unchanged
5. Ensure natural readability and proper grammar

**Examples:**

Example 1:
Input: "A slow loris with large round eyes in reference image 1 holding a chocolate ice cream bar in reference image \
2, tropical jungle backdrop with dappled sunlight, playful contrast between the primate's soft fur and the melting \
dessert, warm golden hour lighting, shallow depth of field highlighting textures, cinematic composition with vibrant \
colors and soft bokeh."

Analysis:
- Highest subject reference: 2
- Style reference: 3
- Remove: "with dappled sunlight" (lighting), "warm golden hour lighting" (lighting), "playful contrast between the \
primate's soft fur and the melting dessert" (artistic style), "shallow depth of field highlighting textures" \
(technical/camera details), "cinematic composition with vibrant colors and soft bokeh" (artistic style)
- Keep: subject markers, "holding" (action), "tropical jungle backdrop" (background)

Output: "A slow loris with large round eyes in reference image 1 holding a chocolate ice cream bar in reference image \
2, tropical jungle backdrop. The overall visual and artistic style of the generated image should resemble the style of \
reference image 3."

Example 2:
Input: "A golden retriever in reference image 1 running through a field of sunflowers in reference image 2 under blue \
sky with fluffy clouds, professional photography, shallow depth of field"

Analysis:
- Highest subject reference: 2
- Style reference: 3
- Remove: "professional photography" (technical/camera details)
- Keep: subject markers, "running through" (action), "under blue sky with fluffy clouds" (background)

Output: "A golden retriever in reference image 1 running through a field of sunflowers in reference image 2 under blue \
sky with fluffy clouds. The overall visual and artistic style of the generated image should resemble the style of \
reference image 3."

Example 3:
Input: "A cute white kitten with blue eyes in reference image 1 playing with yarn in reference image 2, cozy home \
interior with fireplace, soft lighting, bokeh effect, highly detailed fur texture"

Analysis:
- Highest reference: 2
- Style reference: 3
- Remove: "soft lighting" (lighting style), "bokeh effect" (technical), "highly detailed" (quality)
- Keep: subject markers, "with blue eyes" (subject feature), "playing with" (action), "cozy home interior with \
fireplace" (scene), "fur texture" (subject property)

Output: "A cute white kitten with blue eyes in reference image 1 playing with yarn in reference image 2, cozy home \
interior with fireplace, fur texture. The overall visual and artistic style of the generated image should resemble the \
style of reference image 3."
"""



USER_TEMPLATE = """**Instructions:**
- Remove: visual/artistic style descriptions, camera details, lighting details
- Replace with: style reference marker (auto-calculate number)
- Keep: all subject markers, actions, scene, background

**Output format:** Modified prompt only, no explanations

Now process the given prompt: {original_prompt}
"""


def process_prompts(client: OpenAI, ori_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_TEMPLATE.format(original_prompt=ori_prompt)}
            ],
        },
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.9,
            max_tokens=4096,
            messages=messages,
            extra_body={"thinking_level": "low"},
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
                    extra_body={"thinking_level": "low"},
                )
                if resp.choices[0].message.content is not None:
                    break
            if resp.choices[0].message.content is None:
                return "error"
        return resp.choices[0].message.content.strip()
    except Exception as e:  # pylint: disable=broad-except
        print(f"API request failed: {e}")
        return "error"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process prompts with style reference markers.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base data directory")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    with open(args.input, "r", encoding="utf-8") as f:
        subject_pairs = [json.loads(line) for line in f]

    for subject_pair in tqdm(subject_pairs):
        image_dir = '/'.join(subject_pair['image'].split('/')[:-1])
        image_dir_path = base_dir / image_dir
        style_dir = image_dir_path / "style"
        if not style_dir.exists():
            continue
        style_score_path = style_dir / "score.json"
        if not style_score_path.exists():
            continue
        with open(style_score_path, "r", encoding="utf-8") as f:
            style_scores = json.load(f)
        if style_scores["score"] < 6:
            continue
        if not (image_dir_path / "style/target.png").exists():
            print(f"target.png not exists at {image_dir_path}")
            continue

        subject_pair["image"] = f"{image_dir}/style/target.png"
        subject_pair["prompt"] = process_prompts(client, subject_pair["prompt"])
        subject_pair["edit_image"].append(f"{image_dir}/style/reference.png")

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dumps(subject_pair, ensure_ascii=False) + "\n")

    print("Done")


if __name__ == "__main__":
    main()
