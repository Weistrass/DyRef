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

MODEL = "deepseek-v3-local-II"

SYSTEM_MSG = """You are an expert text processing assistant specializing in image generation prompt optimization. Your \
task is to identify and REPLACE lighting effect descriptions in already subject-marked prompts with a lighting \
reference marker.

**Your Task:**
Given a text-to-image prompt that ALREADY has subject markers (e.g., "in reference image 1", "in reference image 2"), \
you must:
1. Identify the highest reference number in the prompt
2. Identify and REMOVE all lighting effect descriptions
3. INSERT "The overall lighting effect of the generated image should resemble the lighting in reference image Y." to \
REPLACE the removed descriptions
    - Y = highest existing reference number + 1
4. Preserve all subject markers, actions, relationships, scene descriptions

**Lighting Reference Number Calculation:**
- Find all existing "in reference image X" or "reference image X" markers in the prompt
- Identify the maximum X value (e.g., if you see references 1, 2, 3, 4, then max = 4)
- Lighting reference number = max + 1

**Lighting Description Identification (REMOVE AND REPLACE THESE):**

**REMOVE these types of lighting descriptions:**

1. **Lighting Style/Technique:**
    - General styles: "dramatic lighting", "soft lighting", "natural lighting", "artificial lighting"
    - Techniques: "rim lighting", "backlighting", "side lighting", "front lighting"
    - Studio techniques: "studio lighting", "three-point lighting", "key lighting", "fill lighting"
    - Photography lighting: "portrait lighting", "Rembrandt lighting", "butterfly lighting"

2. **Lighting Contrast/Intensity:**
    - Contrast: "high contrast", "low contrast", "chiaroscuro"
    - Key styles: "high key lighting", "low key lighting"
    - Intensity: "bright lighting", "dim lighting", "harsh lighting"
    - Shadow styles: "harsh shadows", "soft shadows", "deep shadows"

3. **Special Lighting Effects:**
    - Volumetric: "volumetric lighting", "god rays", "light rays", "light beams"
    - Glow effects: "glowing effect", "luminous", "radiant lighting"
    - Technical effects: "lens flare", "light leaks", "light streaks", "bloom effect"
    - Atmosphere lighting: "atmospheric lighting", "hazy lighting"

4. **Light Quality/Characteristics:**
    - Diffusion: "hard light", "soft light", "diffused light", "scattered light"
    - Direction: "directional lighting", "ambient lighting", "omnidirectional"
    - Coverage: "even lighting", "uneven lighting", "spotlit"

5. **Time-Based Lighting (as lighting effect, not scene time):**
    - "golden hour lighting", "blue hour lighting"
    - "sunset lighting", "sunrise lighting", "midday lighting"
    - "dawn lighting", "dusk lighting", "nighttime lighting"
    - Note: "at sunset" is scene time (KEEP); "sunset lighting" is lighting effect (REMOVE)

6. **Lighting Temperature/Color (as lighting effect):**
    - "warm lighting", "cool lighting", "cold lighting"
    - "warm glow", "cool tones in lighting"
    - Note: "warm colors" is color style (KEEP); "warm lighting" is lighting effect (REMOVE)

7. **Lighting Direction/Angle (as effect description):**
    - "lit from above", "lit from below", "top-lit", "bottom-lit"
    - "front-lit", "back-lit", "side-lit"
    - Note: "sunlight from window" is light source position (KEEP)

8. **Lighting Modifiers:**
    - "with soft shadows", "with no shadows", "shadowless"
    - "with highlights", "with bright highlights"
    - "illuminated", "well-lit", "poorly lit" (as lighting description)

**KEEP these elements (DO NOT REMOVE):**

1. **All Existing Reference Markers:**
    - Subject markers: "in reference image X"
    - Background markers: "using the scene of reference image X as background"
    - Style markers: "The overall visual and artistic style... reference image X"
    - Do NOT modify any existing markers

2. **Subject Physical Descriptions:**
    - Colors: "red car", "blue eyes", "golden fur"
    - Sizes: "large", "small", "tiny", "massive"
    - Textures: "fluffy", "smooth", "rough", "metallic"
    - Materials: "wooden", "glass", "chrome", "fabric"
    - Self-luminous objects: "glowing eyes" (if it's the object's property), "neon sign"

3. **Subject Actions and Poses:**
    - Actions: "holding", "running", "sitting", "jumping", "flying", "swimming"
    - Expressions: "smiling", "frowning", "looking at camera"
    - States: "with eyes closed", "sleeping", "awake"

4. **Object/Subject Relationships:**
    - Spatial: "next to", "on top of", "between", "under", "above", "behind"
    - Physical: "holding", "touching", "carrying", "wearing"
    - Interactions: "looking at", "reaching for"

5. **Scene/Environment Descriptions:**
    - Locations: "in a jungle", "at the park", "inside a room", "on a beach"
    - Settings: "tropical backdrop", "urban setting", "Victorian workshop"
    - Environment: "surrounded by trees", "with mountains", "field of flowers"
    - Landscape: "rocky cliff", "sandy desert", "ocean waves"

6. **Time/Weather as Scene Elements (NOT as lighting):**
    - Time of day: "at sunset", "at dawn", "at night", "in the evening", "at noon"
    - Weather: "in the rain", "with snow", "foggy", "cloudy", "stormy"
    - Natural phenomena: "with rainbow", "under stars", "with aurora"
    - Note: These describe WHEN or WHAT weather, not HOW the lighting looks

7. **Light Sources as Scene Objects (NOT lighting effects):**
    - Natural sources: "sun", "moon", "stars", "sunlight", "moonlight"
    - Artificial sources: "fireplace", "candle", "lamp", "neon signs", "streetlights"
    - Light source position: "sunlight through window", "light from door"
    - Note: "sunlight" (source) is KEPT; "soft lighting" (effect) is REMOVED

8. **Technical/Camera Settings:**
    - Focus: "shallow depth of field", "deep focus", "bokeh"
    - Composition: "rule of thirds", "centered", "symmetrical composition"
    - Camera: "wide angle", "macro lens", "telephoto"
    - Quality: "8k resolution", "sharp focus", "highly detailed"

9. **Artistic Style (visual/color style, not lighting):**
    - Color style: "vibrant colors", "muted colors", "pastel tones"
    - Art style: "cinematic", "photorealistic", "oil painting style"
    - Rendering: "3D render", "digital art", "watercolor"
    - Mood: "dreamy", "ethereal" (visual mood, not lighting mood)

10. **Quality/Detail Descriptors:**
    - "highly detailed", "intricate", "professional"
    - "masterpiece", "award-winning", "high quality"

11. **Spatial Scene Context:**
    - Distance: "in the distance", "far away", "in foreground", "in background"
    - Space: "spacious", "cramped", "vast"
    - Perspective: "viewed from above", "from the side"

12. **Color Descriptions (object colors, not lighting colors):**
    - "red", "blue", "golden", "silver" (when describing objects)
    - "colorful", "multicolored", "rainbow-colored"
    - "bright red", "deep blue" (color intensity, not light intensity)

**Key Distinctions:**

| Remove (Lighting Effect) | Keep (Scene/Object/Style) |
|-------------------------|---------------------------|
| "golden hour lighting" | "at sunset" (time) |
| "soft lighting" | "soft texture" / "soft colors" |
| "dramatic lighting" | "dramatic composition" / "dramatic scene" |
| "warm lighting" | "warm colors" / "warm tones" |
| "volumetric lighting" | "fog" / "mist" (weather) |
| "rim lighting" | "sun" / "bright sun" (light source) |
| "backlit" / "back-lit" | "sunlight from behind" (source position) |
| "high contrast lighting" | "high contrast colors" |
| "glowing effect" | "glowing object" (self-luminous) |
| "harsh shadows" | "shadows" (as scene element) |
| "illuminated by" | "near" / "in front of" (spatial) |
| "sunset lighting" | "at sunset" (scene time) |
| "soft shadows" | "cloudy" (weather causing natural soft shadows) |

**Lighting Marker Insertion Rules:**

1. **Placement:**
    - Insert at the END of the prompt, after all other descriptions
    - Place after background marker and style marker (if present)
    - Use this exact format as a separate sentence

2. **Format:**
    - Add exactly: "The overall lighting effect of the generated image should resemble the lighting in reference image \
Y."
    - Y = (highest existing reference number + 1)
    - Use proper capitalization and punctuation
    - Place as final sentence with period

3. **Replacement Strategy:**
    - Remove all lighting effect descriptions throughout the prompt
    - Add the lighting marker once at the end
    - Do not repeat the lighting marker
    - Ensure smooth grammatical flow

**Output Requirements:**
1. Return ONLY the modified prompt text
2. Do NOT add explanations, notes, or comments
3. Do NOT use quotes around the output
4. Maintain all existing subject markers unchanged
5. Remove all lighting descriptions and replace with single lighting marker at end
6. Ensure natural readability and proper grammar

**Examples:**

Example 1:
Input: "A slow loris with large round eyes in reference image 1 holding a chocolate ice cream bar in reference image \
2, tropical jungle backdrop with dappled sunlight, playful contrast between the primate's soft fur and the melting \
dessert, warm golden hour lighting, shallow depth of field highlighting textures, cinematic composition with vibrant \
colors and soft bokeh."

Analysis:
- Highest reference: 2
- Lighting reference: 3
- Remove: "with dappled sunlight" (lighting effect), "warm golden hour lighting" (lighting style)
- Keep: subject markers, "tropical jungle backdrop" (scene), "playful contrast" (style), "soft fur" (texture), \
technical details, "vibrant colors" (color style), "soft bokeh" (technical)

Output: "A slow loris with large round eyes in reference image 1 holding a chocolate ice cream bar in reference image \
2, tropical jungle backdrop, playful contrast between the primate's soft fur and the melting dessert, shallow depth of \
field highlighting textures, cinematic composition with vibrant colors and soft bokeh. The overall lighting effect of \
the generated image should resemble the lighting in reference image 3."

Example 2:
Input: "A cute white kitten with blue eyes in reference image 1 playing with yarn in reference image 2, cozy home \
interior with fireplace, soft lighting from window, bokeh effect, highly detailed fur texture. The overall visual and \
artistic style of the generated image should resemble the style of reference image 3."

Analysis:
- Highest reference: 3 (style)
- Lighting reference: 4
- Remove: "soft lighting from window" (lighting effect - note: "window" as source is kept in scene, but "soft \
lighting" effect is removed)
- Keep: all markers including style marker, scene descriptions, technical details

Output: "A cute white kitten with blue eyes in reference image 1 playing with yarn in reference image 2, cozy home \
interior with fireplace and window, bokeh effect, highly detailed fur texture. The overall visual and artistic style \
of the generated image should resemble the style of reference image 3. The overall lighting effect of the generated \
image should resemble the lighting in reference image 4."

"""



USER_TEMPLATE = """**Instructions:**
- Remove: lighting effect descriptions
- Replace with: lighting reference marker (auto-calculate number)
- Keep: all subject markers, actions, technical details

**Output format:** Modified prompt only, no explanations

Now process the given prompt: {original_prompt}
"""


def process_prompts(client: OpenAI, ori_prompt) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.9,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",
            "content": [
                {"type": "text", "text": USER_TEMPLATE.format(original_prompt=ori_prompt)}
                ]
                },
        ],
    )
    if resp.choices[0].message.content is None:
        max_retries = 3
        for _ in range(max_retries):
            print("Retrying...")
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",
                    "content": [
                        {"type": "text", "text": USER_TEMPLATE.format(original_prompt=ori_prompt)}
                    ]
                    },
                ],
                max_tokens=800,
                temperature=0.9,
                )
            if resp.choices[0].message.content is not None:
                break
        if resp.choices[0].message.content is None:
            return 'error'
    return resp.choices[0].message.content.strip()


def main():
    parser = argparse.ArgumentParser(description="Process prompts to add lighting reference markers.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root data directory (e.g. /path/to/data_4500)")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)
    with open(args.input, "r", encoding="utf-8") as f:
        subject_pairs = [json.loads(line) for line in f]

    for subject_pair in tqdm(subject_pairs):
        img_dir = '/'.join(subject_pair['image'].split('/')[:-1])
        img_dir_path = base_dir / img_dir
        reference_path = img_dir_path / "lighting/lighting_reference.jpeg"
        if not reference_path.exists():
            reference_path = img_dir_path / "lighting/lighting_reference.png"
            if not reference_path.exists():
                print(f"Warning: lighting reference not found for {img_dir}, skipping.")
                continue
        processed_prompt = process_prompts(client, subject_pair['prompt'])
        subject_pair['prompt'] = processed_prompt
        edit_images = subject_pair['edit_image']
        rel_path = f"{img_dir}/lighting/{reference_path.name}"
        edit_images.append(rel_path)
        subject_pair['edit_image'] = edit_images

        with open(args.output, "a", encoding="utf-8") as f:
            f.write(json.dump(subject_pair, ensure_ascii=False) + "\n")

    print("Done")

if __name__ == "__main__":
    main()
