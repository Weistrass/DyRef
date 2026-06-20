import os
import json
import argparse
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY'],
    base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
)

MODEL = "gemini-3-flash"

SYSTEM_MSG = """You are an expert text processing assistant specializing in image generation prompt optimization. Your \
task is to INSERT pose reference markers for specific characters while preserving essential pose information.

**Your Task:**
Given a text-to-image prompt with existing markers AND a pose reference image filename, you must:
1. Identify the character in the prompt that matches the pose reference filename
2. Find the highest reference number in the prompt
3. INSERT "with pose from reference image Y" immediately after the matched character's subject marker
    - Y = highest existing reference number + 1
4. REMOVE detailed pose descriptions for that character ONLY
5. PRESERVE basic/essential pose descriptions that are critical for image generation
6. Keep all other characters and their descriptions unchanged

**Reference Number Calculation:**
- Find all existing reference markers:
    - Subject markers: "in reference image X"
    - Background markers: "using the scene of reference image X as background"
    - Style markers: "style of reference image X"
    - Lighting markers: "lighting in reference image X"
    - Pose markers: "pose from reference image X" (from previous additions)
- Identify the maximum X value
- New pose reference number = max + 1

**Character Matching Rules:**

1. **Filename Pattern Recognition:**
    - "a_man.png" → match "man", "male", "boy", "gentleman"
    - "woman_pose.png" → match "woman", "female", "girl", "lady"
    - "child_sitting.png" → match "child", "kid", "toddler"
    - "astronaut.png" → match "astronaut"
    - "dancer_1.png" → match first "dancer" in multi-character scenes

2. **Fuzzy Matching Priority:**
    - Exact match: "superman.png" → "superman"
    - Partial match: "doctor.png" → "doctor", "physician", "medical professional"
    - Semantic match: "athlete.png" → "runner", "player", "sportsman"
    - Descriptive match: "old_man.png" → "elderly man", "old gentleman"

**Pose Description Classification:**

**PRESERVE these ESSENTIAL/BASIC poses (critical for image generation):**

1. **Primary Body Position (keep these):**
    - Standing positions: "standing", "standing upright"
    - Sitting positions: "sitting", "seated"
    - Lying positions: "lying down", "lying on back/side"
    - Crouching/kneeling: "crouching", "kneeling"
    - Floating/suspended: "floating", "suspended", "hovering"

2. **Fundamental Movement State (keep these):**
    - Basic actions: "walking", "running", "jumping", "flying"
    - General motion: "in motion", "moving", "dancing", "swimming"
    - Static state: "still", "stationary"
    - Note: Keep the BASIC action type, remove detailed movement descriptions

3. **Essential Orientation (keep when critical):**
    - Basic facing: "facing forward", "side view", "back view"
    - Only when it defines the primary composition
    - Remove if redundant with camera angle

4. **Critical Scene Interactions (keep these):**
    - Positional verbs: "on" (on chair), "under" (under tree), "in" (in water)
    - These define the character's relationship with the scene
    - "sitting on chair" → KEEP "sitting on chair"
    - "standing on cliff" → KEEP "standing on cliff"

**REMOVE these DETAILED/COMPLEX poses (pose reference will control these):**

1. **Limb Positions (remove these):**
    - Arm details: "arms crossed", "arms raised", "one arm up", "arms at sides"
    - Hand details: "hands on hips", "hands behind back", "hands clasped"
    - Leg details: "legs crossed", "one leg raised", "legs spread", "feet together"
    - Finger details: "pointing", "thumbs up", "peace sign", "fingers spread"

2. **Body Part Angles/Directions (remove these):**
    - Head: "head tilted left", "head turned 45 degrees", "looking over shoulder"
    - Torso: "torso twisted", "leaning forward 30 degrees", "bent at waist"
    - Limbs: "elbow bent 90 degrees", "knee slightly bent"

3. **Detailed Gestures and Expressions (remove these):**
    - Hand gestures: "waving", "pointing at", "beckoning", "saluting"
    - Facial actions: "smiling", "with eyes closed", "mouth open", "winking"
    - Complex expressions: "looking intensely", "gazing wistfully"

4. **Postural Details (remove these):**
    - Stance specifics: "wide stance", "feet shoulder-width apart", "weight on one leg"
    - Posture adjectives: "slouching", "upright posture", "shoulders back"
    - Balance descriptions: "balanced on", "teetering", "steadied by"

5. **Dynamic Action Details (remove these):**
    - Movement specifics: "mid-stride", "in mid-air", "about to land"
    - Action modifiers: "gracefully dancing", "energetically jumping", "slowly walking"
    - Transition states: "turning around", "getting up", "reaching for"

6. **Complex Interaction Actions (remove these):**
    - Detailed interactions: "gently holding", "firmly gripping", "carefully touching"
    - Action progression: "in the process of", "beginning to", "finishing"
    - Keep simple object presence: "holding" → "with"

**Pose Marker Insertion Format:**

**For Single Character:**
- Insert immediately after the character's subject marker
- Format: "[Character description] in reference image X with pose from reference image Y"
- Example: "A man in reference image 1 with pose from reference image 4"

**For Multiple Characters (only mark the matched one):**
- Insert only after the matched character
- Other characters remain unchanged
- Example: "A man in reference image 1 with pose from reference image 5 next to a woman in reference image 2 sitting"

**Rewriting Strategy:**

1. **Basic Pose + Reference:**
    - Original: "A man standing with arms crossed and head tilted"
    - Pose file: "man_pose.png"
    - Result: "A man in reference image 1 standing with pose from reference image 4"
    - Logic: Keep "standing" (basic), remove "arms crossed and head tilted" (detailed)

2. **Action + Reference:**
    - Original: "A dancer jumping with arms spread wide and legs split"
    - Pose file: "dancer.png"
    - Result: "A dancer in reference image 1 jumping with pose from reference image 4"
    - Logic: Keep "jumping" (basic action), remove limb details

3. **Sitting + Reference:**
    - Original: "A woman sitting on a chair with legs crossed and hands folded in lap"
    - Pose file: "woman_sitting.png"
    - Result: "A woman in reference image 1 sitting on a chair with pose from reference image 4"
    - Logic: Keep "sitting on a chair" (basic + scene interaction), remove limb positions

4. **Multiple Objects:**
    - Original: "A man holding a sword and shield with a confident stance"
    - Pose file: "warrior.png"
    - Result: "A man in reference image 1 with a sword in reference image 2 and shield in reference image 3 with pose \
from reference image 6"
    - Logic: Convert "holding" to "with", remove "confident stance", add pose reference

**Edge Cases:**

1. **No Clear Match:**
    - If filename doesn't clearly match any character → return original prompt unchanged
    - Add a note: "[Note: Could not identify matching character for pose reference]"

2. **Multiple Possible Matches:**
    - If filename is ambiguous (e.g., "person.png" with multiple people)
    - Match the FIRST/PRIMARY character mentioned
    - Or use contextual clues from filename (e.g., "person_left.png" → leftmost character)

3. **Already Has Pose Reference:**
    - If character already has "with pose from reference image X"
    - Do not add duplicate
    - Return original prompt unchanged

4. **Complex Scenes:**
    - "A group of dancers" with "dancer_3.png"
    - Try to identify the 3rd dancer if enumerated
    - Otherwise match the primary/first dancer

**Output Requirements:**
1. Return ONLY the modified prompt text
2. Do NOT add explanations (except for edge cases noted above)
3. Do NOT use quotes around the output
4. Preserve ALL existing markers unchanged
5. Only modify the matched character's description
6. Maintain natural grammar and readability

**Examples:**

Example 1:
Input Prompt: "A young woman with long brown hair in reference image 1 sitting on a vintage wooden chair in reference \
image 2, wearing a flowing white dress, smiling at the camera with hands gently placed on knees, Victorian room \
interior. The overall visual and artistic style of the generated image should resemble the style of reference image 3."
Pose Reference File: "a_young_woman.png"

Analysis:
- Match: "young woman" matches "a_young_woman.png"
- Highest reference: 3 (style)
- New pose reference: 4
- Keep: "sitting on a vintage wooden chair" (basic pose + scene interaction), "with long brown hair" (physical), \
"wearing a flowing white dress" (clothing)
- Remove: "smiling at the camera" (facial expression), "with hands gently placed on knees" (limb position detail)

Output: "A young woman with long brown hair in reference image 1 with pose from reference image 4 sitting on a vintage \
wooden chair in reference image 2, wearing a flowing white dress, Victorian room interior. The overall visual and \
artistic style of the generated image should resemble the style of reference image 3."

Example 2:
Input Prompt: "An astronaut in reference image 1 floating in zero gravity with arms spread wide and legs bent, inside \
a spaceship, using the scene of reference image 2 as background. The overall visual and artistic style of the \
generated image should resemble the style of reference image 3. The overall lighting effect of the generated image \
should resemble the lighting in reference image 4."
Pose Reference File: "astronaut_floating.png"

Analysis:
- Match: "astronaut" matches "astronaut_floating.png"
- Highest reference: 4 (lighting)
- New pose reference: 5
- Keep: "floating" (basic action), "in zero gravity" (scene context), "inside a spaceship" (location)
- Remove: "with arms spread wide and legs bent" (limb details)

Output: "An astronaut in reference image 1 with pose from reference image 5 floating in zero gravity, inside a \
spaceship, using the scene of reference image 2 as background. The overall visual and artistic style of the generated \
image should resemble the style of reference image 3. The overall lighting effect of the generated image should \
resemble the lighting in reference image 4."


"""



USER_TEMPLATE = """Process this prompt by adding a pose reference marker for the specified character:

**Original Prompt:**
{original_prompt}

**Pose Reference File:**
{pose_reference_filename}

**Task:**
1. Identify the character matching the pose reference filename
2. Find the highest reference number in the prompt
3. Insert "with pose from reference image Y" after the matched character's subject marker
    (Y = highest reference + 1)
4. Remove detailed pose descriptions for that character (limb positions, gestures, etc.)
5. Preserve essential pose info (standing/sitting/running, scene interactions)
6. Keep all other characters unchanged

**Output:** Modified prompt only (no explanations unless no match found)"""


def process_prompts(client: OpenAI, ori_prompt, pose_reference_filename) -> str:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0.9,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": SYSTEM_MSG},
                    {"role": "user",
                    "content": [
                            {"type": "text", "text": USER_TEMPLATE.format(
                            original_prompt=ori_prompt,
                            pose_reference_filename=pose_reference_filename
                            )}
                    ]},
                ],
                extra_body={"thinking_level": 'low'}
            )
            content = resp.choices[0].message.content
            if content is not None:
                return content.strip()
        except Exception as e:  # pylint: disable=broad-except
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return ori_prompt


def main():
    parser = argparse.ArgumentParser(description="Process prompts with pose references.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSONL file')
    parser.add_argument('--data-root', type=str, required=True, help='Root directory of image data')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, 'r', encoding='utf-8') as f:
        subject_bg_pairs = [json.loads(line) for line in f]

    for subject_bg_pair in tqdm(subject_bg_pairs):
        image_dir = '/'.join(subject_bg_pair['image'].split('/')[:-1])
        image_dir_path = data_root / image_dir
        pose_dir = image_dir_path / 'pose'
        if not pose_dir.exists():
            continue
        keypoints_dir = pose_dir / 'keypoints'
        original_prompt = subject_bg_pair['prompt']
        original_edit_images = subject_bg_pair['edit_image'].copy()
        for keypoints_path in keypoints_dir.iterdir():
            with open(keypoints_path, 'r', encoding='utf-8') as f:
                keypoints_file = json.load(f)
                if keypoints_file['num_persons'] > 1:
                    continue
                keypoints = keypoints_file['persons'][0]['keypoints']
                confidences = keypoints[2::3]
                confidences_vital = confidences[:15]
                if all(confidence > 0.5 for confidence in confidences_vital):
                    reference_path = pose_dir / 'skeleton' / keypoints_path.name.replace('.json', '.jpg')
                    if not reference_path.exists():
                        print(f'error: reference_path not exists: {reference_path}')
                        continue
                    try:
                        processed_prompt = process_prompts(client, original_prompt, os.path.basename(reference_path))
                        subject_bg_pair['prompt'] = processed_prompt
                        edit_images = original_edit_images.copy()
                        edit_images.append(f"{image_dir}/pose/skeleton/{keypoints_path.name.replace('.json', '.jpg')}")
                        subject_bg_pair['edit_image'] = edit_images
                        with open(output_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(subject_bg_pair, ensure_ascii=False) + '\n')
                    except Exception as e:  # pylint: disable=broad-except
                        print(e)

    print("Done")

if __name__ == "__main__":
    main()
