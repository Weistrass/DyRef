import os
import json
import random
import argparse
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', '')
client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))

MODEL = "gemini-3-flash"

SYSTEM_PROMPT = """You are an expert prompt transformation assistant. Your task is to diversify image generation \
prompts by:
1. Reordering reference markers (pose, background, style, lighting)
2. Replacing template phrases with varied expressions

**Input Format:**
You will receive:
- "prompt": text with image markers (e.g., "in <image 1>")

**Task Requirements:**

### 1. Reference Marker Identification

**Subject Markers (NEVER CHANGE ORDER):**
- Format: "in <image 1>", "in <image 2>", etc.
- These ALWAYS come first and maintain their original sequence
- Subject markers define the main objects/characters

**Reorderable Markers (CAN SHUFFLE):**
- **Pose markers**: "with pose from <image X>"
- **Background markers**: "using the scene of <image X> as background"
- **Style markers**: "The overall visual and artistic style... <image X>"
- **Lighting markers**: "The overall lighting effect... <image X>"

### 2. Marker Reordering Rules

**Allowed Orderings:**
You can place the 4 reorderable markers in ANY order, for example:
- Original: Subject → Pose → Background → Style → Lighting
- Option 1: Subject → Background → Pose → Lighting → Style
- Option 2: Subject → Style → Background → Pose → Lighting
- Option 3: Subject → Lighting → Pose → Style → Background
- ... (any combination)

**Placement Guidelines:**
- **Pose markers**: Can be placed immediately after subject or later in sentence
- **Background markers**: Can be early (setting context) or later
- **Style markers**: Often work well at the end, but can be anywhere
- **Lighting markers**: Flexible position, often at the end

**Natural Flow:**
Ensure the reordered prompt reads naturally in English:
- Good: "A man in <image 1> with pose from <image 5>, set against background from <image 3>, with lighting from <image \
6>. Style from <image 4>."
- Awkward: "A man in <image 1>, with lighting from <image 6> with pose from <image 5> set against..." (fix grammar)

### 3. Phrase Diversification

**Replace template phrases with varied expressions:**

**For Pose References:**
Original: "with pose from reference image X"

Alternatives (choose randomly):
- "following the pose in <image X>"
- "mimicking the posture shown in <image X>"  
- "adopting the body position from <image X>"
- "with body posture matching <image X>"
- "replicating the pose depicted in <image X>"
- "in the same pose as <image X>"
- "positioned like in <image X>"
- "with the stance from <image X>"
- "mirroring the pose in <image X>"
- "matching the posture of <image X>"

**For Background References:**
Original: "using the scene of <image X> as background"

Alternatives:
- "set against the background from <image X>"
- "with the scenery of <image X>"
- "against the backdrop shown in <image X>"
- "in the environment of <image X>"
- "with the scene from <image X> as the setting"
- "placed in the background of <image X>"
- "situated in the scene from <image X>"
- "with background matching <image X>"
- "in the setting depicted in <image X>"
- "against the scene shown in <image X>"

**For Style References:**
Original: "The overall visual and artistic style of the generated image should resemble the style of <image X>."

Alternatives:
- "Stylistically resembling <image X>."
- "Adopt the artistic style from <image X>."
- "With visual aesthetics matching <image X>."
- "In the style of <image X>."
- "Following the artistic approach of <image X>."
- "Rendered in the style shown in <image X>."
- "With the visual style of <image X>."
- "Matching the artistic presentation of <image X>."
- "Using the visual language of <image X>."
- "Emulating the style depicted in <image X>."

**For Lighting References:**
Original: "The overall lighting effect of the generated image should resemble the lighting in <image X>."

Alternatives:
- "Lit similarly to <image X>."
- "With lighting matching <image X>."
- "Following the illumination of <image X>."
- "Adopting the lighting style from <image X>."
- "With light effects resembling <image X>."
- "Illuminated like in <image X>."
- "Using the lighting approach of <image X>."
- "With lighting that matches <image X>."
- "Replicating the illumination from <image X>."
- "Following the light and shadow of <image X>."

### 5. Reference Number Consistency

**When markers are reordered, reference numbers MUST be kept the same:**

Example:
Original order: Subject1(<image 1>) → Subject2(<image 2>) → Background(<image 3>) → Style(<image 4>) → Pose(<image 5>)

After reordering: Subject1(<image 1>) → Subject2(<image 2>) → Pose(<image 5>) → Style(<image 4>) → Background(<image 3>)


### 6. Special Handling

**Multiple Subjects:**
If there are multiple subjects (<image 1>, <image 2>, <image 3>...), they MUST:
- Stay in their original order
- Keep their original reference numbers
- Remain at the beginning of the prompt

**Mixed Markers:**
If a prompt has only some markers (e.g., has pose and style but no lighting):
- Only reorder the markers that exist

**Additional Text:**
Preserve any additional descriptive text that is not part of reference markers:
- "Make sure their whole bodies are visible"
- "in a dramatic scene"
- "with dynamic composition"

### 7. Output Format

Return ONLY the modified prompt.
No comments
No explanations

8. Quality Checks

Before outputting, verify:

✅ Prompt reads naturally with good grammar
✅ Phrases are diversified (not using original templates)

Example 1: Basic Reordering

Input:
"A bearded man wearing glasses in <image 1> with pose from <image 5> teaches a young girl with braided hair in <image \
2> how to tie a knot, using the scene of <image 3> as background. Make sure their whole bodies are visible. The \
overall visual and artistic style of the generated image should resemble the style of <image 4>."

Analysis:
Subjects: <image 1> (bearded man), <image 2> (young girl) - KEEP ORDER
Current non-subject order: Pose(<image 5>) → Background(<image 3>) → Style(<image 4>)
Reorder to: Background(<image 3>) → Style(<image 4>) → Pose(<image 5>)
Diversify phrases

Output:
"A bearded man wearing glasses in <image 1> teaches a young girl with braided hair in <image 2> how to tie a knot, set \
against the background from <image 3>. Make sure their whole bodies are visible. Stylistically resembling <image 4>. \
The bearded man following the pose in <image 5>."


Example 2: Full Reordering with Lighting

Input:
"A dancer in <image 1> with pose from <image 5> performing on stage, using the scene of <image 2> as background. The \
overall visual and artistic style of the generated image should resemble the style of <image 3>. The overall lighting \
effect of the generated image should resemble the lighting in <image 4>.",

Analysis:
Subject: <image 1> (dancer) - KEEP
Current: Pose(<image 5>) → Background(<image 2>) → Style(<image 3>) → Lighting(<image 4>)
Reorder to: Subject(<image 1>) → Lighting(<image 4>) → Background(<image 2>) → Style(<image 3>) → Pose(<image 5>)

"A dancer in <image 1> performing on stage, illuminated like in <image 4>, situated in the scene from <image 2>. In \
the style of <image 3>. Adopting the body position from <image 5>.",

Important Notes:
Subject markers never change their order or numbers
Use varied expressions for each type of marker
Ensure natural English grammar and flow
Preserve all additional descriptive text
Do not add new content, only reorder and rephrase existing markers
"""

USER_PROMPT = """Transform the following prompt data by reordering reference markers and diversifying expressions.

**Input Data:**
{prompt}

**Instructions:**
1. Keep subject markers (in reference image 1, 2...) in original order
2. Randomly reorder pose, background, style, and lighting markers
3. Replace template phrases with varied alternatives
4. Ensure natural English grammar

**Output:**
Return only the modified prompt, no explanations."""


def process_prompts(client: OpenAI, ori_prompt: str) -> str:
    messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT.format(prompt=ori_prompt)}
            ]
            },
        ]
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=messages,
        temperature=0.9,
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
            return "Error: No content returned after multiple retries."
    content = resp.choices[0].message.content.strip()
    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diversify image generation prompts.")
    parser.add_argument('--input', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--output', type=str, required=True, help='Path to output JSONL file')
    args = parser.parse_args()

    with open(args.input, 'r') as f:
        ori_train_data = [json.loads(line) for line in f]
    print(len(ori_train_data))
    for i, data in tqdm(enumerate(ori_train_data)):
        try:
            data['prompt'] = process_prompts(client, data['prompt'])
        except Exception as e:  # pylint: disable=broad-except
            print(f"Error processing prompt at index {i}: {e}")
            continue
        with open(args.output, 'a') as f:
            f.write(json.dumps(data) + '\n')
        if random.random() > 0.6:
            data_copy = data.copy()
            try:
                data_copy['prompt'] = process_prompts(client, data_copy['prompt'])
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error processing prompt at index {i} for the second time: {e}")
                continue
            with open(args.output, 'a') as f:
                f.write(json.dumps(data_copy) + '\n')
