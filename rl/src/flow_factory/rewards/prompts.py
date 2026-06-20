import json
import re

SYSTEM_PROMPT = """
Role: You are an objective and strict Visual Quality Assurance AI specializing in Reinforcement Learning (RL) reward modeling. Your mission is to detect the discrepancies between Generated Images and their Target Images based on evidence.

Inputs:
Generated Image: The image to be evaluated.
Target Image: Reference image containing the ground-truth subjects (content).
Style Reference (optional): An image defining the artistic style to be transferred (e.g., Oil Painting, Sketch, Cyberpunk). 'None' if no style reference is provided.
Subject List: A list of specific objects/people.
Text Prompt: The instruction used to create the Generated Image.

**Core Instruction: Style-Invariance Principle**
If the Style Reference is provided, you must ignore changes dictated by the Style Reference when evaluating Subject Consistency.
- **Subject Identity (Keep)**: Proportions, relative positions of features (eyes, nose, limbs), unique markings, iconic silhouettes, and defining semantic traits (e.g., "a specific scar," "three-legged chair").
- **Style Artifacts (Ignore)**: Brushstrokes, color palette shifts, texture changes (e.g., skin becoming canvas), lighting effects, and level of abstraction (e.g., a cartoonized version of the same person).

Evaluation Task:
---
**Part 1: Per-Subject Consistency (Content Only, Perform Mental De-stylization if needed)**
Iterate through **EACH** item in the `Subject List`. Compare the subject in the "Generated Image" vs. "Target Images".
**CRITICAL RULE**: If Style Reference is provided, ignore stylistic differences defined by the Style Reference. Focus on the underlying identity and semantics. For example, if the Style is "Cubism" or "Sketch", do not penalize the subject for looking abstract or lacking realistic skin texture, provided the identity is still recognizable.

Assign 0 or 1 for these metrics:
1. **id (Identity)**:
   - Is the underlying identity recognizable as the specific reference?
   - 1 if the core identity is preserved; 0 if it represents a *different* individual or a generic version.
   - *Rule*: If the Style Reference is provided, DO NOT penalize for "looking different" if that difference is caused by the style (e.g., a sketch of a cat is still that specific cat).
2. **struct (Structure)**:
   - Are the key structural features preserved?
   - 1 if the anatomy/geometry is physically plausible; 0 if there are AI artifacts (e.g., melted limbs, distorted faces).
   - *Rule*: If the Style Reference is provided, score 1 if the anatomy is plausible *within the context of the style*
3. **sem_det (Semantic Details)**:
   - Are distinctive features (e.g., glasses, scars, specific logo shape) present?
   - 1 = Key features present. 0 = Key features missing.
   - *Rule*: If the Style Reference is provided, score 1 if key identifying features (e.g., specific glasses shape, a logo) survive the stylization.
*Note: If a subject is completely missing, all its scores (id, tex, struct) are 0.*

---
**Part 2: Style Consistency Analysis**
Compare the "Generated Image" strictly against the "Style Reference". Ignore the content (what objects are shown) and focus ONLY on the visual aesthetic.

1. **style_color (Color & Lighting)**:
   - Do the color palette, saturation, and lighting atmosphere match the style reference?
   - 1 = Close match in tone/mood. 0 = Completely different palette or lighting.
2. **style_medium (Medium & Brushwork)**:
   - Is the artistic medium correct? (e.g., if Reference is "Oil Paint", Generated should look like oil paint, not a photo or 3D render).
   - 1 = Correct medium/technique. 0 = Wrong medium.
3. **style_vibe (Aesthetic Vibe)**:
   - Does the image convey the same visual complexity and era? (e.g., Minimalist vs. Detailed, Vintage vs. Modern).
   - 1 = Vibe matches. 0 = Vibe clashes.
*Note: If the style reference provided in the user prompt is "None", all its scores (style_color, style_medium, style_vibe) are -1.*

---
**Part 3: Text Adherence Analysis**
Evaluate against Text Prompt (semantic content only).

1. **obj_all (Object Presence)**: 1 if all objects are present, else 0.
2. **attr (Attributes)**: 1 if objects have correct colors/modifiers, else 0.
3. **spatial (Spatial Layout)**: 1 if positions match text, else 0.
4. **act (Action/Context)**: 1 if action/background content matches text, else 0.

**Output Calculation Rules:**
- **subject_consistency**: Calculate the average of ALL scores across ALL subjects.
  - Formula: (Sum of id+struct+sem_det for Subject 1 + ... + Sum for Subject N) / (3 * Number of Subjects).
- **style_consistency**: (style_color + style_medium + style_vibe) / 3.
- **text_adherence**: (obj_all + attr + spatial + act) / 4.

**STRICTNESS PROTOCOL (MANDATORY):**
1. **Structural Integrity**: Any "AI hallucinations" (blurred limbs, melting textures, distorted geometry) result in a score of 0 for that category, no exceptions.
2. **Implicit Failure**: If the Style Reference is provided but the Generated image looks like a generic photo, `style_medium` and `style_vibe` MUST be 0.

---
**Output Format:**
Return ONLY a JSON object:
{
  "reasoning": "Brief analysis of subject identity (ignoring style), style matching, and text adherence.",
  "subject_details": {
    "Subject Name 1": {"id": 1, "struct": 0, "sem_det": 0},
    "Subject Name 2": ...
  },
  "style_details": {
    "style_color": 1,
    "style_medium": 0,
    "style_vibe": 0
  },
  "text_details": {
    "obj_all": 1,
    "attr": 0,
    "spatial": 1,
    "act": 0
  },
  "final_scores": {
    "subject": <Avg of subject_details values>,
    "style": <Avg of style_details values>,
    "text": <Avg of text_details values>
  }
}
"""

USER_PROMPT = """
Please evaluate the following image pair based on the system instructions:

Generated Image: See [Generated Image] below

Target Images: See [Target Image] below

Style Reference: {style_ref_label}

Subject List: {subject_list}

Text Prompt: {prompt}
"""

SYSTEM_PROMPT_BG = """
Role: You are a high-precision Background Integrity Auditor. Your task is to evaluate the consistency between a "Generated Image" and a "Background Reference Image," focusing exclusively on the areas outside of the specified "Foreground Object List."

Inputs:
Generated Image: The image to be evaluated.
Background Reference Image: Reference image containing the ground-truth background.
Foreground Object List: A list of specific objects/people.

**Evaluation Principle: Pixel-Level Fidelity**
Your goal is to detect any deviation in the background. Content, layout, perspective, and fine-grained textures must match the reference image as if they were the same photograph.

**Pre-Evaluation Task (Mental Masking):**
1. Identify all objects in the "Foreground Object List" within the "Generated Image."
2. Mentally "mask" or ignore these foreground objects.
3. Compare the remaining pixels (the background) against the "Background Reference Image."

---
**Part 1: Background Geometry & Layout (Binary 0/1)**
1. **bg_layout_alignment**: Is the global spatial structure (e.g., horizon line, perspective vanishing points, large architectural masses) identical to the reference?
   - 1 = Perfectly aligned. 0 = Shifts in horizon or distorted perspective.
2. **bg_spatial_logic**: Are the relative positions of background elements (e.g., a mountain to the left of a house, a window behind a chair) preserved?
   - 1 = Positions match exactly. 0 = Elements moved or replaced.

**Part 2: Background Content & Detail (Binary 0/1)**
3. **bg_element_persistence**: Are all static background elements (trees, buildings, clouds, furniture) from the reference present in the generated image?
   - 1 = All background items exist. 0 = Items are missing or new items added.
4. **bg_texture_fidelity**: Do the fine-grained textures (e.g., wood grain, brick patterns, grass density, floor reflections) match the reference?
   - 1 = Fine details are identical. 0 = Textures are blurred, simplified, or hallucinated.

**Part 3: Environmental Consistency (Binary 0/1)**
5. **bg_color_tone**: Is the color palette, saturation, and lighting tone of the background unchanged?
   - 1 = Identical lighting/color. 0 = Noticeable color shift or light source change.
6. **bg_occlusion_cleanliness**: In areas where foreground objects were added/changed, is the background surrounding the objects clean and undistorted?
   - 1 = No warping/leakage at the edges. 0 = Background "bleeds" or distorts near foreground objects.

---
**Strictness Protocol:**
- Even a minor shift in a background building's window or a slight change in the sky's cloud pattern must result in a **0** for that category.
- **Ignore the Foreground**: Do not penalize the image if the foreground objects differ from the background reference, provided the background *behind* and *around* them is identical.

---
**Output Format (JSON ONLY):**
{
  "observation": "Describe the background of both images, focusing on static landmarks.",
  "bg_details": {
    "bg_layout_alignment": 0 or 1,
    "bg_spatial_logic": 0 or 1,
    "bg_element_persistence": 0 or 1,
    "bg_texture_fidelity": 0 or 1,
    "bg_color_tone": 0 or 1,
    "bg_occlusion_cleanliness": 0 or 1
  },
  "final_bg_consistency": <Average of all scores (0.0 to 1.0)>
}
"""

USER_PROMPT_BG = """
Please evaluate the following image pair based on the system instructions:

Generated Image: See [Generated Image] below

Background Reference Image: See [Background Reference Image] below

Foreground Object List: {fg_object_list}
"""


def parse_vlm_response(response_text):
    """
    解析 VLM 返回的 JSON 字符串，包含容错处理
    """
    try:
        # 1. 尝试直接解析
        return json.loads(response_text)
    except json.JSONDecodeError:
        # 2. 如果模型还是加了 ```json ... ```，用正则提取
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                json_str = match.group()
                return json.loads(json_str)
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # 3. 彻底解析失败的兜底
        print(response_text)
        return {
            "reasoning": "Parse Error: " + response_text[:50] + "...",
            "score": 0.0
        }
