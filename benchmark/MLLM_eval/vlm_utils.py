import base64
import json
import re
import os
import io
from pathlib import Path
from io import BytesIO
from typing import Union
from PIL import Image

mime_types = {
    ".aac": "audio/aac",
    ".avif": "image/avif",
    ".json": "application/json",
    ".apng": "image/apng",
    ".avi": "video/x-msvideo",
    ".bmp": "image/bmp",
    ".flv": "video/x-flv",
    ".gif": "image/gif",
    ".jpg": "image/jpg",
    ".jpeg": "image/jpeg",
    ".mp3": "audio/mp3",
    ".m4a": "audio/x-m4a",
    ".m4v": "video/x-m4v",
    ".mng": "video/x-mng",
    ".mov": "video/quicktime",
    ".mpeg": "video/mpeg",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".wmv": "video/x-ms-wmv",
    ".weba": "audio/webm",
    ".webm": "video/webm",
    ".webp": "image/webp"
}

system_prompt_for_all = """
You are an expert AI Art Director and Visual Quality Assurance Specialist. Your task is to evaluate a "Generated \
Image" against a "Text Prompt" and a set of "Reference Images" (Control Signals).

You must assess the consistency of the generated image across 6 dimensions:
1. **Text Adherence**: How well does the image match the text description?
2. **Subject Consistency**: Identity preservation (ignoring style).
3. **Style Consistency**: Art style, medium, and texture match.
4. **Lighting Consistency**: Illumination direction, color, and mood.
5. **Background Consistency**: Environment and setting match.
6. **Aesthetic Quality**: Overall visual appeal and technical quality.

**General Rules:**
- **Conservative Scoring**: Adopt a strict, critical mindset. Do not inflate scores.
10 (Perfection): Reserved strictly for flawless execution with zero artifacts, hallucinations, or deviations.
8 (Excellent): High-quality professional work with only negligible flaws visible upon close inspection.
6 (Passable): The standard for "average" or "okay" generations.
< 5 (Flawed): Use these scores freely for any noticeable errors, distortions, or missed instructions.

- **Missing References**: If a specific reference category is provided as an empty list `[]`, `None`, or a \
placeholder, you must set the score for that category to **-1** and the reasoning to "N/A".
- **Independence**: Evaluate each dimension independently. For example, a perfect Subject match can exist even if the \
Style is completely different (if that was the intent).
- **Output Format**: You must output strictly valid JSON without Markdown formatting.
- **Score Range**: Each score must be a float between 0 and 10.
- **Reasoning**: Provide a brief reasoning for each score.
"""

user_prompt_for_all = """
# Input Data
Text Prompt: "{text_prompt}"

Reference Images:
- Subject References: {subject_ref_label}
- Style Reference: {style_ref_label}
- Lighting Reference: {lighting_ref_label}
- Background Reference: {background_ref_label}

Target Generated Image: {generated_img_label}

---

# Evaluation Task
Analyze the "Target Generated Image" against the inputs above. Follow these specific rules for each dimension:

## 1. Text Adherence
Goal: Verify if the generated image strictly follows the instructions in the "Text Prompt", including the implicit or \
explicit instruction to utilize provided Reference Images.

Focus:
Entities & Attributes: Are mentioned objects present with correct colors/shapes?
Actions & Relations: Are interactions and spatial positions correct?
Reference Integration (Crucial): If the prompt implies using a reference (e.g., "in this style," "replace background," \
"this person"), does the image respect those constraints?
**Critical Rules**:
1. A failure to match a Reference Image (Style, Subject, Background, Lighting) should result in a score lower than 6.


Scoring (0-10 Scale, **Conservative Scoring**: Adopt a strict, critical mindset. Do not inflate scores.):

0: Completely Irrelevant. The image bears no relationship to the text prompt OR completely ignores the core reference \
instruction.
Example: Prompt "A dog in the style of reference image 1". Image is "A photo of a car".

2: Severe Hallucination / Major Contradiction. The main subject is present, but the mode of generation is wrong.
Reference Failure: Prompt asks for "Style Transfer" (using Ref A), but the image is Photorealistic. The AI ignored the \
"Style" instruction entirely.
Content Failure: Prompt asks for "A red car," image shows "A blue car."

4: Partial Adherence / Conflict. The AI captures the "Noun" (Subject) but fails the "Adjective" (Style/Attribute) or \
"Verb" (Action).
Criteria: The image shows the correct subject content, but fails to apply the requested Control Signal effectively.
Example: Prompt "A warrior in image 1 in a forest in image 2." The warrior is generic (Subject Ref ignored), even if \
the forest is correct. The instruction to "use this specific warrior" was not followed.

6: General Adherence (Minor Misses). The "Gist" is correct. The subject, action, and setting match the text \
description. However, the integration of references is weak or inconsistent.
Criteria: The prompt is followed, but the fidelity to the requested references is relatively low.

8: High Precision. All major text instructions are met. The reference images are clearly utilized and recognizable.
Criteria: The image correctly combines the Text Description with the Reference Constraints.
Deviation: Only trivial details (e.g., a minor color shift or a small background element) are slightly off.

10: Flawless Execution. Perfect alignment of Text and References.
Criteria: Every instruction is executed perfectly. The image contains exactly what was asked for (counts, colors, \
actions) AND perfectly integrates the requested Subject/Style/Background references.
Logic: You cannot get a 10 in Text Adherence if the Style or Subject Consistency scores are low, because that means \
the instruction to "use that style/subject" was failed.

## 2. Subject Consistency
Goal: Determine if the primary subject (person, animal, object, or product) in the Target matches the "Subject \
References".
CRITICAL RULE: You must IGNORE the artistic style, rendering technique (2D vs 3D), lighting, and pose.
Focus:
Humans/Animals: Facial features, skull/muzzle structure, fur patterns, and distinctive markings (scars, moles).
Objects/Products: Geometric shape, logo placement, specific design details (e.g., button layout, stitching), and \
material breaks.

Scoring (0-10 Scale, **Conservative Scoring**: Adopt a strict, critical mindset. Do not inflate scores.):
0: Wrong Subject or Complete absence. The subject is completely different (e.g., wrong species, wrong object category, \
or unrecognizable) or entirely missing.
Example: Reference is a Cat; Target is a Dog. Reference is a Sports Car; Target is a Truck.

2: Generic Class Match. The Target shares the broad category but lacks specific identity. It looks like a "stock \
photo" version rather than the specific reference.
Human: Correct gender/hair color, but face is generic.
Object: It is a "red sneaker," but the design lines and shape do not match the reference brand/model at all.

4: "Knock-off" / Distorted Resemblance. The subject attempts to look like the reference but fails on structural \
accuracy. It resembles a "bad drawing" or a "counterfeit product."
Human: Features are present but distorted (e.g., eyes too far apart, nose shape changed).
Object: The logo is misspelled or warped; the proportions of the product are noticeably wrong (e.g., phone is too wide).

6: Recognizable Identity. Clearly the same subject. The identity is established, but there are noticeable minor \
deviations or simplifications.
Human: Instantly recognizable as the person, though some fine details (like exact eye crease) might be smoothed out.
Object: The car is clearly a Porsche 911, but the rim design or minor grill details are slightly inaccurate.

8: High Fidelity. Strong structural integrity. Distinctive features are preserved with high accuracy. Only close \
inspection reveals very minor discrepancies.
Human: Specific moles, ear shape, facial features, and exact jawline are preserved.
Object: Textures, labels, and complex mechanical details (e.g., camera lens rings) are accurate to the reference.

10: Perfect Structural Clone. The subject's geometry and defining features are preserved perfectly, even if the art \
style is completely different.
Criteria: Every scar, whisker pattern, screw placement, or logo alignment matches the reference logic perfectly. It \
looks like the same 3D model or physical object was used.

## 3. Style Consistency
Goal: Determine if the Target adopts the visual style of the "Style Reference".
Focus: Art medium (e.g., oil painting, anime, polaroid), brushwork, color palette, texture, lighting atmosphere, and \
line weight.
Ignore: The actual content (who or what is in the image).

Scoring (0-10 Scale, **Conservative Scoring**: Adopt a strict, critical mindset. Do not inflate scores.):
-1 (special case): Reference missing / Not Applicable.  

0: Complete Style Mismatch. The medium is fundamentally different. There is no attempt to match the reference.
Example: Reference is a black-and-white sketch; Target is a hyper-realistic color photograph.

2: Superficial / Vague Resemblance. Matches the broad "mood" or color palette, but the medium is incorrect or \
unconvincing. It feels like a cheap filter was applied.
Example: Reference is a thick impasto oil painting; Target is a flat digital illustration that just uses similar \
colors but lacks the 3D texture of the paint.

4: Generic Category Match. Correct medium (e.g., both are "Anime" or both are "Polaroids"), but fails to capture the \
specific nuances of the reference. It looks like a "stock" version of that style.
Example: Reference is a specific 1990s anime style (grainy, muted colors); Target is a modern, high-gloss 4K anime \
style. The genre is right, but the era/technique is wrong.

6: Consistent Vibe & Palette. Accurately captures the medium and the dominant color grading. The image clearly belongs \
to the same "family" as the reference, though specific details (like line weight consistency or background rendering) \
may vary slightly.
Example: The lighting and colors match the reference photography perfectly, but the film grain texture is slightly \
different.

8: Deep Stylistic Match. Captures complex stylistic signatures, including specific brushwork patterns, lighting \
behavior, or material rendering.
Example: If the reference is a watercolor, the target replicates the specific "wet-on-wet" bleed effects and paper \
texture accurately. It feels like a cohesive piece of art.

10: Perfect Artist/Medium Mimicry. The Target looks like it was created by the exact same artist or the exact same \
camera/lens setup.
Criteria: Texture, noise, stroke dynamics, and color science are identical. You could place the Target alongside the \
Reference in a portfolio, and no one would suspect they were generated differently.

## 4. Lighting Consistency
Goal: Determine if the Target creatively adapts the lighting scheme of the "Lighting Reference" to the new scene \
defined in the text prompt.
Focus: Light direction, shadow hardness/softness, contrast, and color grading (warm/cool atmosphere).

CRITICAL DISTINCTION:
Good: The lighting logic is applied to a new 3D geometry.
Bad (Leakage): The Target copies the background pixels from the reference, ignoring the text prompt's requested \
environment.

Scoring (0-10 Scale, **Conservative Scoring**: Adopt a strict, critical mindset. Do not inflate scores.):
-1 (special case): Reference missing / Not Applicable.  

0: Lighting Mismatch. The lighting environment is completely contradictory to the reference.
Example: Reference is "Cyberpunk Neon Night" (dark, colorful); Target is "Natural Sunlight" (bright, white).

2: Background Leakage / Flat Overlay. Major Failure. The model fails to generate the new scene requested in the prompt \
and instead copies the background/layout of the Lighting Reference.
OR: The lighting looks like a flat 2D color filter applied over the image, lacking depth or interaction with the \
subject's shape.

4: Atmosphere Only (Vague). Captures the general "mood" or dominant color (e.g., "it is generally warm/orange"), but \
misses the physics of the light.
Failure: Light direction is wrong (e.g., reference is side-lit, target is front-lit), or shadow hardness doesn't match \
(e.g., reference has sharp shadows, target has soft ambient light).

6: Correct Direction & Tone. The primary light source comes from the correct direction, and the color grading matches. \
The subject feels reasonably integrated.
Limitation: Complex effects like rim lighting, volumetric fog, or secondary bounce lights are missing or simplified.

8: High Fidelity Transfer. Accurately replicates the specific lighting setup: key light, fill light, and back light \
are all present and correct.
Success: Shadows fall naturally on the new subject. The contrast ratio (difference between light and dark areas) \
matches the reference well.

10: Perfect Physical Integration. The complex lighting scheme is perfectly mapped onto the new geometry defined by the \
text prompt.
Criteria: Even difficult lighting interactions (e.g., subsurface scattering, caustics, or multi-colored light sources) \
are preserved and physically accurate. The image looks like a 3D render using the exact same "lighting rig" as the \
reference, with zero background leakage.

## 5. Background Consistency
Goal: Determine if the Target preserves the structural integrity and layout of the "Background Reference".
Focus: Spatial layout, object placement, perspective, and depth.

Context Awareness (The "Style Exception"):
If No Style Reference: Expect pixel-perfect preservation (Inpainting standard). 
If Style Reference Exists: Pixel-level redrawing is ALLOWED (changing textures/colors to match style), but Scene \
Layout must be RIGID. The "underlying geometry" must not change.

Scoring (0-10 Scale, **Conservative Scoring**: Adopt a strict, critical mindset. Do not inflate scores.):
-1: Reference missing / Not Applicable.

0: Complete Irrelevance. The background is totally unrelated to the reference.
Example: Reference is "Snowy Mountain"; Target is "Kitchen".

2: Severe Distortion / Hallucination / Paste effect. Attempts to use the reference, but the geometry collapses. \
Perspective is warped, or the background blends horribly with the subject. The location is barely recognizable due to \
structural errors.
Image that looks like multiple pictures simply pasted together should also receive a score of 2.

4: Thematic Match / "Vibe" Only. The "Trap" Score. The AI generated a background of the same category, but ignored the \
specific layout.
Criteria: Reference is a specific bedroom with a window on the left. Target is a generic bedroom with a window on the \
right. The furniture arrangement is completely different.

6: Structural Drift / Object Hallucination. The general perspective (horizon line, room shape) is correct, but \
specific objects have changed identity or position.
Criteria: The layout is 80%% correct, but a "bookshelf" in the reference has become a "wardrobe" in the target. Or, \
the pattern on the floor has changed from "tiles" to "wood" (unless requested by style). It looks like a "remake" of \
the scene rather than the same scene.

8: Structural Lock (High Fidelity). The scene layout is identical. Every major object is in the correct position.
Style Context: If a style is applied (e.g., "Oil Painting"), the texture has changed, but the shapes are correct. A \
tree is still that specific tree, just painted.
Deviation: Minor blurring of very small background details (e.g., text on a distant sign is unreadable), but the \
structure is solid.

10: Perfect Structural Integrity. Even if the style has changed the pixels (colors/textures), the geometry aligns \
perfectly.
Criteria: If you were to mask out the subject in Photoshop and layer the reference image underneath, they would align \
perfectly.
Visual Check: No objects are added, removed, or moved. It looks like the exact same photo shoot location, either \
preserved perfectly (photorealism) or strictly re-rendered in the new style (stylization).

## 6. Aesthetic Quality
Goal: Evaluate the overall visual appeal, technical quality, and artistic composition of the Target.
Focus: Sharpness, lack of artifacts (distorted faces, extra fingers), composition, and color harmony.
Context Awareness: Evaluate aesthetics within the intended style (e.g., grain is acceptable in "Vintage Film", but not \
in "Clean 3D Render").

Scoring (0-10 Scale, **Conservative Scoring**: Adopt a strict, critical mindset. Do not inflate scores.):
0: Unusable / Severe Corruption. The image is incoherent, blurry beyond recognition, or dominated by noise. It looks \
like a technical error rather than an image.

2: Major Artifacts / Broken Anatomy. The subject is recognizable, but the image is ruined by severe AI hallucinations.
Criteria: Melted faces, extra/missing limbs, nonsensical hands (6+ fingers), or floating disconnected objects. The \
texture quality is muddy or pixelated.

4: Generic / Obvious "AI Look". Technically "correct" (no severe breakage), but visually unappealing or boring.
Characteristics: Flat lighting, poor composition (subject centered awkwardly), or that specific "waxy/plastic" skin \
texture common in low-quality models. It lacks artistic intent.

6: Clean & Competent. A solid, usable image. The resolution is sharp, and anatomy is generally correct.
Level: Good enough for a casual social media post. The composition follows basic rules (e.g., rule of thirds), but it \
doesn't stand out as particularly artistic or striking.

8: High-End Professional. Excellent technical execution. Textures are detailed and realistic (skin pores, fabric weave).
Artistry: Strong color harmony and lighting. It looks like a high-quality stock photograph or a polished digital \
painting. No visible AI artifacts.

10: Artistic Masterpiece. Exceptional visual impact. The image goes beyond "correct" to be emotionally or visually \
stunning.
Criteria: Cinematic lighting, dynamic composition, and perfect attention to detail. Indistinguishable from \
award-winning human photography or top-tier concept art.

---

# Output Format
Return a single JSON object. Do NOT use Markdown code blocks.

JSON Structure:
{{
  "text_adherence": {{
    "score": <float>,
    "reasoning": "<string>"
  }},
  "subject_consistency": {{
    "score": <float>,
    "reasoning": "<string>"
  }},
  "style_consistency": {{
    "score": <float>,
    "reasoning": "<string>"
  }},
  "lighting_consistency": {{
    "score": <float>,
    "reasoning": "<string>"
  }},
  "background_consistency": {{
    "score": <float>,
    "reasoning": "<string>"
  }},
  "aesthetic_quality": {{
    "score": <float>,
    "reasoning": "<string>"
  }},
  "overall_average_score": <float> (Average of non-negative scores)
}}
"""


def pil_to_base64(image: Image.Image, fmt='JPEG') -> str:
    """
    将 PIL Image 对象转换为 Base64 字符串
    :param image: PIL Image 对象
    :param fmt: 保存格式，如 'JPEG', 'PNG', 'WEBP'。JPEG 体积小，PNG 无损。
    :return: Base64 编码的字符串
    """
    output_buffer = BytesIO()

    # 注意：如果是 RGBA 模式（透明底）存为 JPEG 会报错，需要转为 RGB
    if fmt == 'JPEG' and image.mode == 'RGBA':
        image = image.convert('RGB')

    # 将图片保存到内存缓冲中
    image.save(output_buffer, format=fmt)

    # 获取二进制数据并进行 base64 编码
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')

    return base64_str


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
            except:
                pass

        # 3. 彻底解析失败的兜底
        print(response_text)
        return {
            "reasoning": "Parse Error: " + response_text[:50] + "...", 
            "score": 0.0
        }


def encode_image(image_path: Union[str, Path]) -> str:
    """将图片编码为base64"""
    if isinstance(image_path, str):
        image_mime_type = mime_types.get(image_path.split(".")[-1], "image/jpeg")
    else:
        image_mime_type = mime_types.get(image_path.suffix, "image/jpeg")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8'), image_mime_type


def encode_image_to_base64(image_path, max_size=512, quality=80):
    """
    读取图片，缩放，转为 JPEG 格式的 Base64 字符串。
    """
    try:
        with Image.open(image_path) as img:
            # 1. 转换颜色模式：如果是 PNG (RGBA)，转为 RGB，否则 JPEG 不支持透明通道
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            # 2. 缩放图片：限制长边最大为 max_size (例如 1024 或 512)
            # VLM 不需要 4K 分辨率也能看清内容，过大只会浪费 Token 和带宽
            if max(img.size) > max_size:
                img.thumbnail((max_size, max_size))

            # 3. 保存为 Bytes 流 (JPEG 格式)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=quality)

            # 4. 转 Base64
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 5. 拼接完整 Data URL
            return f"data:image/jpeg;base64,{img_str}"

    except Exception as e:  # pylint: disable=broad-except
        print(f"Error processing image {image_path}: {e}")
        return None


def call_vlm_for_all(gen_image_path, test_set_base_dir, item, client):
    """
    调用 VLM 进行一致性评估
    :return: 解析后的 JSON 结果
    """
    category = item['category']
    subject_ref_label = "[See subject reference images below]"
    if 'bg' in category:
        bg_ref_label = "[See background reference image below]"
    else:
        bg_ref_label = "None"
    if 'lighting' in category:
        lighting_ref_label = "[See lighting reference image below]"
    else:
        lighting_ref_label = 'None'
    if 'style' in category:
        style_ref_label = "[See style reference image below]"
    else:
        style_ref_label = 'None'

    generated_img_label = "[See generated image below]"

    user_prompt = user_prompt_for_all.format(
        text_prompt=item['prompt'],
        subject_ref_label=subject_ref_label,
        background_ref_label=bg_ref_label,
        lighting_ref_label=lighting_ref_label,
        style_ref_label=style_ref_label,
        generated_img_label=generated_img_label
    )

    gen_image_base64 = encode_image_to_base64(gen_image_path)
    subject_ref_base64_list = []
    style_ref_base64 = None
    lighting_ref_base64 = None
    bg_ref_base64 = None
    for edit_image in item['edit_image']:
        if 'transfer_subjects' in edit_image:
            subject_ref_base64_list.append(encode_image_to_base64(os.path.join(test_set_base_dir, edit_image)))
        elif 'style/reference' in edit_image:
            style_ref_base64 = encode_image_to_base64(os.path.join(test_set_base_dir, edit_image))
        elif 'lighting/lighting_reference' in edit_image:
            lighting_ref_base64 = encode_image_to_base64(os.path.join(test_set_base_dir, edit_image))
        elif 'background' in edit_image:
            bg_ref_base64 = encode_image_to_base64(os.path.join(test_set_base_dir, edit_image))
    user_content = [{"type": "text", "text": user_prompt}]
    user_content.append({"type": "text", "text": 'Subject reference images:'})
    for subject_ref_base64 in subject_ref_base64_list:
        # base64_str, mime_type = subject_ref_base64
        user_content.append({"type": "image_url", "image_url": {"url": subject_ref_base64}})
    if style_ref_base64 is not None:
        user_content.append({"type": "text", "text": 'Style reference image:'})
        user_content.append({"type": "image_url", "image_url": {"url": style_ref_base64}})
    if lighting_ref_base64 is not None:
        user_content.append({"type": "text", "text": 'Lighting reference image:'})
        user_content.append({"type": "image_url", "image_url": {"url": lighting_ref_base64}})
    if bg_ref_base64 is not None:
        user_content.append({"type": "text", "text": 'Background reference image:'})
        user_content.append({"type": "image_url", "image_url": {"url": bg_ref_base64}})
    user_content.append({"type": "text", "text": 'Generated image:'})
    user_content.append({"type": "image_url", "image_url": {"url": gen_image_base64}})

    messages = [
        {"role": "system", "content": system_prompt_for_all},
        {
            "role": "user", 
            "content": user_content
        }
    ]

    # 调用模型
    response = client.chat.completions.create(
        model="gemini-3-flash",
        messages=messages,
        max_tokens=4096,
        temperature=0.9,
        extra_body={"thinking_level": 'low'}
        )

    # 解析结果
    if response.choices[0].message.content is None:
        max_retries = 3
        for _ in range(max_retries):
            print("Retrying...")
            response = client.chat.completions.create(
                model="gemini-3-flash",
                messages=messages,
                max_tokens=4096,
                temperature=0.9,
                extra_body={"thinking_level": 'low'}
                )
            if response.choices[0].message.content is not None:
                break
        if response.choices[0].message.content is None:
            return {
                "reasoning": "Parse Error: " + response.choices[0].message.content[:50] + "...", 
                "score": 0.0
            }
    return parse_vlm_response(response.choices[0].message.content.strip())
