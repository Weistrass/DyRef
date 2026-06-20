import base64
from io import BytesIO
from typing import Union
from PIL import Image
import json
import re
from pathlib import Path
import os
import io

system_prompt_for_sf = """
Role
You are an expert AI assistant specializing in the objective evaluation of the consistency of objects in two \
images.You must determine if the specific subject (person, object, or animal) in the Reference Image is faithfully \
preserved in the Target Image.

Input Format
You will receive two images and the name of a subject. You need to observe the two pictures and determine whether the \
subject in the first picture (reference) is in the second picture (target).

Task Rule Description
**1. Allowed Changes (Do NOT penalize):**
* Viewpoint / Camera angle.
* Background / Environment / Context.
* Lighting / Color tone / Style.
* Pose / Action / Expression (for humans/animals).
* Scale / Position.
* Additional **NON-IDENTITY accessories** appearing only in the Target (e.g., extra hat/glasses), UNLESS they cover or \
alter identity-critical cues.
* **Style Invariance Rule:** If one image is a stylized result (e.g., a cartoon version of a real photo), you must \
**disregard the change in texture, shading, and dimension**. Focus ONLY on whether the **shapes, proportions, and \
specific design elements** (e.g., specific hair bang shape, specific collar design) are preserved.

**2. Forbidden Changes (Identity must be preserved):**
* Loss of **Core Identity Features**: unique shape/structure, distinctive patterns, logos, texts, unique marks \
(moles/scars), and subject-specific design details.
* **Instance Replacement**: Replacing the subject with a different instance of the same category (e.g., a different \
dog breed, a different person) is a FAIL.

**3. Core Identity Checklist (Use as guidance):**
* **Visibility:** Sometimes the subject is partially occluded in the reference image. Ensure the subject is completely \
visible without excessive occlusion in the target image.
* **Humans:** Facial structure (eyes/nose/mouth proportions), hairstyle outline, distinctive marks.
* **Animals:** Fur pattern distribution, face/ear shape, distinctive markings, tail shape.
* **Objects/Products:** Overall structure, key components, unique design elements, logos/text, unique textures/patterns.
"""

user_prompt_for_sf = """
# Input Data
subject name: {subject}
reference image: see reference image below.
target image: see target image below.

Compare the given subject in these two images and evaluate their consistency, specifically focusing on identity \
preservation across potential style changes.

**Step 1: Subject Identification**

* Identify the main subject in the Reference Image.
* Locate the corresponding subject in the Target Image.

**Step 2: Part Breakdown**

* Mentally break the Reference subject into **5–10 identity-relevant PARTS**.
* **MUST Include:** Global shape/silhouette + At least 2 identity-critical parts (face/head for living beings; \
logo/text for objects).
* **Text Rule:** If the subject contains TEXT/LOGO, treat EACH text region as a separate CRITICAL part.

**Step 3: Extreme Detail Comparison**

* For EACH part, compare Reference vs. Target strictly.
* **Quantitative Check:** Look for counts, shapes, relative positions, and specific text strings.
* **Ask yourself:** "Is this the same person/object rendered differently, or are the physical features actually \
different?"
* Synthesize a brief reasoning explaining what features match and what features differ.
* **Ignore:** Background, pose, lighting, artistic style (e.g., photorealistic vs. anime, sketch, or 3D render) or \
non-identity accessories.

**Step 4: Final Scoring**
Score the overall consistency from 0-4 based on the average of key parts:

- 0: No resemblance / Subject missing
- 1: Minimal resemblance (Major structural differences) or severe asthetic defects in the target image.
- 2: Moderate resemblance (Some features match, but key details are lost in translation)
- 3: Strong resemblance (High consistency in features despite style differences)
- 4: Structurally Identical (Perfect identity preservation, even if the art style is completely different)

Output Format:
Return a single valid JSON object containing exactly two keys: "reasoning" and "score".

- "reasoning": A concise string (1-2 sentences) summarizing the key similarities and differences, explicitly \
mentioning if style differences were ignored.
- "score": A float number representing the overall average score.

Example Output:
{{"reasoning": "Although one image is an oil painting and the other is a photo, the facial structure, eye shape, and \
clothing pattern are structurally identical.", "score": 4.0}}

Constraint:

- Output ONLY the raw JSON string.
- Do NOT use Markdown code blocks (e.g., ```json).
- If the subject is not present in either image, set "score" to 0.
"""

system_prompt_for_bg = """
You are an expert image quality assessor specializing in foreground object removal and background maintenance. Your \
task is to compare the original image and the edited image to determine (1) whether the specified foreground subject \
has been completely removed without any visible remnants (ghosting, halos, edge fragments, shadows, reflections, or \
deformation traces), and (2) whether the revealed background is clear, sharp, and visually natural (no blur smearing, \
low-resolution patches, inconsistent textures, or obvious inpainting artifacts).

Input:

1. Original Source Image: The image containing the foreground subjects.
2. Generated Background: The image after foreground subjects were removed.
3. Subject List: The list of foreground subjects that were supposed to be removed.

Core Responsibility:
You must detect Remnants (leftover parts of objects), Hallucinations (new objects appearing), and Inpainting Artifacts \
(blurring, distortion).

Critical Failure Rules (Zero Tolerance):

1. Human Limbs: If the subject was a person, and ANY body part (finger, foot, ear, hair silhouette) remains -> Score 0.
2. Compound Objects: If the subject is "Man on a Bike", BOTH the Man AND the Bike must be gone. If the Bike remains -> \
Score 1.
3. Ghosting: If a ghostly outline or shadow of the original subject is visible -> Score 2.

Context Awareness (The "Style Exception"):

- If No Style Reference: Expect pixel-perfect preservation (Inpainting standard).
- If Style Reference Exists: Pixel-level redrawing is ALLOWED (changing textures/colors to match style), but Scene \
Layout must be RIGID. The "underlying geometry" must not change.

Scoring Scale (0-10, Forensic Standard):

- 0 (Critical Failure): Human limbs found, or the subject is clearly still present.
- 2 (Major Remnants): Significant parts of the subject remain (e.g., a bag they were holding, a chair they sat on).
- 4 (Obvious Artifacts): Subject is gone, but the removed area is a blurry mess, has wrong colors, or obvious "patch" \
marks. Background geometry is warped.
- 6 (Passable but Detectable): Subject is gone. The background fill is reasonable but edges are unnatural (seams \
visible) or texture is muddy.
- 8 (High Quality): Clean removal. Background structure (lines, floor tiles) continues logically. No obvious seams.
- 10 (Perfection): Indistinguishable from an empty room photo. Lighting and grain match perfectly. No trace of the \
original subject exists.

Output Format
Return a SINGLE JSON object.
"""

user_prompt_for_bg = """
Please evaluate the quality of object removal by comparing the original and edited images.
Input Data:

- Original Image: [Image A] (The image containing the foreground subjects to be removed)
- Generated Image: [Image B] (The result to check)
- Subject List to Remove: {subject_list}

**Step-by-Step Inspection Protocol:**

STEP 1 - CRITICAL: Parse Compound Object Phrases

Before examining images, analyze the Subject List:

- Decompose: Split complex phrases (e.g., "Man holding a cup" -> Check Man + Check Cup).
- Rule: If the prompt implies associated objects (chair, bike, bag), they must also be removed unless part of the \
background.
- *Goal:* Ensure verification covers ALL components.

STEP 2 - CRITICAL: Human Residue Scan (Highest Priority)

- Zoom into the area in [Image B] where the subjects were likely located.
- Scan for: Hands, fingers, feet, toes, hair strands, ears, or shadows of limbs.
- Verdict: If ANY human part is found -> Set `has_human_remnants`: True and Score: 0.

STEP 3: Structural Alignment Check (Compare vs Reference)

- Compare [Image B] against [Image A].
- Geometry: Do the perspective lines (floorboards, ceiling corners) match the Reference?
- Landmarks: Are windows, doors, and furniture in the same positions?
- Consistency: Does the "filled" area logically continue the background lines?

STEP 4: Texture & Clarity Inspection (The "Clear" Check)

- Inspect the specific area where the subject was removed.
- Blurriness Check: Is the texture muddy or lower resolution than the surrounding background?
- Artifact Check: Are there weird "hallucinated" blobs, color smudges, or jagged seams? Are there weird phenomena that \
contradict physical laws (e.g., floating objects, impossible lighting)?
- *Goal:* The background should look crisp and natural, not like a bad Photoshop clone stamp.

STEP 5: Scoring Decision
Scoring Scale (0-10, Forensic Standard):

- 0 (Critical Failure): Human limbs found, or the subject is clearly still present.
- 2 (Major Remnants): Significant parts of the subject remain (e.g., a bag they were holding, a chair they sat on).
- 4 (Obvious Artifacts): Subject is gone, but the removed area is a blurry mess, has wrong colors or obvious "patch" \
marks, or contradicts physical laws. Background geometry is warped.
- 6 (Passable but Detectable): Subject is gone. The background fill is reasonable but edges are unnatural (seams \
visible) or texture is muddy.
- 8 (High Quality): Clean removal. Background structure (lines, floor tiles) continues logically. No obvious seams.
- 10 (Perfection): Indistinguishable from an empty room photo. Lighting and grain match perfectly. No trace of the \
original subject exists.

**Output Requirement:**
Return a JSON object with the following structure:
{{
    "compound_phrases_analysis": [
        {{"phrase": "original", "components": ["a", "b"], "status": "removed/visible"}}
    ],
    "has_human_remnants": `<bool>`,
    "structural_consistency": {{
        "aligned_with_reference": `<bool>`,
        "issues": "`<string>`"
    }},
    "clarity_quality": {{
        "is_blurry": `<bool>`,
        "has_artifacts": `<bool>`
    }},
    "final_score": <int 0-10>,
    "reasoning": "`<Concise summary of findings>`"
}}
"""

system_prompt_for_style = """
You are an expert AI Art Director and Visual Quality Assurance Specialist.
Your task is to evaluate a **Generated Image** against a **Style Reference Image**.

**Goal:** Assess whether the generated image faithfully applies the visual and artistic style from the reference image \
while maintaining high aesthetic quality.

### Evaluation Dimensions & Criteria

#### 1. Style Consistency (Medium & Texture)

Goal: Determine if the Target adopts the visual style of the "Style Reference".
Focus: Art medium (oil, watercolor, 3D, photo), brushwork, noise/grain, color palette, and line weight.
Ignore: The actual content (who or what is in the image).

Scoring (0-10, Conservative):
- 0: Complete Style Mismatch. The medium is fundamentally different. There is no attempt to match the reference.
  Example: Reference is a black-and-white sketch; Target is a hyper-realistic color photograph.
- 2: Superficial / Vague Resemblance. Matches the broad "mood" or color palette, but the medium is incorrect or \
unconvincing. It feels like a cheap filter was applied.
  Example: Reference is a thick impasto oil painting; Target is a flat digital illustration that just uses similar \
colors but lacks the 3D texture of the paint.
- 4: Generic Category Match. Correct medium (e.g., both are "Anime" or both are "Polaroids"), but fails to capture the \
specific nuances of the reference. It looks like a "stock" version of that style.
  Example: Reference is a specific 1990s anime style (grainy, muted colors); Target is a modern, high-gloss 4K anime \
style. The genre is right, but the era/technique is wrong.
- 6: Consistent Vibe & Palette. Accurately captures the medium and the dominant color grading. The image clearly \
belongs to the same "family" as the reference, though specific details (like line weight consistency or background \
rendering) may vary slightly.
  Example: The lighting and colors match the reference photography perfectly, but the film grain texture is slightly \
different.
- 8: Deep Stylistic Match. Captures complex stylistic signatures, including specific brushwork patterns, lighting \
behavior, or material rendering.
  Example: If the reference is a watercolor, the target replicates the specific "wet-on-wet" bleed effects and paper \
texture accurately. It feels like a cohesive piece of art.
- 10: Perfect Artist/Medium Mimicry. The Target looks like it was created by the exact same artist or the exact same \
camera/lens setup.
  Criteria: Texture, noise, stroke dynamics, and color science are identical. You could place the Target alongside the \
Reference in a portfolio, and no one would suspect they were generated differently.

### Output Format

Return a SINGLE valid JSON object.

JSON Structure:
{{
  "score": `<float>`, 
  "reasoning": "<concise_string>"

}}
"""

user_prompt_for_style = """
## Input Data

**Style Reference Images:**
See style reference image below.

**Target Generated Image:**
See target generated image below.

## Evaluation Task

Please analyze the Target Generated Image against the provided references.

**Constraint:**
- Be critical. Look for "AI look" artifacts.

Scoring (0-10, Conservative):
- 0: Complete Style Mismatch. The medium is fundamentally different. There is no attempt to match the reference.
  Example: Reference is a black-and-white sketch; Target is a hyper-realistic color photograph.
- 2: Superficial / Vague Resemblance. Matches the broad "mood" or color palette, but the medium is incorrect or \
unconvincing. It feels like a cheap filter was applied.
  Example: Reference is a thick impasto oil painting; Target is a flat digital illustration that just uses similar \
colors but lacks the 3D texture of the paint.
- 4: Generic Category Match. Correct medium (e.g., both are "Anime" or both are "Polaroids"), but fails to capture the \
specific nuances of the reference. It looks like a "stock" version of that style.
  Example: Reference is a specific 1990s anime style (grainy, muted colors); Target is a modern, high-gloss 4K anime \
style. The genre is right, but the era/technique is wrong.
- 6: Consistent Vibe & Palette. Accurately captures the medium and the dominant color grading. The image clearly \
belongs to the same "family" as the reference, though specific details (like line weight consistency or background \
rendering) may vary slightly.
  Example: The lighting and colors match the reference photography perfectly, but the film grain texture is slightly \
different.
- 8: Deep Stylistic Match. Captures complex stylistic signatures, including specific brushwork patterns, lighting \
behavior, or material rendering.
  Example: If the reference is a watercolor, the target replicates the specific "wet-on-wet" bleed effects and paper \
texture accurately. It feels like a cohesive piece of art.
- 10: Perfect Artist/Medium Mimicry. The Target looks like it was created by the exact same artist or the exact same \
camera/lens setup.
  Criteria: Texture, noise, stroke dynamics, and color science are identical. You could place the Target alongside the \
Reference in a portfolio, and no one would suspect they were generated differently.

Return ONLY the JSON object.
JSON Structure:
{{
  "score": `<float>`, 
  "reasoning": "<concise_string>"

}}
"""


def encode_image_to_base64(image_path, max_size=1024, quality=80):
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


def call_vlm_for_sf(image_path, train_data_base_path, subject, client):
    """
    调用 VLM 进行一致性评估
    :return: 解析后的 JSON 结果
    """
    target_path = Path(train_data_base_path, image_path)
    image_dir = target_path.parent.parent
    reference_path = image_dir / 'cropped_subjects' / (subject.replace(' ', '_').replace('-', '_-_') + '.png')

    user_prompt = user_prompt_for_sf.format(
        subject=subject
    )

    target_image_base64 = encode_image_to_base64(target_path)
    reference_image_base64 = encode_image_to_base64(reference_path)
    user_content = [{"type": "text", "text": user_prompt}]
    user_content.append({"type": "text", "text": 'reference images:'})
    user_content.append({"type": "image_url", "image_url": {"url": reference_image_base64}})  
    user_content.append({"type": "text", "text": 'target image:'})
    user_content.append({"type": "image_url", "image_url": {"url": target_image_base64}})

    messages = [
        {"role": "system", "content": system_prompt_for_sf},
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
        extra_body={"thinking_level": 'low'})
    
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
                extra_body={"thinking_level": 'low'})
            if response.choices[0].message.content is not None:
                break
        if response.choices[0].message.content is None:
            return {
                "reasoning": "Parse Error: " + response.choices[0].message.content[:50] + "...", 
                "score": 0.0
            }
    return parse_vlm_response(response.choices[0].message.content.strip())


def call_vlm_for_sf2(ori_img_path, transfer_img_path, subject, client):
    user_prompt = user_prompt_for_sf.format(
        subject=subject
    )

    transfer_image_base64 = encode_image_to_base64(transfer_img_path)
    ori_image_base64 = encode_image_to_base64(ori_img_path)
    user_content = [{"type": "text", "text": user_prompt}]
    user_content.append({"type": "text", "text": 'reference images:'})
    user_content.append({"type": "image_url", "image_url": {"url": ori_image_base64}})  
    user_content.append({"type": "text", "text": 'target image:'})
    user_content.append({"type": "image_url", "image_url": {"url": transfer_image_base64}})

    messages = [
        {"role": "system", "content": system_prompt_for_sf},
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
        extra_body={"thinking_level": 'low'})
    
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
                extra_body={"thinking_level": 'low'})
            if response.choices[0].message.content is not None:
                break
        if response.choices[0].message.content is None:
            return {
                "reasoning": "Parse Error: " + response.choices[0].message.content[:50] + "...", 
                "score": 0.0
            }
    return parse_vlm_response(response.choices[0].message.content.strip())


def call_vlm_for_bg(image_path, train_data_base_path, subject_list, client):
    target_path = Path(train_data_base_path, image_path)
    image_dir = target_path.parent.parent
    reference_path = image_dir / (image_dir.name + '.jpeg')

    user_prompt = user_prompt_for_bg.format(
        subject_list=subject_list
    )

    target_image_base64 = encode_image_to_base64(target_path)
    reference_image_base64 = encode_image_to_base64(reference_path)
    user_content = [{"type": "text", "text": user_prompt}]
    user_content.append({"type": "text", "text": 'original image:'})
    user_content.append({"type": "image_url", "image_url": {"url": reference_image_base64}})  
    user_content.append({"type": "text", "text": 'Generated background image:'})
    user_content.append({"type": "image_url", "image_url": {"url": target_image_base64}})

    messages = [
        {"role": "system", "content": system_prompt_for_bg},
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
        extra_body={"thinking_level": 'low'})
    
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
                extra_body={"thinking_level": 'low'})
            if response.choices[0].message.content is not None:
                break
        if response.choices[0].message.content is None:
            return {
                "reasoning": "Parse Error: " + response.choices[0].message.content[:50] + "...", 
                "score": 0.0
            }
    return parse_vlm_response(response.choices[0].message.content.strip())


def call_vlm_for_style(ref_image_path, train_data_base_path, client, target_img_path=None):
    reference_path = Path(train_data_base_path, ref_image_path)
    target_path = target_img_path if target_img_path else reference_path.parent / 'target.png'

    if not target_path.exists():
        target_path = reference_path.parent / 'target.jpeg'

    if not target_path.exists():
        return {
            "reasoning": "Target image not found", 
            "score": 0.0
        }


    user_prompt = user_prompt_for_style

    target_image_base64 = encode_image_to_base64(target_path)
    reference_image_base64 = encode_image_to_base64(reference_path)
    user_content = [{"type": "text", "text": user_prompt}]
    user_content.append({"type": "text", "text": 'Style reference image:'})
    user_content.append({"type": "image_url", "image_url": {"url": reference_image_base64}})  
    user_content.append({"type": "text", "text": 'Generated image:'})
    user_content.append({"type": "image_url", "image_url": {"url": target_image_base64}})

    messages = [
        {"role": "system", "content": system_prompt_for_style},
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
        extra_body={"thinking_level": 'low'})
    
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
                extra_body={"thinking_level": 'low'})
            if response.choices[0].message.content is not None:
                break
        if response.choices[0].message.content is None:
            return {
                "reasoning": "Parse Error: " + "...", 
                "score": 0.0
            }
    return parse_vlm_response(response.choices[0].message.content.strip())


def call_vlm_for_style2(reference_path, target_path, client):
    user_prompt = user_prompt_for_style

    target_image_base64 = encode_image_to_base64(target_path)
    reference_image_base64 = encode_image_to_base64(reference_path)
    user_content = [{"type": "text", "text": user_prompt}]
    user_content.append({"type": "text", "text": 'Style reference image:'})
    user_content.append({"type": "image_url", "image_url": {"url": reference_image_base64}})  
    user_content.append({"type": "text", "text": 'Generated image:'})
    user_content.append({"type": "image_url", "image_url": {"url": target_image_base64}})

    messages = [
        {"role": "system", "content": system_prompt_for_style},
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
        extra_body={"thinking_level": 'low'})
    
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
                extra_body={"thinking_level": 'low'})
            if response.choices[0].message.content is not None:
                break
        if response.choices[0].message.content is None:
            return {
                "reasoning": "Parse Error: " + "...", 
                "score": 0.0
            }
    return parse_vlm_response(response.choices[0].message.content.strip())
