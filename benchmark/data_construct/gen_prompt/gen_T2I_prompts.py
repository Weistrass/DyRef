from itertools import combinations
import os
import json
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import random

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

MODEL = "gemini-3-flash"  # Can also use gpt-4o-mini to save cost

SYSTEM_MSG = """
Role:
You are a prompt composer for text-to-image generation. Combine multiple subject phrases into one coherent, vivid, \
and conflict-free prompt suitable for modern diffusion models (e.g., SDXL, Midjourney, Flux). 

Tasks:
1. Merge and prioritize attributes across all subjects; resolve conflicts logically.
2. Clarify relationships, composition, and focal hierarchy (primary subject vs. secondary).
3. Add tasteful, photography/cinema/art direction: lighting, environment, background, camera, lens, shot type, color \
palette, mood, time of day, material details, and post-processing.
4. Preserve each subject's core identity; distribute attributes sensibly (don't duplicate or contradict).
5. For each subject that falls in the category of person, specify his/her pose. **The pose should be diversified, \
ranging from static to dynamic** (e.g., sitting with legs crossed, jumping with arms outstretched).
6. Avoid trademarked logos, personal identities, or disallowed content. Keep it SFW unless explicitly told otherwise.
7. Write fluent, natural English; avoid filler words.
8. Ensure the result is visually specific without being overly long (less than 60 words).
9. Emphasize that every subject is fully visible and well-defined.

Input format:
subject_phrases: a list of short noun phrases describing subjects or objects.
variants: number of alternative prompts to produce (default: 1).

Output format:
Produce exactly N variants (N = variants). For each variant output:
A single cohesive prompt ready for a text-to-image model.

Output requirements:
- Return ONLY the prompt text itself
- Do NOT include any labels like "Output:", "Prompt:", etc.
- Do NOT include any notes, explanations, or parenthetical remarks
- Do NOT use markdown formatting (**, *, _, etc.)

Example A 
Input:
subject phrases: ["A woman with neon pink hair", "Rustic farmhouse desk with chipped paint"]
variants: 1

Output:
A neon-pink-haired woman seated with legs crossed at a rustic farmhouse desk with chipped paint, sunlit studio corner, \
soft window light kissing worn wood grain, gentle filmic tones; medium portrait at eye level, shallow depth of field, \
warm neutrals with a pop of neon accent in hair and stationery.
"""

system_prompt_for_combination_check = """
# Role
You are a Senior Visual Logic Auditor and Scene Architect. Your goal is to analyze a list of subjects provided by the \
user and determine if they can be logically and harmoniously combined into a single, coherent Text-to-Image (T2I) scene.

# Task
Evaluate the "Visual Feasibility" of the subject list. You must assess whether the combination makes sense based on \
physical laws, environmental consistency, and narrative logic.

# Evaluation Dimensions
1. **Spatial Scale:** Are the relative sizes of the subjects too disparate? (e.g., a galaxy and a microscopic bacteria \
in a realistic shot).
2. **Environmental Compatibility:** Do the subjects belong to contradictory biomes or settings? (e.g., a deep-sea fish \
and a desert camel in a natural setting).
3. **Physical & Biological Logic:** Are there irreconcilable physical conflicts? (e.g., fire burning underwater \
without a magical/sci-fi context).
4. **Narrative Cohesion:** Do these subjects form a meaningful composition, or are they just a chaotic, cluttered pile \
of elements?

# Classification Criteria
- **[Reasonable]**: Subjects can coexist naturally, or can be unified through creative framing (e.g., Surrealism, \
Cyberpunk, Miniature photography, or Fantasy).
- **[Unreasonable]**: There is a fundamental logical break. Combining them would result in a visually jarring or \
nonsensical image that lacks aesthetic value (unless specifically intended as "glitch art").

# Output Format
You must respond only a boolean value: `true` if the combination is reasonable, `false` otherwise.

# Constraints
- Be open to "Creative Logic": If a combination is physically impossible but artistically compelling (e.g., a forest \
growing inside a lightbulb), mark it as `true`.
- Reject "Semantic Noise": If the list contains too many unrelated high-level subjects (e.g., "a pizza, a skyscraper, \
a medieval knight, and a quantum computer"), mark it as `false`.
"""


@dataclass
class CategoryInterval:
    """Category interval for interval-based sampling."""
    category: str
    subjects: List[str]
    start: float
    end: float

    def __repr__(self):
        return f"{self.category}: [{self.start:.4f}, {self.end:.4f})"


class IntervalSampler:
    """Interval-mapping sampler that samples subjects with category-level probability control."""

    def __init__(
        self,
        subjects: Dict[str, List[str]],
        priority_category: str = "person",
        priority_weight: float = 0.2
    ):
        """
        Args:
            subjects: mapping of {category: [subject list]}
            priority_category: name of the category to up-weight
            priority_weight: fraction of the [0, 1) interval assigned to the priority category
        """
        self.subjects = subjects
        self.priority_category = priority_category
        self.priority_weight = priority_weight

        self.intervals = self._build_intervals()
        self._prepare_binary_search()

    def _build_intervals(self) -> List[CategoryInterval]:
        """Build the category-to-interval mapping."""
        intervals = []
        current_pos = 0.0

        if self.priority_category in self.subjects:
            intervals.append(CategoryInterval(
                category=self.priority_category,
                subjects=self.subjects[self.priority_category],
                start=0.0,
                end=self.priority_weight
            ))
            current_pos = self.priority_weight

        other_categories = [
            cat for cat in self.subjects.keys()
            if cat != self.priority_category
        ]

        num_other = len(other_categories)
        remaining_interval = 1.0 - self.priority_weight
        interval_per_category = remaining_interval / num_other if num_other > 0 else 0

        for category in other_categories:
            intervals.append(CategoryInterval(
                category=category,
                subjects=self.subjects[category],
                start=current_pos,
                end=current_pos + interval_per_category
            ))
            current_pos += interval_per_category

        # Fix floating-point precision at the boundary
        if intervals:
            intervals[-1].end = 1.0

        return intervals

    def _prepare_binary_search(self):
        """Pre-compute interval end-points for binary search."""
        self.interval_bounds = [interval.end for interval in self.intervals]

    def _find_interval_linear(self, random_value: float) -> CategoryInterval:
        """Linear scan to find the interval containing random_value."""
        for interval in self.intervals:
            if interval.start <= random_value < interval.end:
                return interval
        return self.intervals[-1]

    def _find_interval_binary(self, random_value: float) -> CategoryInterval:
        """Binary search to find the interval containing random_value."""
        left, right = 0, len(self.intervals) - 1

        while left < right:
            mid = (left + right) // 2
            if random_value < self.intervals[mid].end:
                right = mid
            else:
                left = mid + 1

        return self.intervals[left]

    def sample_category(self, method: str = "binary") -> str:
        """
        Sample a category according to the interval distribution.

        Args:
            method: "linear" or "binary"

        Returns:
            Category name.
        """
        random_value = random.random()

        if method == "binary":
            interval = self._find_interval_binary(random_value)
        else:
            interval = self._find_interval_linear(random_value)

        return interval.category

    def sample_subject(self, method: str = "binary") -> Tuple[str, str]:
        """
        Sample a category and one subject from that category.

        Returns:
            (category name, subject description)
        """
        random_value = random.random()

        if method == "binary":
            interval = self._find_interval_binary(random_value)
        else:
            interval = self._find_interval_linear(random_value)

        subject = random.choice(interval.subjects)
        return interval.category, subject

    def sample_combination(
        self,
        num_subjects: int,
        allow_duplicate_categories: bool = True,
        max_attempts: int = 1000
    ) -> Dict:
        """
        Sample a combination of subjects.

        Args:
            num_subjects: number of subjects to select
            allow_duplicate_categories: whether the same category can appear more than once
            max_attempts: maximum retries to avoid infinite loops

        Returns:
            {
                "categories": [...],
                "subjects": [...],
                "random_values": [...]
            }
        """
        selected_categories = []
        selected_subjects = []
        random_values = []
        attempts = 0

        while len(selected_subjects) < num_subjects and attempts < max_attempts:
            attempts += 1

            random_value = random.random()
            interval = self._find_interval_binary(random_value)
            category = interval.category

            if not allow_duplicate_categories and category in selected_categories:
                continue

            subject = random.choice(interval.subjects)
            while subject in selected_subjects:
                subject = random.choice(interval.subjects)

            selected_categories.append(category)
            selected_subjects.append(subject)
            random_values.append(random_value)

        if len(selected_subjects) < num_subjects:
            print(f"Warning: only selected {len(selected_subjects)}/{num_subjects} subjects after {attempts} attempts.")

        return {
            "categories": selected_categories,
            "subjects": selected_subjects,
            "random_values": random_values
        }

    def visualize_sampling(self, num_samples: int = 1000):
        """Visualize the empirical sampling distribution."""
        category_counts = Counter()

        print(f"\nRunning {num_samples} sampling trials...")
        for _ in range(num_samples):
            category = self.sample_category()
            category_counts[category] += 1

        print("\nSampling statistics:")
        print(f"{'Category':<20} {'Count':<10} {'Actual':<12} {'Expected':<12} {'Error':<10}")
        print("-" * 70)

        if self.priority_category in category_counts:
            count = category_counts[self.priority_category]
            actual_prob = count / num_samples
            expected_prob = self.priority_weight
            error = abs(actual_prob - expected_prob)
            print(
                f"{self.priority_category:<20} {count:<10} "
                f"{actual_prob:<12.4f} {expected_prob:<12.4f} {error:<10.4f}"
            )

        other_categories = [
            (cat, count) for cat, count in category_counts.items()
            if cat != self.priority_category
        ]
        other_categories.sort(key=lambda x: -x[1])

        if other_categories:
            expected_prob_other = (1.0 - self.priority_weight) / len([
                c for c in self.subjects.keys() if c != self.priority_category
            ])

            print("\nOther categories (top 10):")
            for cat, count in other_categories[:10]:
                actual_prob = count / num_samples
                error = abs(actual_prob - expected_prob_other)
                print(f"{cat:<20} {count:<10} {actual_prob:<12.4f} {expected_prob_other:<12.4f} {error:<10.4f}")


def generate_prompts_for_combination(client: OpenAI, category: list) -> str:
    user_msg = f"subject phrases: {category}, variants: 1"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.9,
        max_tokens=2048,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
        extra_body={"thinking_level": "low"}
    )
    return resp.choices[0].message.content.strip()


def check_combination_validity(client: OpenAI, combination: List) -> bool:
    user_msg = f"subject phrases: {combination}"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.9,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt_for_combination_check},
            {"role": "user", "content": user_msg},
        ],
        extra_body={"thinking_level": "low"}
    )
    content = resp.choices[0].message.content.strip()
    return content == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    args = parser.parse_args()
    random.seed(40)

    with open(args.input, "r") as f:
        subjects = [json.loads(line) for line in f]

    sampler = IntervalSampler(
        subjects=subjects,
        priority_category="Person",
        priority_weight=0.4
    )
    sampler.visualize_sampling(num_samples=10000)

    for subjects_num in range(4, 6):
        for i in tqdm(range(500)):
            combination = sampler.sample_combination(
                subjects_num, allow_duplicate_categories=True, max_attempts=1000,
            )
            while not check_combination_validity(client, combination):
                print(combination)
                combination = sampler.sample_combination(
                    subjects_num, allow_duplicate_categories=True, max_attempts=1000,
                )
            result = generate_prompts_for_combination(client, combination["subjects"])
            dic = {
                "index": i,
                "categories": combination["categories"],
                "subjects": combination["subjects"],
                "prompt": result
            }
            with open(f"{args.output}/{subjects_num}_subjects.jsonl", "a") as f:
                f.write(json.dumps(dic, ensure_ascii=False) + "\n")

    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
