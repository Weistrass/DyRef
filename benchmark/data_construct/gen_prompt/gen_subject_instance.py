import os
import json
import argparse
from typing import List, Dict
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

MODEL = "deepseek-v3-local-II"

SYSTEM_MSG = """
You are a realistic prompt generator for text-to-image. Output only the prompts.
Role:
Please be very realistic and generate 20 brief subject prompts for text-to-image generation.

Follow these rules:
1. You will be given an <asset category>, you need to create an asset(breif subject prompt) based on
the <asset category>.
2. These descriptions can refer only to appearance descriptions/or to certain brands. e.g. "Elon Musk
in pajamas", "a tiger in a black hat", "A Mercedes sports car", "A blonde", "A door red on the left
and green on the right"
3. Focus on the given <asset category> ONLY. Avoid adding separate accessories or objects (e.g., when the <asset \
category> is "man", don't say "a man with a scarf", say "a man in red"; 
when the <asset category> is "camera", don't say "a child holding a camera", say "a vintage camera").
4. Do not repeat each asset, you need to use your logic and common sense of life to create.
5. No more than 12 words for each asset.

Example1
[asset category]: Book
Output:
[asset1]: A book with a green cover
[asset2]: commic book
[asset3]: math book
[asset4]: An open book
[asset5]: Rotten books
[asset6]: The book with "love and power" on the cover
[asset7]: A book with a blue key on it
...
(Up to [asset20])

User:
[asset category]: {category}
"""

CATEGORIES = [
    'Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle',
    'Desk', 'Cup', 'Street Lights', 'Cabinet', 'Handbag', 'Bracelet', 'Plate', 'Picture',
    'Helmet', 'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench',
    'Potted Plant', 'Bowl', 'Basin', 'Flag', 'Pillow', 'Boots', 'Vase', 'Microphone',
    'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 'Monitor', 'TV', 'Backpack', 'Umbrella',
    'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin', 'Slippers', 'Bicycle', 'Stool',
    'Barrel', 'Van', 'Couch', 'Sandals', 'Basket', 'Drum', 'Pen', 'Pencil', 'Bus', 'Wild Bird',
    'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned',
    'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat',
    'Laptop', 'Awning', 'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink',
    'Apple', 'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork',
    'Traffic Sign', 'Balloon', 'Tripod', 'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake',
    'Dining Table', 'Sheep', 'Hanger', 'Blackboard', 'Whiteboard', 'Napkin', 'Other Fish',
    'Orange', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan',
    'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin',
    'Soccer', 'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley',
    'Headphone', 'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane',
    'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat', 'Surveillance Camera',
    'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', 'Gun',
    'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite',
    'Strawberry', 'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper',
    'Cleaning Products', 'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 'Cutting Board',
    'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder', 'Snowboard',
    'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape', 'Giraffe',
    'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy',
    'Fire Truck', 'Billiards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase',
    'Cucumber', 'Cigar', 'Cigarette', 'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger',
    'Extractor', 'Extension Cord', 'Tong', 'Tennis Racket', 'Folder', 'American Football',
    'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide',
    'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine', 'Chicken',
    'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair balloon',
    'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach',
    'Rice', 'Wallet', 'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet',
    'Pineapple', 'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle',
    'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce',
    'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign',
    'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit',
    'Router', 'Modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper',
    'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hami melon', 'Flask',
    'Mushroom', 'Screwdriver', 'Soap', 'Recorder', 'Bear', 'Eggplant', 'Board Eraser',
    'Coconut', 'Tape Measure', 'Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak',
    'Crosswalk Sign', 'Stapler', 'Camel', 'Formula 1 Car', 'Pomegranate', 'Dishwasher',
    'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya',
    'Antelope', 'Parrot', 'Seal', 'Butterfly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal',
    'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill',
    'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Spring Rolls',
    'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus',
    'Barbell', 'Scallop', 'Noodles', 'Comb', 'Dumpling', 'Oyster', 'Table Tennis paddle',
    'Cosmetics Brush', 'Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra',
    'Lipstick', 'Cosmetics Mirror', 'Table Tennis',
]


def generate_prompts_for_category(client: OpenAI, category: str) -> List[str]:
    """Generate up to 20 deduplicated subject prompts for the given category."""
    user_msg = f"[asset category]: {category}"
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.9,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_msg},
        ],
    )
    content = resp.choices[0].message.content.strip()
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    # Deduplicate while preserving order, keep at most 20 entries
    seen = set()
    deduped = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            deduped.append(l)
        if len(deduped) >= 20:
            break
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Generate subject instance prompts for each category.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        help="OpenAI-compatible API base URL.",
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=args.base_url,
    )

    with open(args.output, "a", encoding="utf-8") as out_f:
        for cat in tqdm(CATEGORIES):
            try:
                prompts = generate_prompts_for_category(client, cat)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error processing category '{cat}': {e}")
                continue
            for j, p in enumerate(prompts, start=1):
                dic: Dict = {"category": cat, "index": j, "prompt": p}
                out_f.write(json.dumps(dic, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"Done. Saved to {args.output}.")


if __name__ == "__main__":
    main()
