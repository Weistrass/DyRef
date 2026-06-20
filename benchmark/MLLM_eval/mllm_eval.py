import os
import argparse
import json
import logging
import time
import threading
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from vlm_utils import call_vlm_for_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Thread locks for file writing and rate limiting
file_lock = threading.Lock()
rate_limit_lock = threading.Lock()
last_request_time = 0

# Rate limit: 20 requests/min = 3s interval, with 0.1s buffer
MIN_REQUEST_INTERVAL = 3.1


def process_single_item(gen_img_path, test_set_dir, metadata, client, output_file, retries=3):
    global last_request_time
    index = gen_img_path.stem

    # --- 1. Strict rate limiting ---
    with rate_limit_lock:
        elapsed = time.time() - last_request_time
        wait_time = MIN_REQUEST_INTERVAL - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        last_request_time = time.time()

    # --- 2. API call with retry ---
    for attempt in range(retries):
        try:
            item = metadata[int(index)]
            response = call_vlm_for_all(gen_img_path, test_set_dir, item, client=client)
            response['index'] = int(index)

            with file_lock:
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(response, ensure_ascii=False) + "\n")
            return True

        except Exception as e:  # pylint: disable=broad-except
            error_msg = str(e)
            if "429" in error_msg:
                wait_backoff = 15 * (attempt + 1)
                logger.warning(f"Index {index} hit rate limit (429). Waiting {wait_backoff}s before retry...")
                time.sleep(wait_backoff)
            else:
                logger.error(f"Index {index}: {error_msg}")
                break  # No retry for non-rate-limit errors

    return False


def main():
    parser = argparse.ArgumentParser(description="Subject fidelity evaluation using VLM.")
    parser.add_argument('--gen_image_dir', type=str, required=True, help="Directory of generated images.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument('--test_set_dir', type=str, required=True, help="Path to the test set directory.")
    parser.add_argument('--metadata_file', type=str, required=True, help="Path to the metadata JSON file.")
    parser.add_argument('--base_url', type=str, default=None, help="OpenAI API base URL (optional).")
    # Rate limit is 20/min; workers > 4 will mostly wait in queue
    parser.add_argument('--max_workers', type=int, default=1, help="Number of concurrent threads (default: 1).")
    args = parser.parse_args()

    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(base_url=args.base_url, api_key=api_key)

    gen_image_dir = Path(args.gen_image_dir)
    test_set_dir = Path(args.test_set_dir)
    output_file = args.output_file

    with open(args.metadata_file, 'r') as f:
        metadata = json.load(f)

    all_img_paths = list(gen_image_dir.iterdir())
    logger.info(f"Total images: {len(all_img_paths)}. Rate limit: 20/min. Starting...")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(process_single_item, img_path, test_set_dir, metadata, client, output_file)
            for img_path in all_img_paths
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="VLM Eval"):
            pass


if __name__ == "__main__":
    main()
