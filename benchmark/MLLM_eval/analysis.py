import json
import argparse
import numpy as np
from collections import defaultdict

# Configure input paths here
EVAL_RESULTS_PATH = (
    "/path/to/your/eval_results.jsonl"
)

DIMENSIONS = [
    'subject_consistency',
    'style_consistency',
    'background_consistency',
    'lighting_consistency',
    'aesthetic_quality',
    'text_adherence',
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result_json_path",
        type=str,
        required=True,
        help="Path to the evaluated result json path"
    )
    args = parser.parse_args()
    with open(args.result_json_path, "r") as f:
        data = [json.loads(line) for line in f]

    score_dict = defaultdict(list)
    for item in data:
        score_dict['overall_average_score'].append(item['overall_average_score'])
        for dimension in DIMENSIONS:
            if item[dimension]['score'] != -1:
                score_dict[dimension].append(item[dimension]['score'])

    print(len(score_dict['overall_average_score']))
    average_scores_per_dimension = []
    for dimension in DIMENSIONS:
        print(f'{dimension}: {np.mean(score_dict[dimension])}')
        average_scores_per_dimension.append(np.mean(score_dict[dimension]))

    print('Average score: ', np.mean(average_scores_per_dimension))
