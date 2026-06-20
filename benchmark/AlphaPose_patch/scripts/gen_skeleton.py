"""Script for saving pose skeleton visualization only."""
import argparse
import os
import platform
import sys
import time
import cv2
import numpy as np
import torch
from tqdm import tqdm
import natsort
import json

from detector.apis import get_detector
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.file_detector import FileDetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter
from pathlib import Path


def select_largest_person(heatmaps, boxes, scores=None):
    """
    Select the person with the largest bounding box area from multiple detections.

    Args:
        heatmaps: torch.Tensor, shape (num_persons, num_joints, H, W)
        boxes: numpy array, shape (num_persons, 4 or 5), [x1, y1, x2, y2, (score)]
        scores: optional detection scores

    Returns:
        heatmaps_largest: torch.Tensor, shape (1, num_joints, H, W)
        boxes_largest: numpy array, shape (1, 4/5)
        scores_largest: numpy array or None
        largest_idx: int, index of the person with the largest area
    """
    if heatmaps is None or heatmaps.shape[0] == 0:
        return heatmaps, boxes, scores, -1

    if boxes is None or boxes.shape[0] == 0:
        return heatmaps, boxes, scores, -1

    if heatmaps.shape[0] == 1:
        return heatmaps, boxes, scores, 0

    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)

    largest_idx = np.argmax(areas)

    heatmaps_largest = heatmaps[largest_idx:largest_idx+1]
    boxes_largest = boxes[largest_idx:largest_idx+1]

    if scores is not None:
        scores_largest = scores[largest_idx:largest_idx+1]
    else:
        scores_largest = None

    return heatmaps_largest, boxes_largest, scores_largest, largest_idx


class NestedDetectionLoader(DetectionLoader):
    """Detection loader that supports nested directory structures."""

    def __init__(self, image_dict, base_input_dir, detector, cfg, opt, batchSize=1, queueSize=1024):
        """
        Args:
            image_dict: dict mapping relative_path -> full_path
            base_input_dir: base input directory path
        """
        self.image_dict = image_dict
        self.base_input_dir = base_input_dir
        self.rel_paths = natsort.natsorted(list(image_dict.keys()))

        full_paths = [image_dict[rp] for rp in self.rel_paths]

        opt._original_inputpath = opt.inputpath
        opt.inputpath = self.base_input_dir

        super().__init__(full_paths, detector, cfg, opt,
                        batchSize=batchSize, mode='image', queueSize=queueSize)

        self.im_name_mapping = dict(zip(full_paths, self.rel_paths))

    def read(self):
        """Override read() to return relative paths instead of full paths."""
        result = super().read()
        if result[2] is not None:
            full_path = result[2]
            rel_path = self.im_name_mapping.get(full_path, full_path)
            result = list(result)
            result[2] = rel_path
            result = tuple(result)
        return result


"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Skeleton Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')
parser.add_argument('--detector', dest='detector',
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=64,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma '
                         'to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')

"""----------------------------- NEW: Skeleton options -----------------------------"""
parser.add_argument('--save_skeleton', default=False, action='store_true',
                    help='save skeleton visualization only (no original image)')
parser.add_argument('--skeleton_dir', type=str, default='skeletons',
                    help='directory to save skeletons (relative to outputpath)')
parser.add_argument('--background', type=str, default='black',
                    choices=['white', 'black', 'transparent'],
                    help='background color for skeleton')
parser.add_argument('--line_thickness', type=int, default=3,
                    help='thickness of skeleton lines')
parser.add_argument('--point_radius', type=int, default=2,
                    help='radius of keypoint circles')
parser.add_argument('--show_keypoint_name', default=False, action='store_true',
                    help='show keypoint names on skeleton')
parser.add_argument('--img_size', type=str, default='original',
                    help='output image size: original, 256x192, 512x384, or WxH')

"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)
"""----------------------------- New: Nested options -----------------------------"""
parser.add_argument('--nested_structure', default=False, action='store_true',
                    help='Enable nested directory structure processing (in_dir/i/cropped_subjects/)')
parser.add_argument('--cropped_subdir', type=str, default='cropped_subjects',
                    help='Name of the subdirectory containing cropped images')
parser.add_argument('--pose_subdir', type=str, default='pose',
                    help='Name of the subdirectory to save pose skeletons')
parser.add_argument('--save_keypoints', default=True, action='store_true',
                    help='save keypoint coordinates data')
parser.add_argument('--keypoints_subdir', type=str, default='keypoints',
                    help='subdirectory name to save keypoints data')
                    
args = parser.parse_args()
cfg = update_config(args.cfg)

if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector == 'tracker'

if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


# ==================== NEW: Skeleton Visualization Functions ====================

# COCO skeleton connections (17 keypoints)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
    [6, 12], [7, 13],                                    # torso
    [6, 8], [7, 9], [8, 10], [9, 11],                   # arms
    [2, 3], [1, 2], [1, 3],                             # head - eyes
    [2, 4], [3, 5],                                      # head - ears
    [4, 6], [5, 7]                                       # ears to shoulders
]

# COCO keypoint names (17 keypoints)
COCO_KEYPOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder',  # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle'     # 16
]

# Skeleton colors (RGB)
SKELETON_COLORS = [
    (255, 0, 0),    # left leg - red
    (255, 0, 0),
    (0, 0, 255),    # right leg - blue
    (0, 0, 255),
    (255, 255, 0),  # torso - yellow
    (255, 255, 0),
    (255, 255, 0),
    (255, 0, 255),  # left arm - magenta
    (255, 0, 255),
    (0, 255, 255),  # right arm - cyan
    (0, 255, 255),
    (0, 255, 0),    # head - green
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (128, 128, 128),  # ears to shoulders - gray
    (128, 128, 128)
]


def draw_skeleton(keypoints, img_size=(800, 600), background='white',
                 line_thickness=2, point_radius=4, show_names=False,
                 person_id=0):
    """
    Draw a pose skeleton image.

    Args:
        keypoints: numpy array, shape (num_joints, 3) [x, y, confidence]
        img_size: tuple (width, height)
        background: 'white', 'black', or 'transparent'
        line_thickness: thickness of skeleton lines
        point_radius: radius of keypoint circles
        show_names: whether to display keypoint names
        person_id: person ID (used for color variation)

    Returns:
        img: numpy array, RGB image
    """
    width, height = img_size

    if background == 'white':
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
    elif background == 'black':
        img = np.zeros((height, width, 3), dtype=np.uint8)
    elif background == 'transparent':
        img = np.zeros((height, width, 4), dtype=np.uint8)
        img[:, :, 3] = 0

    for i, (start_idx, end_idx) in enumerate(COCO_SKELETON):
        # COCO skeleton indices are 1-based
        start_idx -= 1
        end_idx -= 1

        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue

        start_point = keypoints[start_idx]
        end_point = keypoints[end_idx]

        if start_point[2] > 0.1 and end_point[2] > 0.1:
            x1, y1 = int(start_point[0]), int(start_point[1])
            x2, y2 = int(end_point[0]), int(end_point[1])

            x1 = np.clip(x1, 0, width - 1)
            y1 = np.clip(y1, 0, height - 1)
            x2 = np.clip(x2, 0, width - 1)
            y2 = np.clip(y2, 0, height - 1)

            color = SKELETON_COLORS[i % len(SKELETON_COLORS)]
            cv2.line(img, (x1, y1), (x2, y2), color, line_thickness)

    for joint_idx, (x, y, conf) in enumerate(keypoints):
        if conf > 0.1:
            x, y = int(x), int(y)
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)

            if joint_idx <= 4:      # head
                color = (0, 255, 0)
            elif joint_idx <= 10:   # upper limbs
                color = (255, 0, 255) if joint_idx % 2 == 1 else (0, 255, 255)
            else:                   # lower limbs
                color = (255, 0, 0) if joint_idx % 2 == 1 else (0, 0, 255)

            cv2.circle(img, (x, y), point_radius, color, -1)
            cv2.circle(img, (x, y), point_radius + 1, (255, 255, 255), 1)

            if show_names:
                text = COCO_KEYPOINT_NAMES[joint_idx]
                cv2.putText(img, text, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    return img


def heatmap_to_coord(heatmaps, boxes):
    """
    Convert heatmaps to original image coordinates.

    Args:
        heatmaps: torch.Tensor, shape (num_persons, num_joints, H, W)
        boxes: detection boxes used for coordinate transformation

    Returns:
        coords: numpy array, shape (num_persons, num_joints, 3)
    """
    heatmaps = heatmaps.cpu().numpy()
    num_persons, num_joints, hm_height, hm_width = heatmaps.shape

    coords = np.zeros((num_persons, num_joints, 3))

    for person_idx in range(num_persons):
        for joint_idx in range(num_joints):
            heatmap = heatmaps[person_idx, joint_idx]

            max_val = np.max(heatmap)
            if max_val < 0.01:
                coords[person_idx, joint_idx] = [0, 0, 0]
                continue

            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y, x = max_idx

            if boxes is not None and person_idx < boxes.shape[0]:
                box = boxes[person_idx]
                x1, y1, x2, y2 = box[:4]

                x_scaled = x1 + (x / hm_width) * (x2 - x1)
                y_scaled = y1 + (y / hm_height) * (y2 - y1)

                coords[person_idx, joint_idx] = [x_scaled, y_scaled, max_val]
            else:
                scale_x = 4
                scale_y = 4
                coords[person_idx, joint_idx] = [x * scale_x, y * scale_y, max_val]

    return coords


def save_skeleton_visualization(heatmaps, boxes, im_name, orig_img_shape, args):
    """
    Save pose skeleton visualization.

    Args:
        heatmaps: torch.Tensor, pose heatmaps
        boxes: detection boxes
        im_name: image name
        orig_img_shape: original image shape (H, W, C)
        args: parsed arguments
    """
    if heatmaps is None or heatmaps.shape[0] == 0:
        return

    keypoints = heatmap_to_coord(heatmaps, boxes)

    if args.img_size == 'original':
        img_size = (orig_img_shape[1], orig_img_shape[0])
    else:
        try:
            w, h = map(int, args.img_size.split('x'))
            img_size = (w, h)
        except ValueError:
            img_size = (800, 600)

    skeleton_path = os.path.join(args.outputpath, args.skeleton_dir)
    os.makedirs(skeleton_path, exist_ok=True)

    base_name = os.path.splitext(im_name)[0]

    for person_idx in range(keypoints.shape[0]):
        kp_person = keypoints[person_idx]

        skeleton_img = draw_skeleton(
            kp_person,
            img_size=img_size,
            background=args.background,
            line_thickness=args.line_thickness,
            point_radius=args.point_radius,
            show_names=args.show_keypoint_name,
            person_id=person_idx
        )

        if args.background == 'transparent':
            save_file = os.path.join(skeleton_path, f'{base_name}_person{person_idx}_skeleton.png')
            cv2.imwrite(save_file, cv2.cvtColor(skeleton_img, cv2.COLOR_RGBA2BGRA))
        else:
            save_file = os.path.join(skeleton_path, f'{base_name}_person{person_idx}_skeleton.jpg')
            cv2.imwrite(save_file, cv2.cvtColor(skeleton_img, cv2.COLOR_RGB2BGR))

    if args.debug:
        print(f'Saved {keypoints.shape[0]} skeleton(s) for {im_name}')


def save_keypoints_data(keypoints, rel_path, args):
    """
    Save keypoint data to file.

    Args:
        keypoints: numpy array, shape (num_persons, num_joints, 3) [x, y, confidence]
        rel_path: relative path of the source image
        args: parsed arguments
    """
    if keypoints is None or keypoints.shape[0] == 0:
        return

    parts = Path(rel_path).parts
    if len(parts) < 3:
        print(f"Warning: Unexpected path structure: {rel_path}")
        return

    subdir_name = parts[0]
    img_name = parts[-1]
    base_name = os.path.splitext(img_name)[0]

    if args.nested_structure:
        output_dir = Path(args._base_input_dir) / subdir_name / args.pose_subdir / 'reference'
    else:
        output_dir = Path(args.outputpath) / args.keypoints_subdir

    output_dir.mkdir(parents=True, exist_ok=True)

    save_keypoints_json(keypoints, output_dir, base_name, keypoints.shape[0], args)


def save_keypoints_json(keypoints, output_dir, base_name, num_persons, args):
    """Save keypoints in JSON format."""
    data = {
        'image_name': base_name,
        'num_persons': num_persons,
        'persons': []
    }

    for person_idx in range(num_persons):
        person_data = {
            'person_id': person_idx,
            'keypoints': []
        }

        for joint_idx in range(keypoints.shape[1]):
            x, y, conf = keypoints[person_idx, joint_idx]
            person_data['keypoints'].extend([x, y, conf])

        data['persons'].append(person_data)

    json_file = output_dir / f'{base_name}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if args.debug:
        print(f'Saved keypoints JSON to {json_file}')


def save_skeleton_for_nested_structure(heatmaps, boxes, rel_path, orig_img_shape, args):
    """
    Save skeleton images for nested directory structure.

    Args:
        heatmaps: torch.Tensor, pose heatmaps
        boxes: detection boxes
        rel_path: relative path, e.g. "0/cropped_subjects/img1.jpg"
        orig_img_shape: original image shape
        args: parsed arguments
    """
    if heatmaps is None or heatmaps.shape[0] == 0:
        return

    heatmaps, boxes, _, largest_idx = select_largest_person(heatmaps, boxes)

    if heatmaps is None or heatmaps.shape[0] == 0:
        if args.debug:
            print(f"No valid person detected for {rel_path}")
        return

    if args.debug:
        print(f"Selected person {largest_idx} with largest area for {rel_path}")

    # rel_path format: subdir_name/cropped_subjects/img_name.jpg
    parts = Path(rel_path).parts
    if len(parts) < 3:
        print(f"Warning: Unexpected path structure: {rel_path}")
        return

    subdir_name = parts[0]
    img_name = parts[-1]

    output_base = Path(args._base_input_dir) / subdir_name / args.pose_subdir / 'skeleton'
    output_base.mkdir(parents=True, exist_ok=True)

    keypoints = heatmap_to_coord(heatmaps, boxes)

    if args.img_size == 'original':
        img_size = (orig_img_shape[1], orig_img_shape[0])
    else:
        try:
            w, h = map(int, args.img_size.split('x'))
            img_size = (w, h)
        except ValueError:
            img_size = (800, 600)

    base_name = os.path.splitext(img_name)[0]

    for person_idx in range(keypoints.shape[0]):
        kp_person = keypoints[person_idx]

        skeleton_img = draw_skeleton(
            kp_person,
            img_size=img_size,
            background=args.background,
            line_thickness=args.line_thickness,
            point_radius=args.point_radius,
            show_names=args.show_keypoint_name,
            person_id=person_idx
        )

        if args.background == 'transparent':
            save_file = output_base / f'{base_name}.png'
            cv2.imwrite(str(save_file), cv2.cvtColor(skeleton_img, cv2.COLOR_RGBA2BGRA))
        else:
            save_file = output_base / f'{base_name}_skeleton.jpg'
            cv2.imwrite(str(save_file), cv2.cvtColor(skeleton_img, cv2.COLOR_RGB2BGR))

    if args.save_keypoints:
        save_keypoints_data(keypoints, rel_path, args)


def check_input():
    """Check and parse input source. Supports nested directory structures."""
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)

    if len(args.video):
        if os.path.isfile(args.video):
            videofile = args.video
            return 'video', videofile
        else:
            raise IOError('Error: --video must refer to a video file, not directory.')

    if len(args.detfile):
        if os.path.isfile(args.detfile):
            detfile = args.detfile
            return 'detfile', detfile
        else:
            raise IOError('Error: --detfile must refer to a detection json file, not directory.')

    if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
        inputpath = args.inputpath
        inputlist = args.inputlist
        inputimg = args.inputimg

        if len(inputlist):
            with open(inputlist, 'r') as f:
                im_names = f.readlines()
            return 'image', im_names

        elif len(inputpath) and inputpath != '/':
            if args.nested_structure:
                """
                Expected directory structure:
                in_dir/
                    0/
                        cropped_subjects/
                            img1.jpg
                            img2.jpg
                    1/
                        cropped_subjects/
                            img1.jpg
                """
                image_dict = {}
                base_dir = Path(inputpath)

                for subdir in tqdm(natsort.natsorted(base_dir.iterdir(), key=lambda x: x.name)):
                    if not subdir.is_dir():
                        continue
                    cropped_dir = subdir / 'cropped_subjects'
                    if not cropped_dir.exists():
                        continue

                    for image_path in cropped_dir.iterdir():
                        rel_path = image_path.relative_to(base_dir)
                        image_dict[str(rel_path)] = str(image_path)

                if len(image_dict) == 0:
                    raise ValueError(f"No images found in nested structure under {inputpath}")

                print(f"Found {len(image_dict)} images in {len(list(base_dir.iterdir()))} subdirectories")

                args._image_dict = image_dict
                args._base_input_dir = str(base_dir)

                im_names = natsort.natsorted(list(image_dict.keys()))
                return 'nested_image', im_names

            else:
                for root, dirs, files in os.walk(inputpath):
                    im_names = files
                im_names = natsort.natsorted(im_names)
                return 'image', im_names

        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]
            return 'image', im_names

    else:
        raise NotImplementedError


def print_finish_info():
    print('===========================> Finish Model Running.')
    if args.save_skeleton:
        print(f'===========================> Skeletons saved to: {os.path.join(args.outputpath, args.skeleton_dir)}')
    elif (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')


def loop():
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":
    mode, input_source = check_input()

    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    if mode == 'webcam':
        det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
        det_worker = det_loader.start()
    elif mode == 'detfile':
        det_loader = FileDetectionLoader(input_source, cfg, args)
        det_worker = det_loader.start()
    elif mode == 'nested_image':
        print("Using nested directory structure mode")
        det_loader = NestedDetectionLoader(
            args._image_dict,
            args._base_input_dir,
            get_detector(args),
            cfg,
            args,
            batchSize=args.detbatch,
            queueSize=args.qsize
        )
        det_worker = det_loader.start()
    else:
        det_loader = DetectionLoader(
            input_source, get_detector(args), cfg, args,
            batchSize=args.detbatch, mode=mode, queueSize=args.qsize,
        )
        det_worker = det_loader.start()

    # Load pose model
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,))
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)

    if args.pose_track:
        tracker = Tracker(tcfg, args)

    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)
    else:
        pose_model.to(args.device)
    pose_model.eval()

    runtime_profile = {'dt': [], 'pt': [], 'pn': []}

    # Init data writer (skeleton-only mode does not require DataWriter)
    if args.save_skeleton:
        writer = None
        print('Skeleton-only mode: DataWriter disabled')
    else:
        queueSize = 2 if mode == 'webcam' else args.qsize
        if args.save_video and mode != 'image':
            from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
            if mode == 'video':
                video_save_opt['savepath'] = os.path.join(
                    args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
            else:
                video_save_opt['savepath'] = os.path.join(
                    args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
            video_save_opt.update(det_loader.videoinfo)
            writer = DataWriter(
                cfg, args, save_video=True,
                video_save_opt=video_save_opt, queueSize=queueSize,
            ).start()
        else:
            writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()

    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...')
        sys.stdout.flush()
        im_names_desc = tqdm(loop())
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)

    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    if writer is not None:
                        writer.save(None, None, None, None, None, orig_img, im_name)
                    continue

                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)

                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)

                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)

                if args.pose_track:
                    boxes, scores, ids, hm, cropped_boxes = track(
                        tracker, args, orig_img, inps, boxes, hm,
                        cropped_boxes, im_name, scores,
                    )

                hm = hm.cpu()

                if args.save_skeleton:
                    if mode == 'nested_image':
                        save_skeleton_for_nested_structure(hm, boxes, im_name, orig_img.shape, args)
                    else:
                        save_skeleton_visualization(hm, boxes, im_name, orig_img.shape, args)
                else:
                    writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)

                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']),
                        pt=np.mean(runtime_profile['pt']),
                        pn=np.mean(runtime_profile['pn']))
                )

        print_finish_info()

        if writer is not None:
            while writer.running():
                time.sleep(1)
                print(
                    '===========================> Rendering remaining '
                    + str(writer.count()) + ' images in the queue...',
                    end='\r',
                )
            writer.stop()

        det_loader.stop()

    except Exception as e:  # pylint: disable=broad-except
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print_finish_info()
        if args.sp:
            det_loader.terminate()
            if writer is not None:
                while writer.running():
                    time.sleep(1)
                    print(
                        '===========================> Rendering remaining '
                        + str(writer.count()) + ' images in the queue...',
                        end='\r',
                    )
                writer.stop()
        else:
            det_loader.terminate()
            if writer is not None:
                writer.terminate()
                writer.clear_queues()
            det_loader.clear_queues()
