"""Script for saving pose skeleton visualization only."""
import argparse
import os
import platform
from re import T
import sys
import time
import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import natsort
from consistency_check import PoseConsistencyEvaluator
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def select_largest_person(heatmaps, boxes, scores=None):
    """
    从多个检测结果中选择面积最大的人
    
    参数:
        heatmaps: torch.Tensor, shape (num_persons, num_joints, H, W)
        boxes: numpy array, shape (num_persons, 4 or 5), [x1, y1, x2, y2, (score)]
        scores: 可选，检测分数
    
    返回:
        heatmaps_largest: torch.Tensor, shape (1, num_joints, H, W)
        boxes_largest: numpy array, shape (1, 4/5)
        scores_largest: numpy array or None
        largest_idx: int, 最大面积人物的索引
    """
    if heatmaps is None or heatmaps.shape[0] == 0:
        return heatmaps, boxes, scores, -1

    if boxes is None or boxes.shape[0] == 0:
        return heatmaps, boxes, scores, -1

    # 如果只有一个人，直接返回
    if heatmaps.shape[0] == 1:
        return heatmaps, boxes, scores, 0

    # 计算每个检测框的面积
    areas = []
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        area = (x2 - x1) * (y2 - y1)
        areas.append(area)

    # 找到最大面积的索引
    largest_idx = np.argmax(areas)

    # 只保留最大的那个
    heatmaps_largest = heatmaps[largest_idx:largest_idx+1]  # 保持维度
    boxes_largest = boxes[largest_idx:largest_idx+1]

    if scores is not None:
        scores_largest = scores[largest_idx:largest_idx+1]
    else:
        scores_largest = None

    return heatmaps_largest, boxes_largest, scores_largest, largest_idx

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
                    help='choose which cuda device to use by index and input comma to use multi gpus.')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')

"""----------------------------- NEW: Skeleton options -----------------------------"""
parser.add_argument('--save_skeleton', default=False, action='store_true',
                    help='save skeleton visualization only (no original image)')

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

parser.add_argument('--output_path', type=str, default=None,
                    help='output directory path')
parser.add_argument('--reference_dir', type=str, required=True, default='',
                    help='path to reference dataset directory')
parser.add_argument('--json_data_path', type=str, required=True, default='',
                    help='path to JSON data file')

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

# COCO格式的骨架连接关系（17个关键点）
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # 腿部
    [6, 12], [7, 13],                                    # 躯干
    [6, 8], [7, 9], [8, 10], [9, 11],                   # 手臂
    [2, 3], [1, 2], [1, 3],                             # 头部-眼睛
    [2, 4], [3, 5],                                      # 头部-耳朵
    [4, 6], [5, 7]                                       # 耳朵到肩膀
]

# 关节名称（COCO 17个关键点）
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

# 骨架颜色（RGB格式）
SKELETON_COLORS = [
    (255, 0, 0),    # 左腿 - 红色
    (255, 0, 0),
    (0, 0, 255),    # 右腿 - 蓝色
    (0, 0, 255),
    (255, 255, 0),  # 躯干 - 黄色
    (255, 255, 0),
    (255, 255, 0),
    (255, 0, 255),  # 左臂 - 品红
    (255, 0, 255),
    (0, 255, 255),  # 右臂 - 青色
    (0, 255, 255),
    (0, 255, 0),    # 头部 - 绿色
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (128, 128, 128),  # 耳朵到肩膀 - 灰色
    (128, 128, 128)
]


def heatmap_to_coord(heatmaps, boxes):
    """
    从热图转换为原始图像坐标
    
    参数:
        heatmaps: torch.Tensor, shape (num_persons, num_joints, H, W)
        boxes: 检测框，用于坐标变换
    
    返回:
        coords: numpy array, shape (num_persons, num_joints, 3)
    """
    heatmaps = heatmaps.cpu().numpy()
    num_persons, num_joints, hm_height, hm_width = heatmaps.shape

    coords = np.zeros((num_persons, num_joints, 3))

    for person_idx in range(num_persons):
        for joint_idx in range(num_joints):
            heatmap = heatmaps[person_idx, joint_idx]

            # 找到热图峰值
            max_val = np.max(heatmap)
            if max_val < 0.01:
                coords[person_idx, joint_idx] = [0, 0, 0]
                continue

            max_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            y, x = max_idx

            # 映射回原始图像坐标
            if boxes is not None and person_idx < boxes.shape[0]:
                box = boxes[person_idx]
                x1, y1, x2, y2 = box[:4]

                # 缩放到检测框坐标
                x_scaled = x1 + (x / hm_width) * (x2 - x1)
                y_scaled = y1 + (y / hm_height) * (y2 - y1)

                coords[person_idx, joint_idx] = [x_scaled, y_scaled, max_val]
            else:
                # 如果没有box信息，使用热图坐标（需要缩放）
                scale_x = 4  # 根据模型输出调整
                scale_y = 4
                coords[person_idx, joint_idx] = [x * scale_x, y * scale_y, max_val]

    return coords


def get_keypoints(heatmaps, boxes, im_name, orig_img_shape, args):
    """
    提取关键点坐标
    
    参数:
        heatmaps: torch.Tensor, 姿态热图
        boxes: 检测框
        im_name: 图像名称
        orig_img_shape: 原始图像尺寸 (H, W, C)
        args: 参数
    """
    if heatmaps is None or heatmaps.shape[0] == 0:
        return

    heatmaps, boxes, _, _ = select_largest_person(heatmaps, boxes)

    # 转换热图为关键点坐标
    keypoints = heatmap_to_coord(heatmaps, boxes)

    base_name = os.path.splitext(im_name)[0]
    dic = save_keypoints_json(keypoints, '', base_name, keypoints.shape[0])
    return dic


def save_keypoints_json(keypoints, output_dir, base_name, num_persons):
    """保存为 JSON 格式"""
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
    return data

# ==================== Original Functions ====================


def check_input():
    """修改版：支持嵌套目录结构"""
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
            for _, _, files in os.walk(inputpath):
                im_names = files
            im_names = natsort.natsorted(im_names)
            return 'image', im_names

        elif len(inputimg):
            args.inputpath = os.path.split(inputimg)[0]
            im_names = [os.path.split(inputimg)[1]]
            return 'image', im_names

    else:
        raise NotImplementedError


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
    else:
        det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch,
            mode=mode, queueSize=args.qsize)
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

    # Init data writer (如果只保存骨架，可以不启动writer)
    if args.save_skeleton:
        writer = None
        print('Skeleton-only mode: DataWriter disabled')
    else:
        queueSize = 2 if mode == 'webcam' else args.qsize
        if args.save_video and mode != 'image':
            from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
            if mode == 'video':
                video_save_opt['savepath'] = os.path.join(args.outputpath,
                    'AlphaPose_' + os.path.basename(input_source))
            else:
                video_save_opt['savepath'] = os.path.join(args.outputpath,
                    'AlphaPose_webcam' + str(input_source) + '.mp4')
            video_save_opt.update(det_loader.videoinfo)
            writer = DataWriter(cfg, args, save_video=True, video_save_opt=video_save_opt, queueSize=queueSize).start()
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

    keypoints = {}

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
                    boxes, scores, ids, hm, cropped_boxes = track(tracker, args, orig_img, inps,
                        boxes, hm, cropped_boxes, im_name, scores)

                hm = hm.cpu()

                # ==================== MODIFIED: Save skeleton or use writer ====================

                keypoints[im_name.split('/')[-1].split('.')[0]] = get_keypoints(hm, boxes,
                    im_name, orig_img.shape, args)

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

        if writer is not None:
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining '
                + str(writer.count()) + ' images in the queue...', end='\r')
            writer.stop()

        det_loader.stop()

    except Exception as e:  # pylint: disable=broad-except
        print(repr(e))
        print('An error as above occurs when processing the images, please check it')
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        if args.sp:
            det_loader.terminate()
            if writer is not None:
                while(writer.running()):
                    time.sleep(1)
                    print('===========================> Rendering remaining '
                        + str(writer.count()) + ' images in the queue...', end='\r')
                writer.stop()
        else:
            det_loader.terminate()
            if writer is not None:
                writer.terminate()
                writer.clear_queues()
            det_loader.clear_queues()

    pose_dir = Path(args.inputpath)
    reference_base_dir = Path(args.reference_dir)
    with open(args.json_data_path, 'r') as f:
        reference_data = json.load(f)
    evaluator = PoseConsistencyEvaluator()
    results = []
    indices = []
    # print(len(keypoints))
    for img in pose_dir.iterdir():
        index = img.stem
        if keypoints.get(index, None) is None:
            print(f'No keypoints for {index}')
            continue
        item = reference_data[int(index)]
        for edit_image in item['edit_image']:
            if 'pose/reference' in edit_image or 'pose/skeleton' in edit_image:
                if 'pose/reference' in edit_image:
                    ref_image_path = reference_base_dir / edit_image
                    ref_image_name = ref_image_path.stem
                    if '_skeleton' in ref_image_name:
                        ref_image_name = ref_image_name.replace('_skeleton', '')
                    ref_keypoints_path = ref_image_path.parent / f"{ref_image_name}.json"
                    break
                else:
                    ref_image_path = reference_base_dir / edit_image
                    ref_image_name = ref_image_path.stem
                    ref_keypoints_path = ref_image_path.parent.parent / 'keypoints' / f"{ref_image_name}.json"
                    break
        gen_keypoints = keypoints[index]
        with open(ref_keypoints_path, 'r') as f:
            ref_keypoints = json.load(f)
        kp1, kp2 = evaluator.parse_keypoints(gen_keypoints['persons'][0]['keypoints']), evaluator.parse_keypoints(
            ref_keypoints['persons'][0]['keypoints'])
        result = evaluator.comprehensive_evaluation(kp1, kp2)
        results.append(result['overall_score'])
        indices.append(index)

    EXPECTED_SAMPLE_COUNT = 165  # expected number of evaluated samples
    num_missing = EXPECTED_SAMPLE_COUNT - len(list(pose_dir.iterdir()))
    results.extend([0] * num_missing)
    print("pose consistency: ", np.mean(results))
    for index, result in zip(indices, results):
        with open(os.path.join(args.output_path, 'pose.jsonl'), 'a') as f:
            f.write(json.dumps({'index': int(index), 'result': result}) + '\n')

    with open(os.path.join(args.output_path, 'pose.json'), 'w') as f:
        json.dump({"pose_score": np.mean(results)}, f)
