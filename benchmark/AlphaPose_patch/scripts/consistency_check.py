import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import euclidean
from tqdm import tqdm


class PoseConsistencyEvaluator:
    """COCO 17关键点姿态一致性评测器"""

    def __init__(self):
        # COCO 17关键点名称
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # COCO关键点的标准差系数 (用于OKS计算)
        # 这些值来自COCO官方，表示每个关键点的定位难度
        self.sigmas = np.array([
            0.026,  # nose
            0.025,  # left_eye
            0.025,  # right_eye
            0.035,  # left_ear
            0.035,  # right_ear
            0.079,  # left_shoulder
            0.079,  # right_shoulder
            0.072,  # left_elbow
            0.072,  # right_elbow
            0.062,  # left_wrist
            0.062,  # right_wrist
            0.107,  # left_hip
            0.107,  # right_hip
            0.087,  # left_knee
            0.087,  # right_knee
            0.089,  # left_ankle
            0.089   # right_ankle
        ])  # 缩放因子

        # 骨架连接（用于角度计算）
        self.skeleton = [
            [0, 1], [0, 2],  # 鼻子-眼睛
            [1, 3], [2, 4],  # 眼睛-耳朵
            [0, 5], [0, 6],  # 鼻子-肩膀
            [5, 7], [7, 9],  # 左臂
            [6, 8], [8, 10],  # 右臂
            [5, 6],          # 肩膀连线
            [5, 11], [6, 12],  # 躯干
            [11, 12],        # 髋部连线
            [11, 13], [13, 15],  # 左腿
            [12, 14], [14, 16]  # 右腿
        ]

    def parse_keypoints(self, keypoints):
        """
        将AlphaPose格式的关键点转为numpy数组

        参数:
            keypoints: list，[x1, y1, conf1, x2, y2, conf2, ...]

        返回:
            numpy数组 shape (17, 3)，每行为 [x, y, confidence]
        """
        kp_array = np.array(keypoints).reshape(-1, 3)
        # print(kp_array.shape)
        return kp_array


    def align_poses(self, kp1, kp2, method='procrustes'):
        """
        对齐两个姿态到相同的位置和尺度

        参数:
            method: 'center' - 对齐中心点
                    'procrustes' - Procrustes分析（更精确）
        """
        if method == 'center':
            # 计算中心点
            center1 = kp1[:, :2].mean(axis=0)
            center2 = kp2[:, :2].mean(axis=0)

            # 平移到原点
            kp1_aligned = kp1.copy()
            kp2_aligned = kp2.copy()
            kp1_aligned[:, :2] -= center1
            kp2_aligned[:, :2] -= center2

            # 归一化尺度（使用边界框对角线）
            scale1 = np.sqrt(((kp1_aligned[:, :2].max(axis=0) -
                            kp1_aligned[:, :2].min(axis=0))**2).sum())
            scale2 = np.sqrt(((kp2_aligned[:, :2].max(axis=0) -
                            kp2_aligned[:, :2].min(axis=0))**2).sum())

            kp1_aligned[:, :2] /= scale1
            kp2_aligned[:, :2] /= scale2

            return kp1_aligned, kp2_aligned

        elif method == 'procrustes':
            from scipy.spatial import procrustes

            # 1. 提取可见点的索引
            v = (kp1[:, 2] > 0) * (kp2[:, 2] > 0)
            visible_idx = np.where(v)[0]

            if len(visible_idx) < 3:
                # 可见点太少，退回到简单的中心对齐
                return self.align_poses(kp1, kp2, method='center')

            # 2. 执行 Procrustes 分析
            # 注意：scipy 会同时对两个输入进行平移、缩放和旋转
            # mtx1 对应第一个参数 (kp2)，mtx2 对应第二个参数 (kp1)
            mtx_ref, mtx_src, _ = procrustes(
                kp2[visible_idx, :2],
                kp1[visible_idx, :2]
            )

            # 3. 创建结果数组
            # 注意：因为 Procrustes 改变了尺度，不可见点在原坐标系已无意义，建议设为0
            kp1_aligned = np.zeros_like(kp1)
            kp2_aligned = np.zeros_like(kp2)

            # 填充变换后的坐标
            kp1_aligned[visible_idx, :2] = mtx_src
            kp1_aligned[visible_idx, 2] = kp1[visible_idx, 2]

            kp2_aligned[visible_idx, :2] = mtx_ref
            kp2_aligned[visible_idx, 2] = kp2[visible_idx, 2]

            return kp1_aligned, kp2_aligned


    def calculate_oks_aligned(self, kp1, kp2, align=True):
        """
        计算对齐后的OKS

        参数:
            align: 是否先对齐姿态
        """
        if align:
            kp1_aligned, kp2_aligned = self.align_poses(kp1, kp2, method='center')
        else:
            kp1_aligned, kp2_aligned = kp1, kp2

        # 使用统一的归一化尺度（设为1，因为已经归一化过了）
        s = 1.0

        # 计算距离
        d = np.sqrt(np.sum((kp1_aligned[:, :2] - kp2_aligned[:, :2])**2, axis=1))

        # 可见性
        v = (kp1[:, 2] > 0) * (kp2[:, 2] > 0)

        # OKS公式
        oks_per_point = np.exp(-d**2 / (2 * s**2 * self.sigmas**2))

        if v.sum() > 0:
            oks = (oks_per_point * v).sum() / v.sum()
        else:
            oks = 0.0

        return oks

    def calculate_pck(self, kp1, kp2, threshold=0.5, normalize='torso'):
        """
        计算PCK (Percentage of Correct Keypoints)

        参数:
            kp1, kp2: 关键点数组
            threshold: 阈值（相对于归一化尺度）
            normalize: 归一化方法 ('bbox', 'torso', 'head')

        返回:
            pck: float (0-1之间)
            correct_keypoints: 正确关键点的索引列表
        """
        # 计算归一化尺度
        kp1_aligned, kp2_aligned = self.align_poses(kp1, kp2, method='center')
        if normalize == 'bbox':
            # 使用边界框对角线长度
            x_min = min(kp1_aligned[:, 0].min(), kp2_aligned[:, 0].min())
            x_max = max(kp1_aligned[:, 0].max(), kp2_aligned[:, 0].max())
            y_min = min(kp1_aligned[:, 1].min(), kp2_aligned[:, 1].min())
            y_max = max(kp1_aligned[:, 1].max(), kp2_aligned[:, 1].max())
            scale = np.sqrt((x_max - x_min)**2 + (y_max - y_min)**2)

        elif normalize == 'torso':
            # 使用躯干长度（左肩到右髋的距离）
            scale1 = euclidean(kp1_aligned[5, :2], kp1_aligned[12, :2])  # left_shoulder to right_hip
            scale2 = euclidean(kp2_aligned[5, :2], kp2_aligned[12, :2])
            scale = (scale1 + scale2) / 2.0

        elif normalize == 'head':
            # 使用头部尺寸（鼻子到左耳的距离）
            scale1 = euclidean(kp1_aligned[0, :2], kp1_aligned[3, :2])  # nose to left_ear
            scale2 = euclidean(kp2_aligned[0, :2], kp2_aligned[3, :2])
            scale = (scale1 + scale2) / 2.0

        # 计算每个关键点的距离
        distances = np.sqrt(np.sum((kp1_aligned[:, :2] - kp2_aligned[:, :2])**2, axis=1))

        # 归一化距离
        normalized_distances = distances / scale

        # 判断哪些关键点是正确的
        v = (kp1_aligned[:, 2] > 0) * (kp2_aligned[:, 2] > 0)  # 两个都可见
        correct = (normalized_distances < threshold) * v

        # 计算PCK
        if v.sum() > 0:
            pck = correct.sum() / v.sum()
        else:
            pck = 0.0

        correct_keypoints = np.where(correct)[0].tolist()

        return pck, correct_keypoints

    def calculate_angle_similarity(self, kp1, kp2):
        """
        基于骨架角度的姿态相似度

        参数:
            kp1, kp2: 关键点数组

        返回:
            similarity: float (0-1之间，1表示完全一致)
        """
        angle_similarities = []

        for connection in self.skeleton:
            i, j = connection

            # 检查关键点是否可见
            if kp1[i, 2] > 0 and kp1[j, 2] > 0 and kp2[i, 2] > 0 and kp2[j, 2] > 0:
                # 计算向量
                vec1 = kp1[j, :2] - kp1[i, :2]
                vec2 = kp2[j, :2] - kp2[i, :2]

                # 计算角度（使用余弦相似度）
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                    # 限制在[-1, 1]范围内，避免数值误差
                    cos_sim = np.clip(cos_sim, -1.0, 1.0)
                    angle_similarities.append((cos_sim + 1) / 2)  # 转换到[0, 1]

        if len(angle_similarities) > 0:
            return np.mean(angle_similarities)
        else:
            return 0.0

    def calculate_internal_angles(self, kp):
        """计算人体主要的内部关节角度（视角相对鲁棒）"""
        # 定义由三个点组成的关节，例如 [左肩, 左肘, 左腕] 组成肘关节角度
        joints_config = [
            [5, 7, 9],   # 左肘
            [6, 8, 10],  # 右肘
            [11, 13, 15],  # 左膝
            [12, 14, 16],  # 右膝
            [5, 11, 13],  # 左髋
            [6, 12, 14],  # 右髋
            [7, 5, 11],  # 左腋下
            [8, 6, 12]   # 右腋下
        ]

        angles = []
        for j in joints_config:
            p1, p2, p3 = kp[j[0], :2], kp[j[1], :2], kp[j[2], :2]
            conf = kp[j, 2].min()
            if conf > 0.3:  # 只计算可见度高的点
                v1 = p1 - p2
                v2 = p3 - p2
                # 计算夹角
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angles.append(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            else:
                angles.append(None)
        return angles

    def calculate_cross_view_angle_consistency(self, kp1, kp2):
        angles1 = self.calculate_internal_angles(kp1)
        angles2 = self.calculate_internal_angles(kp2)

        scores = []
        for a1, a2 in zip(angles1, angles2):
            if a1 is not None and a2 is not None:
                # 使用高斯核或指数核，对小误差宽容，对大误差严厉
                diff = abs(a1 - a2)
                scores.append(np.exp(-diff**2 / (2 * 0.2**2)))  # 0.2弧度约11度
        return np.mean(scores) if scores else 0.0

    def comprehensive_evaluation(self, kp1, kp2):
        """
        综合评估两个姿态的一致性

        返回:
            结果字典，包含所有指标
        """
        results = {
            'oks': self.calculate_oks_aligned(kp1, kp2),
            'pck_0.5': self.calculate_pck(kp1, kp2, threshold=0.5)[0],
            'pck_0.2': self.calculate_pck(kp1, kp2, threshold=0.2)[0],
            'angle_similarity': self.calculate_angle_similarity(kp1, kp2)
        }

        # 综合评分（加权平均）
        results['overall_score'] = (
            results['oks'] * 0.2 +
            results['pck_0.5'] * 0.2 +
            results['pck_0.2'] * 0.1 +
            results['angle_similarity'] * 0.5
            # results['weighted_similarity'] * 0.2
        )

        return results

    def detailed_report(self, kp1, kp2):
        """生成详细的评估报告"""
        results = self.comprehensive_evaluation(kp1, kp2)

        print("=" * 60)
        print("姿态一致性评估报告")
        print("=" * 60)
        print(f"\n【核心指标】")
        oks_mark = '✓' if results['oks'] > 0.9 else '✗' if results['oks'] < 0.5 else '~'
        print(f"  OKS (COCO官方):        {results['oks']:.4f} {oks_mark}")
        overall_mark = (
            '✓' if results['overall_score'] > 0.9
            else '✗' if results['overall_score'] < 0.5
            else '~'
        )
        print(f"  综合评分:              {results['overall_score']:.4f} {overall_mark}")

        print(f"\n【距离指标】")
        print(f"  PCK@0.5:              {results['pck_0.5']:.4f}")
        print(f"  PCK@0.2:              {results['pck_0.2']:.4f}")

        print(f"\n【角度指标】")
        print(f"  角度相似度:            {results['angle_similarity']:.4f}")

        return results


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    evaluator = PoseConsistencyEvaluator()

    root_dir = Path('/cfs/cfs-kuxuxpyv/yusenfu/data_construct/data_4500/3_subjects/')
    results = []
    index = []
    for prompt_dir in tqdm(root_dir.iterdir()):
        pose_dir = prompt_dir / 'pose'
        pose_reference_dir = pose_dir / 'reference'
        if not pose_reference_dir.exists():
            continue
        for keypoint_file in pose_reference_dir.iterdir():
            if not keypoint_file.name.endswith('.json'):
                continue
            with open(keypoint_file, 'r') as f:
                result1 = json.load(f)
            with open(keypoint_file.parent.parent / 'keypoints' / keypoint_file.name, 'r') as f:
                result2 = json.load(f)
            kp1 = evaluator.parse_keypoints(result1['persons'][0]['keypoints'])
            kp2 = evaluator.parse_keypoints(result2['persons'][0]['keypoints'])
            result = evaluator.comprehensive_evaluation(kp1, kp2)
            results.append(result['overall_score'])
            if result['overall_score'] > 0.8:
                index.append((prompt_dir.name, keypoint_file.name))
    print(len(results))
    print(np.mean(results))
    print(len([r for r in results if r > 0.8]))
    print(index)

if __name__ == "__main__":
    example_usage()
