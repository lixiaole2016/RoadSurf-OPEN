import math
import os

import numpy as np


ROPE_CLASS_ALIASES = {
    'car': ['car'],
    'big_vehicle': ['big_vehicle', 'bigvehicle', 'bus', 'truck'],
    'big vehicle': ['big_vehicle', 'bigvehicle', 'bus', 'truck'],
    'pedestrian': ['pedestrian'],
    'cyclist': ['cyclist', 'motorcyclist', 'tricyclist'],
}


def _normalize_class_name(name):
    return name.lower().replace('-', '_')


def _class_aliases(name):
    normalized = _normalize_class_name(name)
    return ROPE_CLASS_ALIASES.get(normalized, [normalized])


def _safe_mean(values, default=0.0):
    return float(np.mean(values)) if len(values) > 0 else default


def _safe_percentile(values, q, default=0.0):
    return float(np.percentile(values, q)) if len(values) > 0 else default


def _read_label_objects(label_path):
    if not os.path.exists(label_path):
        return []

    objects = []
    with open(label_path, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) < 15:
                continue
            obj = {
                'type': _normalize_class_name(values[0]),
                'truncated': float(values[1]),
                'occluded': int(float(values[2])),
                'alpha': float(values[3]),
                'bbox': np.array([float(v) for v in values[4:8]], dtype=np.float64),
                'size_hwl': np.array([float(v) for v in values[8:11]], dtype=np.float64),
                'location': np.array([float(v) for v in values[11:14]], dtype=np.float64),
                'rotation_y': float(values[14]),
                'score': float(values[15]) if len(values) > 15 else 0.0,
            }
            objects.append(obj)
    return objects


def _read_denorm(denorm_path):
    with open(denorm_path, 'r') as f:
        values = f.readline().replace(',', ' ').split()
    if len(values) < 4:
        raise ValueError('Invalid Rope3D denorm file: {}'.format(denorm_path))
    return np.array([float(v) for v in values[:4]], dtype=np.float64)


def _bbox_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _bottom_corners(obj, denorm):
    h3d, w3d, l3d = obj['size_hwl']
    x3d, y3d, z3d = obj['location']
    ry3d = obj['rotation_y']

    rot_y = np.array([
        [math.cos(ry3d), 0.0, math.sin(ry3d)],
        [0.0, 1.0, 0.0],
        [-math.sin(ry3d), 0.0, math.cos(ry3d)],
    ], dtype=np.float64)

    x_corners = np.array([l3d / 2, l3d / 2, -l3d / 2, -l3d / 2], dtype=np.float64)
    y_corners = np.zeros(4, dtype=np.float64)
    z_corners = np.array([w3d / 2, -w3d / 2, -w3d / 2, w3d / 2], dtype=np.float64)
    corners = np.vstack((x_corners, y_corners, z_corners))
    corners = rot_y.dot(corners)

    # Rope3D boxes are evaluated in the road-ground frame induced by denorm.
    ground_rot = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -denorm[1], denorm[2]],
        [0.0, -denorm[2], -denorm[1]],
    ], dtype=np.float64)
    corners = ground_rot.dot(corners)
    corners += np.array([x3d, y3d, z3d], dtype=np.float64).reshape(3, 1)
    return corners.T


def _min_bottom_distance(pred_corners, gt_corners):
    distances = []
    for shift in range(4):
        shifted = np.roll(pred_corners, -shift, axis=0)
        distances.append(np.linalg.norm(shifted - gt_corners, axis=1).mean())
    return float(min(distances))


def _angle_diff(pred_angle, gt_angle):
    diff = abs(pred_angle - gt_angle)
    return min(diff, 2 * math.pi - diff)


def _object_similarity(pred, gt, denorm):
    if pred['type'] == 'pedestrian':
        pred = pred.copy()
        pred['rotation_y'] = gt['rotation_y']

    gt_distance = float(np.linalg.norm(gt['location'][[0, 2]]))
    gt_distance = max(gt_distance, 1e-7)

    pred_corners = _bottom_corners(pred, denorm)
    gt_corners = _bottom_corners(gt, denorm)
    bottom_distance = _min_bottom_distance(pred_corners, gt_corners)

    flipped_pred = pred.copy()
    if _angle_diff(pred['rotation_y'], gt['rotation_y']) > math.pi / 2.0:
        flipped_pred['rotation_y'] = pred['rotation_y'] + math.pi
        pred_corners_pi = _bottom_corners(flipped_pred, denorm)
        bottom_distance = min(bottom_distance, _min_bottom_distance(pred_corners_pi, gt_corners))

    center_delta = float(np.linalg.norm(gt['location'] - pred['location']))
    acs = 1.0 - min(1.0, center_delta / max(float(np.linalg.norm(gt['location'])), 1e-7))

    delta_theta = _angle_diff(pred['rotation_y'], gt['rotation_y'])
    aos = (1.0 + math.cos(2.0 * delta_theta)) / 2.0

    gt_area = max(float(gt['size_hwl'][1] * gt['size_hwl'][2]), 1e-7)
    pred_area = float(pred['size_hwl'][1] * pred['size_hwl'][2])
    aas = 1.0 - min(1.0, abs(gt_area - pred_area) / gt_area)

    ags = 1.0 - min(1.0, bottom_distance / gt_distance)

    return {
        'acs': acs,
        'aos': aos,
        'aas': aas,
        'ags': ags,
        'agd4': bottom_distance,
        'distance': gt_distance,
    }


def _match_file(pred_objects, gt_objects, denorm, class_aliases, iou_thresh, score_thresh, h_thresh):
    preds = [
        obj for obj in pred_objects
        if obj['type'] in class_aliases
        and obj['score'] >= score_thresh
        and obj['size_hwl'][0] > h_thresh
        and obj['location'][2] > 0
    ]
    gts = [
        obj for obj in gt_objects
        if obj['type'] in class_aliases
        and obj['size_hwl'][0] > h_thresh
        and obj['location'][2] > 0
    ]

    if len(preds) == 0 or len(gts) == 0:
        return []

    iou_table = np.array([[ _bbox_iou(pred['bbox'], gt['bbox']) for gt in gts] for pred in preds])
    similarities = []
    for pred_idx, pred in enumerate(preds):
        gt_idx = int(np.argmax(iou_table[pred_idx]))
        iou = iou_table[pred_idx, gt_idx]
        if iou <= iou_thresh:
            continue
        if int(np.argmax(iou_table[:, gt_idx])) != pred_idx:
            continue
        if pred['type'] != gts[gt_idx]['type']:
            continue
        similarities.append(_object_similarity(pred, gts[gt_idx], denorm))
    return similarities


def _summarize(similarities, ap_3d_r40=None):
    acs = _safe_mean([item['acs'] for item in similarities])
    aos = _safe_mean([item['aos'] for item in similarities])
    aas = _safe_mean([item['aas'] for item in similarities])
    ags = _safe_mean([item['ags'] for item in similarities])
    added = (acs + aos + aas + ags) / 4.0
    ap_value = float(ap_3d_r40) if ap_3d_r40 is not None else None
    rope_score = None if ap_value is None else (8.0 * ap_value + 2.0 * added * 100.0) / 10.0
    agd4_values = [item['agd4'] for item in similarities]
    return {
        'total': len(similarities),
        'ap_3d_r40': ap_value,
        'rope_score': rope_score,
        'acs': acs,
        'aos': aos,
        'aas': aas,
        'ags': ags,
        'added': added,
        'agd4_abs': _safe_mean(agd4_values),
        'agd4_q90': _safe_percentile(agd4_values, 90),
        'agd4_q99': _safe_percentile(agd4_values, 99),
    }


def _format_score(value):
    return '-' if value is None else '{:.3f}'.format(value)


def _format_summary_row(name, summary):
    return (
        '{:<10}\t{:d}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t'
        '{:.3f}\t{:.3f}\t{:.3f}'
    ).format(
        name,
        summary['total'],
        _format_score(summary['ap_3d_r40']),
        _format_score(summary['rope_score']),
        summary['acs'],
        summary['aos'],
        summary['aas'],
        summary['ags'],
        summary['added'],
        summary['agd4_abs'],
        summary['agd4_q90'],
        summary['agd4_q99'],
    )


def get_rope_score_eval_result(label_dir,
                               result_dir,
                               denorm_dir,
                               image_ids,
                               class_names,
                               ap_3d_r40=None,
                               iou_thresh=0.5,
                               score_thresh=0.0,
                               h_thresh=1.0,
                               ranges=((0, 30), (30, 60), (60, 90), (90, 120))):
    """Compute Rope3D RopeScore metrics.

    Matching and ground-corner similarity follow the Rope3D dataset tools, while
    RopeScore follows the paper definition: (8 * AP3D_R40 + 2 * S) / 10.
    """
    image_ids = [int(image_id) for image_id in image_ids]
    if isinstance(class_names, str):
        class_names = [class_names]
    ap_3d_r40 = ap_3d_r40 or {}

    lines = [
        'RopeScore metrics (IoU={:.2f})'.format(float(iou_thresh)),
        'class     \ttotal\tAP3D_R40\tRopeScore\tACS\tAOS\tAAS\tAGS\tS\tAGD4-m\tAGD4-90\tAGD4-99',
    ]
    result_dict = {}

    for class_name in class_names:
        class_aliases = _class_aliases(class_name)
        similarities = []
        skipped_denorm = 0

        for image_id in image_ids:
            image_key = '{:06d}'.format(image_id)
            denorm_path = os.path.join(denorm_dir, image_key + '.txt')
            if not os.path.exists(denorm_path):
                skipped_denorm += 1
                continue

            pred_path = os.path.join(result_dir, image_key + '.txt')
            gt_path = os.path.join(label_dir, image_key + '.txt')
            denorm = _read_denorm(denorm_path)
            pred_objects = _read_label_objects(pred_path)
            gt_objects = _read_label_objects(gt_path)
            similarities.extend(_match_file(
                pred_objects, gt_objects, denorm, class_aliases, iou_thresh, score_thresh, h_thresh))

        ap_value = ap_3d_r40.get(class_name)
        summary = _summarize(similarities, ap_value)
        key = _normalize_class_name(class_name)
        result_dict[key] = summary
        lines.append(_format_summary_row(class_name, summary))

        for range_start, range_end in ranges:
            range_items = [
                item for item in similarities
                if item['distance'] > range_start and item['distance'] <= range_end
            ]
            if len(range_items) == 0:
                continue
            range_summary = _summarize(range_items, None)
            range_name = '{}-{}m'.format(range_start, range_end)
            lines.append(_format_summary_row(range_name, range_summary))

        if skipped_denorm > 0:
            lines.append('Skipped {} {} samples without denorm files.'.format(skipped_denorm, class_name))

    return '\n'.join(lines) + '\n', result_dict
