import numpy as np
from scipy.optimize import linear_sum_assignment


def IoU_evaluation(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def evaluation(gt_position_xy, gt_scale_xy, inf_shifts, inf_scales, inf_num, csize=50):

    num_instance = len(gt_position_xy)
    precision = np.zeros([num_instance, 11])
    recall = np.zeros([num_instance, 11])
    gt_max_iou = np.zeros([num_instance])
    detected_max_iou = np.zeros([num_instance])
    global_iou_mean = np.zeros([num_instance])

    for ins_iter in range(num_instance):
        gt_pos_ins, gt_scale_ins = gt_position_xy[ins_iter], gt_scale_xy[ins_iter]
        num_gt, num_inf = len(gt_pos_ins) // 2, int(inf_num[ins_iter])

        IoU = np.zeros([num_gt, num_inf])
        for ind in range(num_gt):
            for ind_inf in range(num_inf):
                gtx1, gty1 = gt_pos_ins[2 * ind], gt_pos_ins[2 * ind + 1]
                gtx2, gty2 = gtx1 + gt_scale_ins[2 * ind], gty1 + gt_scale_ins[2 * ind +
                                                                               1]
                gt_bbox = [gtx1, gty1, gtx2, gty2]

                inf_cx = inf_shifts[ins_iter, ind_inf, 0]
                inf_cy = inf_shifts[ins_iter, ind_inf, 1]
                inf_scale = inf_scales[ins_iter, ind_inf, 0]
                csize_2 = csize / 2

                inf_x1, inf_x2 = (inf_cx + 1) * csize_2 - inf_scale * csize_2, (
                    inf_cx + 1) * csize_2 + inf_scale * csize_2
                inf_y1, inf_y2 = (inf_cy + 1) * csize_2 - inf_scale * csize_2, (
                    inf_cy + 1) * csize_2 + inf_scale * csize_2
                inf_bbox = [inf_x1, inf_y1, inf_x2, inf_y2]

                IoU[ind, ind_inf] = IoU_evaluation(gt_bbox, inf_bbox)

        # gt_detected = np.max(detected, 1)
        if num_gt == 0 and num_inf == 0:
            precision[ins_iter, :] = 1
            recall[ins_iter, :] = 1
            gt_max_iou[ins_iter] = 1
            detected_max_iou[ins_iter] = 1
            global_iou_mean[ins_iter] = 1
        elif num_gt == 0:
            recall[ins_iter, :] = 1
        elif num_inf == 0:
            pass
        else:
            for i in range(11):
                threshold = i * 0.05 + 0.5
                detected = (IoU > threshold).astype(np.int32)
                inf_hited = np.max(detected, 0)
                true_positive = np.sum(inf_hited)
                # false_positive = num_inf - true_positive
                # false_negative = num_gt - np.sum(gt_detected)
                precision[ins_iter, i] = true_positive / num_inf
                recall[ins_iter, i] = true_positive / num_gt
            gt_max_iou[ins_iter] = np.mean(np.max(IoU, 1))
            detected_max_iou[ins_iter] = np.mean(np.max(IoU, 0))

            row_ind, col_ind = linear_sum_assignment(-1 * IoU)
            global_iou_mean[ins_iter] = np.sum(IoU[row_ind, col_ind]) / max(
                [num_inf, num_gt])

        # if ins_iter == 4:
        #     print(num_gt, num_inf, gt_pos_ins, gt_scale_ins, inf_shifts[ins_iter],
        #           inf_scales[ins_iter], IoU, gt_max_iou[ins_iter],
        #           detected_max_iou[ins_iter], precision[ins_iter], recall[ins_iter])

    return np.mean(precision, 0), np.mean(
        recall,
        0), np.mean(gt_max_iou), np.mean(detected_max_iou), np.mean(global_iou_mean)
