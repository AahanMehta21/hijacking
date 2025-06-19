'''apply only one fabrication attack'''
import sys
sys.path.append("../")
from models.yolov5_wrapper import YOLOv3
from pipeline_center import pipeline
from utils.utils import letterbox_image, box_iou
from PIL import Image

import cv2
import glob
import numpy as np
from keras import backend as K
import skvideo.io
import copy
import math
import os
from tqdm import tqdm
import torch

import pdb

class KerasYOLOv3Model_plus(YOLOv3):
    def detect_image(self, image):
        """Determines the locations of the cars in the image

        Args:
            image: numpy array

        Returns:
        detected objects with: bbox, confident score, class index
        [
            dictionary {
                bbox: np.array([left, up, right, down])
                score: confident_score
                class_idx: class_idx
                class_name: class name category
            }
        ]

        """
        pred_dic = self.predict(image)
        pred_list = self._dic2list(pred_dic)
        return pred_list

    def _dic2list(self, pred_dic):
        pred_list = []
        for temp_class, temp_score, temp_bbox in zip(pred_dic['classes'], pred_dic['scores'], pred_dic['boxes']):
            temp_dic = {}
            temp_dic['class_idx'] = temp_class
            temp_dic['score'] = temp_score
            temp_dic['bbox'] = [temp_bbox[1], temp_bbox[0], temp_bbox[3], temp_bbox[2]]
            try:
                temp_dic['class_name'] = self._class_names[temp_class]
            except:
                temp_dic['class_name'] = 'None'
            pred_list.append(temp_dic)
        return pred_list
            
        

def calculate_translation_center(bbox1, bbox2):
    '''
    calculate center translation vector of bbox1 to bbox2
    bbox : nparray ot list
        [left, top, right, bottom]
    ''' 
    bbox1 = np.array(bbox1).astype(float)
    bbox2 = np.array(bbox2).astype(float)
    center_1 = np.array([(bbox1[2] + bbox1[0]) / 2, (bbox1[3] + bbox1[1]) / 2])
    center_2 = np.array([(bbox2[2] + bbox2[0]) / 2, (bbox2[3] + bbox2[1]) / 2])
    return center_2 - center_1

def is_match(target_trk_id, target_det_id, match_info):
    match_list = match_info[0]
    for match_trk, match_det in match_list:
        if match_trk == target_trk_id and match_det == target_det_id:
            return True
    return False

def find_det_id_by_match_info(target_trk_id, match_info):
    match_list = match_info[0]
    unmatched_dets = match_info[1]
    unmatched_trks = match_info[2]
    if target_trk_id in unmatched_trks:
        raise ValueError('Target tracker is not matched to any detection.')
    for match_trk, match_det in match_list:
        if match_trk == target_trk_id:
            return match_det
    raise ValueError('Target tracker is not in tracker list.')

def bgr2rgb(bgr_array):
    temp = []
    temp.append(bgr_array[:,:,2])
    temp.append(bgr_array[:,:,1])
    temp.append(bgr_array[:,:,0])
    return np.transpose(np.array(temp),(1, 2, 0))

def bbox_inside_region(target_bbox, region_bbox):
    """
    Check if target_bbox is inside region_bbox
    target_bbox: [left, top, right, bottom]
    region_bbox: [left, top, right, bottom]
    """
    if (target_bbox[0] >= region_bbox[0] and
        target_bbox[1] >= region_bbox[1] and
        target_bbox[2] <= region_bbox[2] and
        target_bbox[3] <= region_bbox[3]):
        return True
    return False

def find_match_trk(match_info, det_id):
    match_info_pair = match_info[0]
    for temp_pair in match_info_pair:
        if temp_pair[0] == det_id:
            return temp_pair[1]
    return None

def sort_bbox_by_area(detected_objects_list):
    if not detected_objects_list:
        return detected_objects_list
    area_list = []
    for temp_det in detected_objects_list:
        temp_bbox = temp_det['bbox']
        temp_area = (temp_bbox[2] - temp_bbox[0]) * (temp_bbox[3] - temp_bbox[1])
        area_list.append(temp_area)
    sorted_idx = [i[0] for i in sorted(enumerate(area_list), key=lambda x:x[1], reverse = True)]
    ret_list = []
    for temp_idx in sorted_idx:
        ret_list.append(detected_objects_list[temp_idx])
    return ret_list

def _box_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

def nms_fine_tune(detected_objects_list, th=0.5):
    ret_list = []
    for i in range(len(detected_objects_list)):
        is_append = True
        for j in range(len(detected_objects_list)):
            if i == j:
                continue
            iou = box_iou(detected_objects_list[i]['bbox'], detected_objects_list[j]['bbox'])
            if iou > th and _box_area(detected_objects_list[i]['bbox']) <= _box_area(detected_objects_list[j]['bbox']):
                is_append = False
                break
        if is_append:
            ret_list.append(detected_objects_list[i])
    return ret_list

def is_missing_detection(detected_objects_list, target_bbox, target_id=0):
    temp_bbox = detected_objects_list[target_id]['bbox']
    temp_area = _box_area(temp_bbox)
    target_area = _box_area(target_bbox)
    if float(temp_area) / float(target_area) < 0.3:
        return True
    return False

def is_within_size(bbox, size):
    """
    Check if bbox is within the specified size.
    bbox: [left, top, right, bottom]
    size: [width, height]
    """
    bbox_size = _box_area(bbox)
    if bbox_size < size[0] * size[1]:
        return True
    return False
    
def tracker_bbox_list(tracker_list):
    ret = []
    for tracker in tracker_list:
        ret.append((tracker.obj))
    return ret

def apply_patch_to_frame(image_tensor, patch, attack_bbox):
    _, H, W = image_tensor.shape
    patch_size = patch.shape[1]  # assuming square patch

    # Place patch in upper-left corner of the attack_bbox
    x1 = int(attack_bbox[0])
    y1 = int(attack_bbox[1])

    # Ensure we don't go out of bounds
    if x1 + patch_size > W:
        x1 = W - patch_size
    if y1 + patch_size > H:
        y1 = H - patch_size

    # Apply patch to image
    patched_image = image_tensor.clone()
    patched_image[:, y1:y1+patch_size, x1:x1+patch_size] = patch

    return patched_image



def compute_attack_loss(detections, target_class, original_bbox, fabricated_bbox, patch_bbox, device='cuda' if torch.cuda.is_available() else 'cpu'):
    erase_loss = torch.tensor(0.0, device=device)
    fabricate_loss = torch.tensor(0.0, device=device)

    for det in detections:
        score = torch.tensor(det['score'], device=device)

        # Convert bboxes to tensors
        bbox_tensor = torch.tensor(det['bbox'], dtype=torch.float32, device=device)
        original_bbox_tensor = torch.tensor(original_bbox, dtype=torch.float32, device=device)
        fabricated_bbox_tensor = torch.tensor(fabricated_bbox, dtype=torch.float32, device=device)
        patch_bbox_tensor = torch.tensor(patch_bbox, dtype=torch.float32, device=device)

        iou_with_original = box_iou(bbox_tensor, original_bbox_tensor)
        iou_with_fabricated = box_iou(bbox_tensor, fabricated_bbox_tensor)
        iou_with_patch = box_iou(bbox_tensor, patch_bbox_tensor)

        if det['class_idx'] == target_class:
            if iou_with_original > 0.5:
                erase_loss = erase_loss + score  # erase = lower score
            if iou_with_fabricated > 0.5 and iou_with_patch > 0.5:
                fabricate_loss = fabricate_loss - score  # fabricate = raise score

    total_loss = erase_loss + fabricate_loss
    return total_loss


def attack_video(params, video_path=None, attack_det_id_dict=None, patch_bbox=None, moving_direction=None, verbose=0, is_return=False, region_bbox=None, check_region_bbox=False):

    detector = KerasYOLOv3Model_plus(sess = K.get_session())

    n_attacks = None

    videogen = skvideo.io.FFmpegReader(video_path)
    virtual_attack = False
    detected_objects_list_prev = None
    match_info_prev = None

    cal_dx_dy_flag = True
    attack_frame_list = [*attack_det_id_dict]
    attack_frame_list.sort()

    attacking_flag = False
    attack_count_idx = 0

    is_init = True
    params_min_hits = params['min_hits']

    patch_size = 60
    patch = torch.rand((3, patch_size, patch_size), requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=0.03)


    for frame_count, image in enumerate(videogen.nextFrame()): 
        if frame_count > 1:
            is_init = False

        image_yolo, _ = letterbox_image(image, shape=(416, 416), data_format='channels_last')
        # image = bgr2rgb((image_yolo * 255).astype(np.uint8))
        # Convert to tensor (CHW) and normalize to [0,1]
        image_tensor = torch.tensor(np.transpose(image_yolo, (2, 0, 1)), dtype=torch.float32)

        # Inject the patch
        patched_tensor = apply_patch_to_frame(image_tensor, patch, patch_bbox)
        patched_tensor = patched_tensor.to(device='cuda' if torch.cuda.is_available() else 'cpu')

        # Convert back to numpy HWC uint8
        patched_np = (patched_tensor.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        image = bgr2rgb(patched_np)
        image_yolo_pil = Image.fromarray(patched_np)
        detected_objects_list = detector.detect_image(image_yolo_pil)
        detected_objects_list = nms_fine_tune(detected_objects_list)

        detected_objects_list = sort_bbox_by_area(detected_objects_list)
        
        if len(detected_objects_list) != 0:
            nat_detected_objects_list = copy.deepcopy(detected_objects_list)
            
        if frame_count in attack_frame_list or attacking_flag == True:
            # print(attack_det_id_dict)

            target_det_id = attack_det_id_dict[frame_count - attack_count_idx][attack_count_idx]
            while check_region_bbox is True:
                if bbox_inside_region(detected_objects_list[target_det_id]['bbox'], region_bbox):
                    break
                else:
                    target_det_id += 1
                if target_det_id >= len(detected_objects_list):
                    print("No more detection to attack in specified region {}".format(region_bbox))
                    target_det_id = 0
                    break
                    if is_return:
                        return n_attacks
                    else:
                        break

            if attack_count_idx == 0:
                attacking_flag = True
                target_trk_id = find_match_trk(match_info_prev, target_det_id)
                target_init_bbox = detected_objects_list[target_det_id]['bbox']
                target_init_trk_bbox = (params_prev['tracker_list'][target_trk_id].obj)['bbox']
                print("Attack starts at frame {}".format(frame_count))
                print("Target bbox location in the original frame {}: {} ".format(frame_count, target_init_bbox))
            if attack_count_idx != 0:
                _, _, match_info_nat = pipeline(image, nat_detected_objects_list, frame_count, params_prev, detect_output=True, verbose=0, virtual_attack=virtual_attack, return_match_info=True)
                attacking_flag = is_match(target_trk_id, target_det_id, match_info_nat)
                if not attacking_flag:
                    detection_missing = is_missing_detection(nat_detected_objects_list, target_init_bbox, target_det_id)
                    try:
                        tracking_missing = is_missing_detection(tracker_bbox_list(params_prev['tracker_list']), target_init_trk_bbox, target_trk_id)
                    except:
                        pdb.set_trace()
                    if detection_missing and not tracking_missing:
                        attacking_flag = True
                    else:
                        print('Attack finished with {0} attacks.'.format(attack_count_idx))
                        n_attacks =  attack_count_idx
                        cal_dx_dy_flag = True
                        attack_count_idx = 0
                        return n_attacks

            if attacking_flag:
                optimizer.zero_grad()
                # loss compute
                loss = compute_attack_loss(detected_objects_list, detected_objects_list[target_det_id]['class_idx'], target_init_trk_bbox, detected_objects_list[target_det_id]['bbox'], patch_bbox)
                loss.backward()
                print(f"Loss: {loss.item()}, Patch mean: {patch.mean().item():.4f}")
                optimizer.step()
                print(f"Loss: {loss.item()}, Patch mean: {patch.mean().item():.4f}")

                with torch.no_grad():
                    patch.clamp_(0, 1)

                temp_attack_obj = detected_objects_list_prev[target_det_id]
                target_det_prev = temp_attack_obj
                target_trk_prev = params_prev['tracker_list'][target_trk_id].obj
                translation_vecter_center = calculate_translation_center(target_trk_prev['bbox'], target_det_prev['bbox'])

                attack_bbox = temp_attack_obj['bbox']
                attack_param = params_prev
                L = 5 #bbox moving pixel length

                if cal_dx_dy_flag and moving_direction is None:
                    if translation_vecter_center[0] == 0.:
                        ratio = 1000.0
                    else:
                        ratio = abs(translation_vecter_center[1] / translation_vecter_center[0])
                    dx = L * 1 / math.sqrt((1 + ratio * ratio))
                    dy = dx * ratio
                    if translation_vecter_center[0] > 0:
                        dx *= -1
                    if translation_vecter_center[1] > 0:
                        dy *= -1
                    cal_dx_dy_flag = False
                
                if attack_count_idx == 0:
                    for sub_attack_count in range(100):
                        if moving_direction is None:
                            fake_det_bbox = (target_trk_prev['bbox'] +  np.array([dx, dy, dx, dy]) * (sub_attack_count + 1)).astype(int)
                        else:
                            fake_det_bbox = (target_trk_prev['bbox'] +  np.array(moving_direction) * (sub_attack_count + 1)).astype(int)

                        detected_objects_list[target_det_id]['bbox'] = fake_det_bbox
                        _, param_attack, match_info = pipeline(image, detected_objects_list, frame_count, params, detect_output=True, verbose=0, virtual_attack=virtual_attack, return_match_info=True)
                        if is_match(target_trk_id, target_det_id, match_info):
                            attack_bbox = fake_det_bbox
                            attack_param = param_attack
                            if box_iou(patch_bbox, fake_det_bbox) <= 0.0:
                                break
                        else:
                            break
                    detected_objects_list[target_det_id]['bbox'] = attack_bbox
                else:
                    del detected_objects_list[target_det_id]

                print("Fabricate bbox location {} at frame {}".format(attack_bbox, frame_count))
                image_yolo_pil.save('./output/' + 'ori_' + str(frame_count) + '.png')
                attack_count_idx += 1

        image_track, params, match_info = pipeline(image, detected_objects_list, frame_count, params, detect_output=True, verbose=verbose, virtual_attack=virtual_attack, return_match_info=True, is_init=is_init)

        cv2.imwrite('./output/track/' + str(frame_count) + '.png', image_track)
        # Convert PIL image to OpenCV format
        image_to_show = np.array(image_yolo_pil)
        image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_RGB2BGR)

        # ======================= DRAWING =======================

        # Draw NATURAL detections (original YOLO detections) in GREEN
        if 'nat_detected_objects_list' in locals() and nat_detected_objects_list:
            for det in nat_detected_objects_list:
                bbox = det['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(image_to_show, (x1, y1), (x2, y2), (0, 255, 0), 2)  # GREEN for real detection

        # Draw ATTACK (fabricated) bbox in RED if currently attacking
        if 'attack_bbox' in locals() and attacking_flag:
            x1, y1, x2, y2 = int(attack_bbox[0]), int(attack_bbox[1]), int(attack_bbox[2]), int(attack_bbox[3])
            cv2.rectangle(image_to_show, (x1, y1), (x2, y2), (0, 0, 255), 2)  # RED for fabricated bbox
        
        if check_region_bbox and region_bbox is not None:
            # Draw the region bbox in BLUE
            x1, y1, x2, y2 = int(region_bbox[0]), int(region_bbox[1]), int(region_bbox[2]), int(region_bbox[3])
            cv2.rectangle(image_to_show, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.waitKey(0)
        # ========================================================

        # Show image
        cv2.imshow("Attacked Frame", image_to_show)

        # Wait key for a very short time
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        match_info_prev = copy.deepcopy(match_info)
        detected_objects_list_prev = copy.deepcopy(nat_detected_objects_list)
        params_prev = copy.deepcopy(params)

    return n_attacks

def cal_success_rate(input_list):
    results = []
    total_num = len(input_list)
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    for x in xs:
        count = 0
        for ret in input_list:
            if ret <= x:
                count += 1
        results.append(float(count) / float(total_num))
    return results



