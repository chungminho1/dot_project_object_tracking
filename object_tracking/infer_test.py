import mmcv_custom
import mmdet_custom
import ops_dcnv3
import tools
import torch
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import os
import random
import json
from json import JSONEncoder
from flask import Flask, jsonify, send_file
import numpy as np
import mmcv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = 'work_dirs/config_instance/best_bbox_mAP_epoch_12.pth'
configfile_path = 'configs/golf/config_custom.py'

model = init_detector(configfile_path, checkpoint_path)

# 이미지 디렉터리 경로
image_directory = 'data/bboxtest/'

# 디렉터리 내의 이미지 파일 목록 가져오기
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.jpeg', '.png'))]

# 이미지 파일들에 대한 객체 검출 수행
for img_path in image_files:
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)
    # 여기서 결과(result)를 원하는 대로 처리할 수 있습니다.
    print(img_path)
    for bbox in result:
        for box in bbox:
            if box.size>0:
                print(box[:4])


model = init_detector(configfile_path, checkpoint_path, device=device)  

#디렉토리 안에 있는 모든 이미지에 적용
def get_bounding_boxes(model, directory_path):
    # 디렉토리 내의 모든 파일 탐색
    image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    all_bounding_boxes = []

    # 디렉토리 내의 모든 이미지 파일에 대해 처리
    for file_name in image_files:
        file_path = os.path.join(directory_path, file_name)
        img = mmcv.imread(file_path)  # 이미지 파일 읽기 (mmcv 라이브러리 사용)
        
        # 모델을 이용해 이미지 추론 수행
        result = inference_detector(model, img)
        highest_confidence_bbox = None
        highest_confidence = -1.0

        # 각 이미지의 결과 처리
        for img_result in result:
            for obj in img_result:
                if obj.shape[0] > 0:
                    confidence = obj[4]
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        highest_confidence_bbox = obj[:4]
                    
        if highest_confidence_bbox is not None:
            center_x = (highest_confidence_bbox[0] + highest_confidence_bbox[2]) / 2.0
            center_y = (highest_confidence_bbox[1] + highest_confidence_bbox[3]) / 2.0
            center_coordinates = [center_x, center_y]
            all_bounding_boxes.append(center_coordinates)

    return all_bounding_boxes

test_path = 'data/bboxtest'
bboxresult = get_bounding_boxes(model, test_path)
print(bboxresult)

#골프공의 좌표 추출 후 json포맷으로 변환
file_path = os.path.join('./result', 'bboxresult.json')
with open(file_path, 'w') as file:
    json.dump(bboxresult, file)
