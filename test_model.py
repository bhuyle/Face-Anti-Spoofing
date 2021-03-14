# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import imutils

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


def webcam(model_dir,device_id):

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    color = (255, 0, 0)
    cap = cv2.VideoCapture(0)
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
    print("[INFO] starting video stream...")

    max_num_faces = 20
    prev_frame_time = 0 
    while True:
        ret,frame = cap.read()
        if ret is None: 
            break
        (h, w) = frame.shape[:2]
        frame = imutils.resize(frame, width=640)
        new_frame_time = time.time()

        ### Use mobileFacenet
        # image_bbox = model_test.get_bbox(frame)
        # print(image_bbox)

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        model_name = '2.7_80x80_MiniFASNetV2.pth'

        for i in range(0, max_num_faces):
            confidence = detections[0, 0, i, 2]
            if confidence < 0.5:
                continue
            prediction = np.zeros((1, 3))
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            image_bbox = (startX, startY, endX-startX, endY-startY)
            start_time = time.time()
            for model_name in os.listdir(model_dir):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": frame,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            label = np.argmax(prediction)
            value = prediction[0][label]/2
            # print(prediction[0][label])
            if label == 1:
                # print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
                result_text = "RealFace Score: {:.2f}".format(value)
                color = (255, 0, 0)
            else:
                # print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
                result_text = "FakeFace Score: {:.2f}".format(value)
                color = (0, 0, 255)
            end_time = time.time()
            cv2.rectangle(
                frame,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                color, 2)
            cv2.putText(
                frame,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 1*frame.shape[0]/1024, color)
        fps = 1/ (end_time - start_time)
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame, str(fps), (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()
    # test(args.image_name, args.model_dir, args.device_id)
    webcam(args.model_dir, args.device_id)
