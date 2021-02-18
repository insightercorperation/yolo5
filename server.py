import io

import cv2
import torch
import numpy as np
from PIL import Image
from numpy import random
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from starlette.responses import StreamingResponse

from utils.datasets import letterbox
from utils.torch_utils import select_device, time_synchronized
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box

from models.experimental import attempt_load

app = FastAPI()

@app.get("/health")
def read_root():
    return "healthy"

def decode_bytes_to_img(input_bytes):
    image = np.asarray(bytearray(input_bytes), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def numpy_image_to_bytes(np_image):
    im = Image.fromarray((np_image).astype(np.uint8))
    img_byte_arr = io.BytesIO()
    im.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    return img_byte_arr


@app.post("/uploadfile")
async def create_upload_file(category: str = Form(...), file: UploadFile = File(...)):

    detction_opt = {
        "conf_thres": 0.25,
        "iou_thres": 0.45
    }

    contents = await file.read()
    img0 = decode_bytes_to_img(contents)

    # set size to resize
    img_size = 416
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, :].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # set device
    device = select_device('cpu')
    half = device.type != 'cpu'

    # preprocess image
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # load weight according to the category
    weights_root_path = Path('./prod_weights')
    weight_path = weights_root_path / f"{category}_best.pt"

    try:
        model = attempt_load(weight_path, map_location=device)  # load FP32 model
    except FileNotFoundError:
        err_msg = f'카테고리는 {category}가 존재하지 않습니다. animal_head, cook, heart_shape, male, upper_body 중 하나를 선택해 주세요.'
        raise HTTPException(status_code=400, detail=err_msg)


    # execute model once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    # predict
    pred = model(img)[0]

    # apply NMS
    agnostic_nms = True
    classes = 0

    pred = non_max_suppression(pred, detction_opt["conf_thres"], detction_opt["iou_thres"], classes=classes, agnostic=agnostic_nms)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    for i, det in enumerate(pred):
        s = ''

        # 예측 결과 없는 경우
        if not len(det):
            raise HTTPException(status_code=400, detail="예측 결과가 없습니다.")

        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

        # Print results
        for c in det[:, -1].unique():
            n = (det[:, -1] == c).sum()  # detections per class
            s += f'{n} {names[int(c)]}s, '  # add to string

        for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)

    # return image
    img0 = img0[:,:, ::-1] # RGB => BGR
    img_bytes = numpy_image_to_bytes(img0)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpg")
