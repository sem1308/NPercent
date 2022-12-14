from typing import List, Dict
import numpy as np
from fastapi import FastAPI, Form
import keras
import tensorflow as tf
from pydantic import BaseModel
import json
from fastapi.middleware.cors import CORSMiddleware
# uvicorn main:app --reload

app = FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def bbox_clip(frame_x, frame_y, axis):
    x_min = tf.reduce_min(frame_x,axis,keepdims=True)
    x_max = tf.reduce_max(frame_x,axis,keepdims=True)
    y_min = tf.reduce_min(frame_y,axis,keepdims=True)
    y_max = tf.reduce_max(frame_y,axis,keepdims=True)

    center_x = (x_max + x_min)/ 2.
    center_y = (y_max + y_min)/ 2.
    return (frame_x - center_x) , (frame_y - center_y)

num_frames = 3
num_points = 17

def make_frame_to_tensor(frames):
    frame_data = np.zeros((1, num_frames, num_points*2), dtype='float32')
    for f_idx, frame in enumerate(frames):
        for i, f in enumerate(frame):
            frame_data[0, f_idx, 2 * i] = f['x']
            frame_data[0, f_idx, 2 * i + 1] = f['y']
    return frame_data

@app.get("/")
async def root():
    return {"message": "Hello World"}

''' trained model 가져오기 '''
exe_num = 5
model_path = f'./trained_model/npercent_model_{exe_num}'
model = keras.models.load_model(model_path)

class Item(BaseModel):
    frames : List[List[Dict]]
    exe_num : int

@app.post("/convert")
async def convert(item:Item):
    '''모델 불러오기'''
    # model_path = f'./trained_model/npercent_model{exe_num}'
    # model = keras.models.load_model(model_path)

    '''bbox clip'''
    frames = make_frame_to_tensor(item.frames)
    frames[:, :, ::2], frames[:, :, 1::2] = bbox_clip(frames[:, :, ::2], frames[:, :, 1::2],[1, 2])

    ''' trained model 에 frames 넣어 결과 도출'''
    with tf.device("/GPU:0"):  # "/gpu:0"
        result = model(frames, training=False)
        result = result * 100

    print(result)

    return result.cpu().numpy().tolist()[0]