from unittest import result
from matplotlib.pyplot import show
import Facenet
import torch
import numpy as np
import asyncio
import cv2
import imutils
import uvicorn
import os

from customer_data import CustomerData
from camera import Camera
from camera_in import CameraIn
from camera_out import CameraOut
from camera_checkout import CameraCheckout
from camera_behind import CameraBehind
from camera_before import CameraBefore

from facenet_pytorch import MTCNN
from numpy import expand_dims
from PIL import Image
from fastapi import FastAPI, HTTPException
from starlette.responses import StreamingResponse

def init(model):
    img = Image.new('RGB', (160, 160))
    face_pixels = np.array(img)
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)

def load_model(model_name):
    """
    Loads Yolo5 model from pytorch hub.
    :return: Trained Pytorch model.
    """
    if model_name:
        model = torch.hub.load('yolov5', 'custom', path=model_name, force_reload=True, source='local')
    else:
        model = torch.hub.load('yolov5', 'yolov5s', pretrained=True)    
    return model

async def gen(camera_process=None, show_fps=True):
    while True:
        try:
            if show_fps:
                frame = camera_process.get_frame_with_fps()
            else:
                frame = camera_process.get_frame()
            frame = imutils.resize(frame, width=1280)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        except Exception as e:
            print(e)
        finally:
            await asyncio.sleep(1/60)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

async def gen_error_video(cam_id):    
    while True:
        try:
            frame = np.array(Image.new('RGB', (1280, 720)))
            cv2.putText(frame, f'Camera {cam_id} is not found', (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        except Exception as e:
            print(e)
        finally:
            await asyncio.sleep(1/60)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def concat_images(images = None, col = 3):
    if images is None:
        return np.array(Image.new('RGB', (1280, 720)))
    if len(images) == 1:
        return images[0]

    while len(images)%col != 0:
        images.append(np.array(Image.new('RGB', (images[0].shape[1], images[0].shape[0]))))
    
    _list = []
    idx = 0
    while idx < len(images):
        frame = images[idx]
        for i in range(idx+1, idx+col):
            frame = np.concatenate((frame,images[i]),axis=1)
        _list.append(frame)
        idx+=col
    result = _list[0]
    for l in _list[1:]:
        result = np.concatenate((result,l),axis=0)
    return result

async def gen_all_cam(show_fps = True, col=3):
    if col < 1:
        col = 1
    while True:
        try:
            images = []
            for camera_process in list_cam_in_process:
                if show_fps:
                    frame = camera_process.get_frame_with_fps()
                else:
                    frame = camera_process.get_frame()
                images.append(frame)
            for camera_process in list_cam_out_process:
                if show_fps:
                    frame = camera_process.get_frame_with_fps()
                else:
                    frame = camera_process.get_frame()
                images.append(frame)
            for camera_process in list_cam_before_process:
                if camera_process is None:
                    continue
                if show_fps:
                    frame = camera_process.get_frame_with_fps()
                else:
                    frame = camera_process.get_frame()
                images.append(frame)
            for camera_process in list_cam_behind_process:
                if camera_process is None:
                    continue
                if show_fps:
                    frame = camera_process.get_frame_with_fps()
                else:
                    frame = camera_process.get_frame()
                images.append(frame)
            for camera_process in list_cam_checkout_process:
                if show_fps:
                    frame = camera_process.get_frame_with_fps()
                else:
                    frame = camera_process.get_frame()
                images.append(frame)
            frame = concat_images(images, col)
            frame = imutils.resize(frame, width=1280)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
        except Exception as e:
            print(e)
        finally:
            await asyncio.sleep(1/60)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

source_server_url = os.environ.get("source_server_url", "0.0.0.0")
_allow_loop = eval(os.environ.get("allow_loop", "True"))
list_cam_in = os.environ.get("list_cam_in", "cam11").split(",")
list_cam_out = os.environ.get("list_cam_out", "cam12").split(",")
list_cam_checkout = os.environ.get("list_cam_checkout", "cam13").split(",")
list_cam_before_behind = os.environ.get("list_cam_before_behind", "cam14,cam15").split(";")
list_cam_before = []
list_cam_behind = []
for cam_name in list_cam_before_behind:
    cam_names = cam_name.split(",")
    list_cam_before.append(cam_names[0])
    list_cam_behind.append(cam_names[1])


list_cam_in_obj = []
list_cam_out_obj = []
list_cam_checkout_obj = []
list_cam_before_obj = []
list_cam_behind_obj = []

if not source_server_url or source_server_url == "0.0.0.0":
    # cam_in_vs = os.environ.get("cam_in_video_source", "videos/ff_ci_720.mp4")
    # cam_out_vs = os.environ.get("cam_out_video_source", "videos/ff_co_720.mp4")
    # cam_checkout_vs = os.environ.get("cam_checkout_video_source", "videos/ff_ctn_720.mp4")
    # cam_behind_vs = os.environ.get("cam_behind_video_source", "videos/ff_cs_720.mp4")
    # cam_before_vs = os.environ.get("cam_before_video_source", "videos/ff_ct_720.mp4")

    camera_in_obj = Camera(video_source = "videos/ff_ci_720.mp4", cam_name="cam11", allow_loop = _allow_loop, motion_detect_enabled = True)
    camera_out_obj = Camera(video_source = "videos/ff_co_720.mp4", cam_name="cam12", allow_loop = _allow_loop, motion_detect_enabled = True)
    camera_checkout_obj = Camera(video_source = "videos/ff_ctn_720.mp4", cam_name="cam13", allow_loop = _allow_loop, motion_detect_enabled = True)
    camera_before_obj = Camera(video_source = "videos/ff_ct_720.mp4", cam_name="cam14", allow_loop = _allow_loop, motion_detect_enabled = True)
    camera_behind_obj = Camera(video_source = "videos/ff_cs_720.mp4", cam_name="cam15", allow_loop = _allow_loop, motion_detect_enabled = True)

    list_cam_in_obj.append(camera_in_obj)
    list_cam_out_obj.append(camera_out_obj)
    list_cam_checkout_obj.append(camera_checkout_obj)
    list_cam_before_obj.append(camera_before_obj)
    list_cam_behind_obj.append(camera_behind_obj)

else:
    for cam_name in list_cam_in:
        video_url = source_server_url + "/" + cam_name
        camera_obj = Camera(video_source = video_url, cam_name = cam_name, allow_loop = _allow_loop, motion_detect_enabled = True)
        list_cam_in_obj.append(camera_obj)

    for cam_name in list_cam_out:
        video_url = source_server_url + "/" + cam_name
        camera_obj = Camera(video_source = video_url, cam_name = cam_name, allow_loop = _allow_loop, motion_detect_enabled = True)
        list_cam_out_obj.append(camera_obj)

    for cam_name in list_cam_checkout:
        video_url = source_server_url + "/" + cam_name
        camera_obj = Camera(video_source = video_url, cam_name = cam_name, allow_loop = _allow_loop, motion_detect_enabled = True)
        list_cam_checkout_obj.append(camera_obj)

    for cam_name in list_cam_before:
        if cam_name:
            video_url = source_server_url + "/" + cam_name
            camera_obj = Camera(video_source = video_url, cam_name = cam_name, allow_loop = _allow_loop, motion_detect_enabled = True)
            list_cam_before_obj.append(camera_obj)
        else:
            list_cam_before_obj.append(None)

    for cam_name in list_cam_behind:
        if cam_name:
            video_url = source_server_url + "/" + cam_name
            camera_obj = Camera(video_source = video_url, cam_name = cam_name, allow_loop = _allow_loop, motion_detect_enabled = True)
            list_cam_behind_obj.append(camera_obj)
        else:
            list_cam_behind_obj.append(None)

customer_data = CustomerData()
# camera_in_source = Camera(video_source = cam_in_vs, allow_loop = _allow_loop, motion_detect_enabled = True)
# camera_out_source = Camera(video_source = cam_out_vs, allow_loop = _allow_loop, motion_detect_enabled = True)
# camera_checkout_source = Camera(video_source = cam_checkout_vs, allow_loop = _allow_loop, motion_detect_enabled = True)
# camera_behind_source = Camera(video_source = cam_behind_vs, allow_loop = _allow_loop, motion_detect_enabled = True)
# camera_before_source = Camera(video_source = cam_before_vs, allow_loop = _allow_loop, motion_detect_enabled = True)

facenet_model = Facenet.loadModel()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(
    thresholds= [0.4, 0.6, 0.6],
    keep_all=True,
    min_face_size=60,
    device=device
)

init(facenet_model)

yolov5_model = load_model("my_model/yolov5l_5obj_0721.pt")

class UseDeviceToken:
    def __init__(self):
        self.tokens = []
use_device_tokens = UseDeviceToken()

# camera_in_process = CameraIn(camera_in_source, facenet_model, detector, customer_data)
# camera_out_process = CameraOut(camera_out_source, facenet_model, detector, customer_data, use_device_tokens)
# camera_checkout_process = CameraCheckout(camera_checkout_source, facenet_model, detector, customer_data)
# camera_behind_process = CameraBehind(camera_behind_source, facenet_model, detector, customer_data)
# camera_before_process = CameraBefore(camera_before_source, yolov5_model, 0.65, "csrt", camera_behind_process, 600, use_device_tokens)

# camera_before_process_list = list()
# camera_before_process_list.append(camera_before_process)

list_cam_in_process = []
list_cam_out_process = []
list_cam_checkout_process = []
list_cam_behind_process = []
list_cam_before_process = []

process_of = {}

for idx, camera_obj in enumerate(list_cam_in_obj):
    camera_in_process = CameraIn(camera_obj, facenet_model, detector, customer_data)
    list_cam_in_process.append(camera_in_process)
    process_of[list_cam_in[idx]] = camera_in_process

for idx, camera_obj in enumerate(list_cam_out_obj):
    camera_out_process = CameraOut(camera_obj, facenet_model, detector, customer_data, use_device_tokens)
    list_cam_out_process.append(camera_out_process)
    process_of[list_cam_out[idx]] = camera_out_process

for idx, camera_obj in enumerate(list_cam_checkout_obj):
    camera_checkout_process = CameraCheckout(camera_obj, facenet_model, detector, customer_data)
    list_cam_checkout_process.append(camera_checkout_process)
    process_of[list_cam_checkout[idx]] = camera_checkout_process

for idx, camera_obj in enumerate(list_cam_behind_obj):
    if camera_obj is None:
        list_cam_behind_process.append(None)
        continue
    camera_behind_process = CameraBehind(camera_obj, facenet_model, detector, customer_data)
    list_cam_behind_process.append(camera_behind_process)
    process_of[list_cam_behind[idx]] = camera_behind_process

for idx, camera_obj in enumerate(list_cam_before_obj):
    if camera_obj is None:
        list_cam_before_process.append(None)
        continue
    camera_before_process = CameraBefore(camera_obj, yolov5_model, 0.65, "csrt", list_cam_behind_process[idx], 600, use_device_tokens)
    list_cam_before_process.append(camera_before_process)
    process_of[list_cam_before[idx]] = camera_before_process

for camera_obj in list_cam_in_obj:
    camera_obj.run()

for camera_obj in list_cam_out_obj:
    camera_obj.run()

for camera_obj in list_cam_checkout_obj:
    camera_obj.run()

for camera_obj in list_cam_behind_obj:
    if camera_obj is None:
        continue
    camera_obj.run()

for camera_obj in list_cam_before_obj:
    if camera_obj is None:
        continue
    camera_obj.run()

for camera_process in list_cam_in_process:
    camera_process.run()

for camera_process in list_cam_out_process:
    camera_process.run()

for camera_process in list_cam_checkout_process:
    camera_process.run()

for camera_process in list_cam_behind_process:
    if camera_process is None:
        continue
    camera_process.run()

for camera_process in list_cam_before_process:
    if camera_process is None:
        continue
    camera_process.run()

# out = None

# while True:
#     frame1 = camera_in_process.get_frame_with_fps()
#     frame2 = camera_out_process.get_frame_with_fps()
#     frame3 = camera_before_process.get_frame_with_fps()
#     frame4 = camera_behind_process.get_frame_with_fps()
#     frame5 = camera_checkout_process.get_frame_with_fps()

#     if frame1 is None or frame2 is None or frame3 is None or frame4 is None or frame5 is None:
#         continue
#     new_img = Image.new('RGB', (frame5.shape[1], frame5.shape[0]))
#     frame = np.concatenate((np.concatenate((frame1,frame2),axis=1),np.concatenate((frame3,frame4),axis=1),np.concatenate((frame5,new_img),axis=1)),axis=0)
#     frame = imutils.resize(frame, width=1000)

#     # if frame3 is None:
#     #     continue
#     if out is None:
#         out = cv2.VideoWriter("results/main_process_output.mp4", -1, 20.0, (frame.shape[1], frame.shape[0]))
        
#     out.write(frame)

#     cv2.imshow('Main process', frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == 27:
#             break

# if not out is None:
#     out.release()
# cv2.destroyAllWindows()

root_path = os.environ.get("root_path", "")
app = FastAPI(root_path=root_path)
# app = FastAPI()

@app.get("/")
async def home():
    return "Welcom to Lam's service"

@app.post("/add_point/{cam_id}")
async def add_point(cam_id: str, x: int, y: int):
    try:
        camera_process = process_of[cam_id]
        if camera_process.add_point(x, y):
            return True
        else:
            raise HTTPException(status_code=400, detail = "Unavailable point!")
    except (KeyError, AttributeError) as e:
        raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

@app.post("/clear_point/{cam_id}")
async def clear_point(cam_id: str):
    try:
        camera_process = process_of[cam_id]
        camera_process.clear_points()
        return True
    except (KeyError, AttributeError) as e:
        raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

@app.post("/comfirm_box_cam_before/{cam_id}")
async def comfirm_box_cam_before(cam_id: str):
    try:
        idx = list_cam_before.index(cam_id)
        if list_cam_before_process[idx].cam_before_comfirm_box():
            return True
        else:
            raise HTTPException(status_code=400, detail = "Choose 2 points to create box!")
    except ValueError:
        raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

# @app.post("/comfirm_box_cam_in/{cam_id}")
# async def comfirm_box_cam_in(cam_id: str):
#     try:
#         idx = list_cam_in.index(cam_id)
#         if list_cam_in_process[idx].comfirm_box():
#             return True
#         else:
#             raise HTTPException(status_code=400, detail = "Unavailable box!")
#     except ValueError:
#         raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

# @app.post("/comfirm_box_cam_out/{cam_id}")
# async def comfirm_box_cam_out(cam_id: str):
#     try:
#         idx = list_cam_out.index(cam_id)
#         if list_cam_out_process[idx].comfirm_box():
#             return True
#         else:
#             raise HTTPException(status_code=400, detail = "Unavailable box!")
#     except ValueError:
#         raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

@app.post("/comfirm_box_cam_in_out/{cam_id}")
async def comfirm_box_cam_in_out(cam_id: str):
    try:
        camera_process = process_of[cam_id]
        if camera_process.comfirm_box():
            return True
        else:
            raise HTTPException(status_code=400, detail = "Choose 2 points to create box!")
    except (KeyError, AttributeError) as e:
        raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

@app.post("/add_box_cam_before/{cam_id}")
async def add_box_cam_before(cam_id: str, x1: int, y1: int, x2: int, y2: int):
    try:
        idx = list_cam_before.index(cam_id)
        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
            list_cam_before_process[idx].cam_before_add_box(x1, y1, x2, y2)
            return True
        else:
            raise HTTPException(status_code=400, detail = "Unavailable point!")
    except ValueError:
        raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

@app.post("/del_box_cam_before/{cam_id}")
async def del_box_cam_before(cam_id: str, zone_id: int):
    try:
        idx = list_cam_before.index(cam_id)
        if list_cam_before_process[idx].del_box(zone_id):
            return True
        else:
            raise HTTPException(status_code=400, detail = f"Camera {cam_id} not have zone {zone_id}!")
    except ValueError:
        raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

# @app.post("/set_box_cam_in/{cam_id}")
# async def set_box_cam_in(cam_id: str, x1: int, y1: int, x2: int, y2: int):
#     try:
#         idx = list_cam_in.index(cam_id)
#         if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
#             list_cam_in_process[idx].set_box_detection(x1, y1, x2, y2)
#             return True
#         else:
#             raise HTTPException(status_code=400, detail = "Unavailable point!")
#     except ValueError:
#         raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

# @app.post("/set_box_cam_out/{cam_id}")
# async def set_box_cam_out(cam_id: str, x1: int, y1: int, x2: int, y2: int):
#     try:
#         idx = list_cam_out.index(cam_id)
#         if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
#             list_cam_out_process[idx].set_box_detection(x1, y1, x2, y2)
#             return True
#         else:
#             raise HTTPException(status_code=400, detail = "Unavailable point!")
#     except ValueError:
#         raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

@app.post("/set_box_cam_in_out/{cam_id}")
async def set_box_cam_in_out(cam_id: str, x1: int, y1: int, x2: int, y2: int):
    try:
        camera_process = process_of[cam_id]
        if x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1:
            camera_process.set_box_detection(x1, y1, x2, y2)
            return True
        else:
            raise HTTPException(status_code=400, detail = "Unavailable point!")
    except (KeyError, AttributeError) as e:
        raise HTTPException(status_code=400, detail = f"This function unavailable in camera {cam_id}!")

@app.post("/add_use_noti_token")
async def add_use_noti_token(token: str):
    if token:
        if not token in use_device_tokens.tokens:
            use_device_tokens.tokens.append(token)
            return True
        else:
            raise HTTPException(status_code=400, detail = "This token is existed!")
    else:
        raise HTTPException(status_code=400, detail = "This token is empty!")

@app.post("/del_use_noti_token")
async def del_use_noti_token(token: str):
    if token:
        if token in use_device_tokens.tokens:
            use_device_tokens.tokens.remove(token)
            return True
        else:
            raise HTTPException(status_code=400, detail = "This token is not existed!")
    else:
        raise HTTPException(status_code=400, detail = "This token is empty!")

@app.get("/get_size/{cam_id}")
async def add_box_cam_before(cam_id: str):
    if cam_id == "cam_all":
        return [1, 1]
    try:
        camera_process = process_of[cam_id]
        return [camera_process.camera_source.width, camera_process.camera_source.height]
    except:
        raise HTTPException(status_code=400, detail = f"Camera {cam_id} unavailable!")

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str, col: int = 3):
    if cam_id == "cam_all":
        return StreamingResponse(gen_all_cam(show_fps=False, col=col), media_type="multipart/x-mixed-replace; boundary=--frame")
    try:
        camera_process = process_of[cam_id]
        return StreamingResponse(gen(camera_process=camera_process, show_fps=False), media_type="multipart/x-mixed-replace; boundary=--frame")
    except:
        return StreamingResponse(gen_error_video(cam_id), media_type="multipart/x-mixed-replace; boundary=--frame")

@app.get("/video_feed/{cam_id}/fps")
async def video_feed(cam_id: str, col: int = 3):
    if cam_id == "cam_all":
        return StreamingResponse(gen_all_cam(show_fps=True, col=col), media_type="multipart/x-mixed-replace; boundary=--frame")
    try:
        camera_process = process_of[cam_id]
        return StreamingResponse(gen(camera_process=camera_process, show_fps=True), media_type="multipart/x-mixed-replace; boundary=--frame")
    except:
        return StreamingResponse(gen_error_video(cam_id), media_type="multipart/x-mixed-replace; boundary=--frame")

if __name__=="__main__":
    _host = os.environ.get("host", "localhost")
    uvicorn.run(app, host=_host, port = 8000, log_level = "info")
