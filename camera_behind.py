import cv2
import threading
import time
import numpy as np
import imutils

from numpy import expand_dims
from customer_data import CustomerData
from camera import Camera

thread = None

class CameraBehind:    
    def __init__(self, camera_source=None, model=None, detector=None, customer_data = None):
        self.camera_source = camera_source
        self.isrunning = False
        self.model = model
        self.detector = detector
        self.customer_data = customer_data
        self.margin = 60
        self.fps = 0
        self.last_frame = None
        self.last_customer_id = -1

    def run(self):
        global thread
        thread = threading.Thread(target=self._processing,daemon=True)
        if not self.isrunning:
            self.isrunning = True
            thread.start()
            print("Process camera behind run!")
        else:
            print("Process camera behind is running already!")

    def _processing(self):
        start_time = time.time()
        while self.isrunning:
            ret, frame = self.camera_source.read()

            if not ret:
                continue

            frame = imutils.resize(frame, width=800)

            if self.camera_source.motion_detected:
                boxes, faces = self.detect_face(img=frame)
                if not(faces is None or len(faces)==0):
                    for idx, face in enumerate(faces):
                        if self.customer_data.face_ebd_datas.count == 0:
                            frame = self.draw_face(-1, boxes[idx], frame)
                            continue
                        else:
                            if face is None:
                                continue
                            face_ebd = self.get_embedding(face_pixels=face)
                            new_face = True
                            face_id = -1
                            min_dist = 100
                            for face_idx, face_ebd_data in enumerate(self.customer_data.face_ebd_datas):
                                point1 = np.array(self.get_avg_ebd(face_ebd_data))
                                point2 = np.array(face_ebd)
                                dist = np.linalg.norm(point1 - point2)
                                # print(f'{dist} ', end='')
                                if dist < 6:
                                    new_face = False
                                    
                                    if dist < 5.5 and dist < min_dist:
                                        min_dist = dist                                                               
                                        face_id = face_idx       
                            frame = self.draw_face(face_id, boxes[idx], frame)
                            if face_id != -1:
                                self.last_customer_id = face_id
            else:
                cv2.putText(frame, "detect off", (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                self.last_customer_id = -1
                time.sleep(0.7/self.camera_source.fps)                
            
            self.last_frame = frame
            
            end_time = time.time()                
            self.fps = 1/(end_time - start_time + 0.001)
            start_time = end_time    

    def get_embedding(self, face_pixels):
        face_pixels = cv2.resize(face_pixels, (160, 160))
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = self.model.predict(samples)
        return yhat[0]

    def detect_face(self, img):
        boxs, p = self.detector.detect(img)
        # boxs, p = self.detector.detect_face(img, 0.7)
        if boxs is None or len(boxs) == 0:
            return None, None
        faces = list()
        boxes_result = list()
        for box in boxs:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
            if x < 0 or y < 0 or x+w > img.shape[1] or y+h >img.shape[0]:
                continue
            if w + h >= self.margin:
                faces.append(img[y:y + h, x:x + w, :])
                boxes_result.append((int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        return boxes_result, faces

    def draw_face(self, face_id, box, frame):
        if face_id == -1 or self.customer_data.checkout_list[face_id]:
            bgr = (0, 255, 0)
        else:
            bgr = (0, 0, 255)
        (x1, y1, x2, y2) = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(frame, f'[{face_id}]', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def get_avg_ebd(self, ebd_list):
        sum = 0
        for ebd in ebd_list:
            sum += ebd
        return sum/len(ebd_list)

    def on_take_item_event(self):
        if self.last_customer_id != -1:
            self.customer_data.checkout_list[self.last_customer_id] = False

    def get_frame(self):
        return cv2.putText(self.last_frame, "{}".format(self.camera_source.cam_name), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    def get_frame_with_fps(self):
        return cv2.putText(self.get_frame(), "{:.1f}".format(self.fps), (self.last_frame.shape[1]-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
