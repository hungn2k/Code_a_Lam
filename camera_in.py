import cv2
import threading
import time
import numpy as np
import imutils

from numpy import expand_dims
from customer_data import CustomerData
from camera import Camera

thread = None

class CameraIn:    
    def __init__(self, camera_source=None, model=None, detector=None, customer_data = None):
        self.camera_source = camera_source
        self.isrunning = False
        self.model = model
        self.detector = detector
        self.customer_data = customer_data
        self.margin = 60
        self.width_scale = 800
        self.fps = 0
        self.last_frame = None
        self.box_detect = [0, 0, self.camera_source.width, self.camera_source.height]
        self.box_detect = [360, 180, 780, 500]
        self.box_detect_scale = [int(v*self.width_scale/self.camera_source.width) for v in self.box_detect]
        self.points = list()
        self.points_scale = list()

    def run(self):
        global thread
        thread = threading.Thread(target=self._processing,daemon=True)
        if not self.isrunning:
            self.isrunning = True
            thread.start()
            print("Process camera in run!")
        else:
            print("Process camera in is running already!")

    def _processing(self):
        start_time = time.time()
        while self.isrunning:
            ret, frame = self.camera_source.read()

            if not ret:
                continue

            frame = imutils.resize(frame, width=self.width_scale)

            if self.camera_source.motion_detected:
                boxes, faces = self.detect_face(img=frame)
                if not(faces is None or len(faces)==0):
                    for idx, face in enumerate(faces):
                        if self.face_is_in_box(boxes[idx]):
                            if face is None:
                                continue
                            face_ebd = self.get_embedding(face_pixels=face)
                            if self.customer_data.face_ebd_datas.count == 0:
                                face_ebd_list = list()
                                face_ebd_list.append(face_ebd)
                                self.customer_data.face_ebd_datas.append(face_ebd_list)
                                self.customer_data.checkout_list.append(True)
                                frame = self.draw_face(len(self.customer_data.face_ebd_datas)-1, boxes[idx], frame)
                                continue
                            else:
                                new_face = True
                                face_id = -1
                                min_dist = 100
                                for face_idx, face_ebd_data in enumerate(self.customer_data.face_ebd_datas):
                                    point1 = np.array(self.get_avg_ebd(face_ebd_data))
                                    point2 = np.array(face_ebd)
                                    dist = np.linalg.norm(point1 - point2)
                                    # print(f'{dist} ', end='')
                                    if dist < 5.5:
                                        new_face = False
                                        
                                        if dist < 5 and dist < min_dist:
                                            min_dist = dist                                                               
                                            face_id = face_idx
                                if face_id != -1:
                                    self.customer_data.face_ebd_datas[face_idx].append(face_ebd)         
                                if new_face:
                                    face_ebd_list = list()
                                    face_ebd_list.append(face_ebd)
                                    self.customer_data.face_ebd_datas.append(face_ebd_list)
                                    self.customer_data.checkout_list.append(True)
                                    frame = self.draw_face(len(self.customer_data.face_ebd_datas)-1, boxes[idx], frame)
                                else:
                                    frame = self.draw_face(face_id, boxes[idx], frame)
                        else:
                            frame = self.draw_face(-1, boxes[idx], frame)
            else:
                cv2.putText(frame, "detect off", (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                time.sleep(0.7/self.camera_source.fps)    
            
            (x1, y1, x2, y2) = [int(v) for v in self.box_detect_scale]
            self.last_frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if len(self.points_scale) != 0:
                for xy in self.points_scale:
                    cv2.circle(frame, (xy[0], xy[1]), 5, (255,0,0), -1)
            
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
    
    def face_is_in_box(self, face_box):
        (x1, y1, x2, y2) = [int(v) for v in self.box_detect_scale]
        (fx1, fy1, fx2, fy2) = [int(v) for v in face_box]
        if x1 <= fx1 and y1 <= fy1 and x2 >= fx2 and y2 >= fy2:
            return True
        return False

    def set_box_detection(self, x1, y1, x2, y2):
        self.box_detect = [x1, y1, x2, y2]
        self.box_detect_scale = [int(v*self.width_scale/self.camera_source.width) for v in self.box_detect]

    def add_point(self, x, y):
        if len(self.points) < 2:
            if x<0 or y<0 or x>self.camera_source.width or y>self.camera_source.height:
                return False
            self.points.append((x, y))
            x = int(x*self.width_scale/self.camera_source.width)
            y = int(y*self.width_scale/self.camera_source.width)
            self.points_scale.append((x, y))
            return True
        else:
            return False

    def clear_points(self):
        self.points.clear()
        self.points_scale.clear()

    def comfirm_box(self):
        if len(self.points) == 2:
            x1 = self.points[0][0]
            y1 = self.points[0][1]
            x2 = self.points[1][0]
            y2 = self.points[1][1]
            if x1>x2 and y1>y2: 
                x1,x2 = x2,x1
                y1,y2 = y2,y1

            if x1>x2 and y1<y2:
                x1,x2 = x2,x1

            if x1<x2 and y1>y2:
                y1,y2 = y2,y1

            if x1<x2 and y1<y2:
                self.set_box_detection(x1, y1, x2, y2)
                self.clear_points()
                return True
            else:
                self.clear_points()
                return False
        else:
            self.clear_points()
            return False
        
    def get_frame(self):
        return cv2.putText(self.last_frame, "{}".format(self.camera_source.cam_name), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    def get_frame_with_fps(self):
        return cv2.putText(self.get_frame(), "{:.1f}".format(self.fps), (self.last_frame.shape[1]-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
