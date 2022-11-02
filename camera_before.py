import cv2
import threading
import time
import numpy as np
import imutils
import torch
import PythonFCM.FCMManager as fcm

from camera import Camera
from my_tracker import MyTrackers

thread = None

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.legacy.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create
}

class CameraBefore:    
    def __init__(self, camera_source=None, model=None, thres=0.7, tracker="crst", camera_behind_process=None, delay_frame=600, use_device_tokens=None):
        self.camera_source = camera_source
        self.camera_behind_process = camera_behind_process
        self.isrunning = False
        self.model = model
        self.width_scale = 800
        self.fps = 0
        self.last_frame = None
        # self.box_detect = [0, 0, self.camera_source.width, self.camera_source.height]
        # self.box_detect = [360, 180, 780, 500]
        # self.box_detect_scale = [int(v*self.width_scale/self.camera_source.width) for v in self.box_detect]
        self.classes = self.model.names
        self.delay_frame = delay_frame
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.thres = thres
        self.tracker = tracker
        self.zones = list()
        self.zones_scale = list()
        self.box_avgs = list()
        self.use_device_tokens = use_device_tokens
        self.points = list()
        self.points_scale = list()
        self.count_frame_empty = list()

    def run(self):
        global thread
        thread = threading.Thread(target=self._processing,daemon=True)
        if not self.isrunning:
            self.isrunning = True
            thread.start()
            print("Process camera before run!")
        else:
            print("Process camera before is running already!")

    def _processing(self):
        start_time = time.time()
        count = self.delay_frame
        while self.isrunning:
            ret, frame = self.camera_source.read()

            if not ret:
                continue

            frame = imutils.resize(frame, width=self.width_scale)

            if self.camera_source.motion_detected or count == self.delay_frame:
                if count == self.delay_frame:
                    count = 0
                    results = self.score_frame(frame)
                    # frame = self.plot_boxes(results, frame)

                    trackers = MyTrackers()
                    self.box_avgs = list()
                    labels, cord = results
                    n = len(labels)
                    x_shape, y_shape = frame.shape[1], frame.shape[0]
                    for i in range(n):
                        row = cord[i]
                        if row[4] >= self.thres:
                            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                            box = (x1, y1, x2 - x1, y2 - y1)

                            # create a new object tracker for the bounding box and add it
                            # to our multi-object tracker
                            tracker = OPENCV_OBJECT_TRACKERS[self.tracker]()
                            trackers.add(tracker, frame, box)

                            self.box_avgs.append(((x1 + x2)/2, (y1 + y2)/2, labels[i], row[4]))
                            print(row[4])

                # grab the updated bounding box coordinates (if any) for each
                # object that is being tracked
                (success, boxes) = trackers.update(frame)

                # loop over the bounding boxes and draw then on the frame
                for idx, box in enumerate(boxes):
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    (x_avg, y_avg, label, thres) = [v for v in self.box_avgs[idx]]
                    # print(type(x_avg), type(y_avg), type(label), type(thres))
                    cv2.putText(frame, self.class_to_label(label) + f'({thres:.2f})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if (x, y, w, h) != (0, 0, 0, 0) and abs((x + w / 2) - x_avg) > w/2 or abs((y + h / 2) - y_avg > h/2):
                        if not self.camera_behind_process is None:
                            self.camera_behind_process.on_take_item_event()
                        cv2.putText(frame, f'item taken event triggered', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        if count < self.delay_frame - 20:
                            count = self.delay_frame - 20
                    if (x, y, w, h) != (0, 0, 0, 0):
                        x_avg = (x_avg * 19 + (x + w / 2)) / 20
                        y_avg = (y_avg * 19 + (y + h / 2)) / 20
                        self.box_avgs[idx] = (x_avg, y_avg, label, thres)
            
            else:
                time.sleep(0.7/self.camera_source.fps) 
                cv2.putText(frame, "detect off", (150,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                # count = self.delay_frame - 1
            
            for idx, zone in enumerate(self.zones_scale):
                frame = self.drwa_zone(frame, idx, zone)

            for idx, zone in enumerate(self.zones_scale):
                    flag = self.check_zone_is_empty(zone, boxes)
                    if flag:
                        cv2.putText(frame, f'Zone {int(idx)} is empty', (5,20*(idx+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                        self.count_frame_empty[idx]+=1
                        if self.count_frame_empty[idx] > 1800:
                            #TODO
                            print("send noti empty zone")
                            self.count_frame_empty[idx] = 0
                            fcm.sendPush("Out of stock", f'Zone {int(idx)} in camera {self.camera_source.cam_name} is empty', self.use_device_tokens.tokens)
                    else:
                        cv2.putText(frame, f'Zone {int(idx)} is oke', (5,20*(idx+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                        self.count_frame_empty[idx]=0

            if len(self.points_scale) != 0:
                for xy in self.points_scale:
                    cv2.circle(frame,(xy[0], xy[1]), 5, (255,0,0), -1)

            self.last_frame = frame
            count += 1

            end_time = time.time()                
            self.fps = 1/(end_time - start_time + 0.001)
            start_time = end_time
    
    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)

        frame = frame.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        frame = np.ascontiguousarray(frame)

        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= self.thres:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]) + f'({row[4]:.1f})', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame
    
    def drwa_zone(self, frame, idx, zone):
        cv2.rectangle(frame, (zone[0], zone[1]), (zone[2], zone[3]), (255, 255, 255), 2)
        cv2.putText(frame, f'Zone {int(idx)}', (zone[0],zone[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def check_zone_is_empty(self, zone, boxes):
        for box in boxes:
            flag = True
            (x, y, w, h) = [int(v) for v in box]
            if zone[0] > (x + w / 2) or (x + w / 2) > zone[2] or zone[1] > (y + h / 2) or (y + h / 2) > zone[3]:
                flag = False
            if flag:
                return False
        return True

    def cam_before_add_box(self, x1, y1, x2, y2):
        zone = (x1, y1, x2, y2)
        self.zones.append(zone)
        x1 = int(x1*self.width_scale/self.camera_source.width)
        y1 = int(y1*self.width_scale/self.camera_source.width)
        x2 = int(x2*self.width_scale/self.camera_source.width)
        y2 = int(y2*self.width_scale/self.camera_source.width)
        _zone = (x1, y1, x2, y2)
        self.zones_scale.append(_zone)
        self.count_frame_empty.append(0)

    def del_box(self, id):
        if id < len(self.zones):
            del self.zones[id]
            del self.zones_scale[id]
            del self.count_frame_empty[id]
            return True
        else:
            return False

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

    def cam_before_comfirm_box(self):
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
                self.cam_before_add_box(x1, y1, x2, y2)
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