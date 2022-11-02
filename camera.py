import cv2
import threading
import time

from frame_differencing import frame_diff
thread = None

class Camera:
    def __init__(self, max_fps=30, video_source=0, cam_name ="cam_name", allow_loop=False, motion_detect_enabled=False):
        """
        - fps: Rate at which frames are read from video_source
        - video_source: The video_source to read frames from. Defaulted to 0 (webcam). Anything that can be used in cv2.VideoCapture
        - allow_loop: Set to True to allow looping on video files. This turns those files into endless stream
        """
        self.fps = max_fps
        self.max_fps = max_fps
        self.video_source = cv2.VideoCapture(video_source)
        self.cam_name = cam_name
        # We want a max of 1s history to be stored, thats 3s*fps
        self.max_frames = 1*self.fps
        self.frames = []
        self.isrunning = False
        self.start_time = time.time()
        self.end_tine = time.time()

        self.sizeStr = str(int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))) + 'x' + str(int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.sizeStrConcat = str(int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH)*2)) + 'x' + str(int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.allow_loop = allow_loop


        # self.default_error_image = cv2.imread("images/500-err.jpg")

        self.last_frame = None
        self.motion_detect_enabled = motion_detect_enabled
        self.motion_detected = False
        self.motion_lock_counter = 0
        self.motion_detect_skipped_frame=-1
        self.motion_detect_frame_skips=5

        # Define video file output
        self.width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.video_source.get(cv2.CAP_PROP_FPS))
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.out = cv2.VideoWriter("results/output.avi", self.codec, self.fps, (self.width, self.height))

        self.ret_out = None
        self.frame_out = None
        
        # self.run()

    def run(self):
        global thread
        global subthread1
        thread = threading.Thread(target=self._capture_loop,daemon=True)
        # subthread1 = threading.Thread(target=self._lowlight_enhance_loop,daemon=True)
        if not self.isrunning:
            v, img = self.video_source.read()        
            if v:
                self.frames.append(img)
            self.isrunning = True
            thread.start()
            # subthread1.start()
            print(f"A camera {self.cam_name} run! time: {time.time()}")
        else:
            print(f"A camera {self.cam_name} thread is running already!")

    def _capture_loop(self):
        dt = 1/self.fps
        self.frames_read = 0
        while self.isrunning:
            self.frames_read+=1
            v, img = self.video_source.read()
            if v:
                if len(self.frames) >= 2:
                    self.frames = self.frames[1:]
                self.frames.append(img)
            elif(self.allow_loop):
                print(f"camera.py {self.cam_name}: End of Video. Loop from start, time: {time.time()}")
                self.video_source.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                self.frames = self.frames[1:]

            
            if (self.motion_detect_enabled):
                # self.detect_motion()
                if (self.motion_detect_skipped_frame >= self.motion_detect_frame_skips):
                    self.detect_motion()
                    self.motion_detect_skipped_frame = 0
                else:
                    self.motion_detect_skipped_frame += 1

            time.sleep(dt)

    def detect_motion(self):
        start = time.time()
        current_frame = self.get_raw_frame()
        if self.last_frame is None:
            self.last_frame = current_frame
            return 

        if current_frame is not None:
            resized_frame = cv2.resize(current_frame, (300,300), interpolation = cv2.INTER_AREA)
            if self.motion_lock_counter > 0:
                self.motion_lock_counter -= 1
                self.current_frame = None
            elif(frame_diff(self.last_frame, current_frame)):
                self.motion_detected = True
                self.motion_lock_counter = 10
            else:
                self.motion_detected = False
                self.current_frame = None
                
        
        self.last_frame = current_frame

    def read(self):
        if len(self.frames) > 0:
            return True, self.frames[0]
        else:
            return False, None

    def set_frame_skip(self,frameskip):
        '''Set how many frame is skipped before a frame is inferenced by model'''
        self.frame_skip = frameskip

    def set_video_output(self, output):
        self.out = cv2.VideoWriter(output, self.codec, self.fps, (self.width, self.height))

    def set_motion_detect(self,motion_detect_enabled):
        self.motion_detect_enabled = motion_detect_enabled

    def stop(self):
        self.isrunning = False

    def attach_fps(self, frame):
        return cv2.putText(frame, 'FPS: ' + str(self.get_fps()), (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)

    def attach_motion_text(self, frame):
        new_frame = frame.copy()
        return cv2.putText(new_frame, 'motion: ' + str(self.motion_detected), (10, 450), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0, 255, 0), 2, cv2.LINE_AA)

    def encode_to_png(self, frame):
        return cv2.imencode('.png', frame)[1].tobytes()

    def encode_to_jpg(self, frame):
        return cv2.imencode('.jpg', frame)[1].tobytes()

    def get_raw_frame(self):
        if len(self.frames) > 0:
            return self.frames[0]
    
    def has_frame(self):
        if len(self.frames) > 0:
            return True
        return False

    def get_fps(self):
        return self.fps
        
    def get_sizestr(self):
        return self.sizeStr

    def get_sizestrConcat(self):
        return self.sizeStrConcat

    def write_frame_to_output_file(self,frame):
        '''Write frame to out.mp4'''
        self.out.write(frame)

if __name__ == '__main__':
    import imutils
    import numpy as np
    from PIL import Image
    camera1 = Camera(30,"https://4311-14-248-84-166.ap.ngrok.io/cam11")
    camera2 = Camera(30,"https://4311-14-248-84-166.ap.ngrok.io/cam12")
    camera3 = Camera(30,"https://4311-14-248-84-166.ap.ngrok.io/cam13")
    camera4 = Camera(30,"https://4311-14-248-84-166.ap.ngrok.io/cam14")
    camera5 = Camera(30,"https://4311-14-248-84-166.ap.ngrok.io/cam15")
    camera1.run()
    camera2.run()
    camera3.run()
    camera4.run()
    camera5.run()
    start_time = time.time()
    

    while True:
        ret1, frame1 = camera1.read()
        ret2, frame2 = camera2.read()
        ret3, frame3 = camera3.read()
        ret4, frame4 = camera4.read()
        ret5, frame5 = camera5.read()
        
        end_time_1 = time.time()

        if not (ret1 and ret2 and ret3 and ret4 and ret5):
            break 
        frame1 = imutils.resize(frame1, width=800)
        frame2 = imutils.resize(frame2, width=800)
        frame3 = imutils.resize(frame3, width=800)
        frame4 = imutils.resize(frame4, width=800)
        frame5 = imutils.resize(frame5, width=800)
        new_img = Image.new('RGB', (frame5.shape[1], frame5.shape[0]))

        frame = np.concatenate((np.concatenate((frame1,frame2),axis=1),np.concatenate((frame3,frame4),axis=1),np.concatenate((frame5,new_img),axis=1)),axis=0)

        frame = imutils.resize(frame, width=800)

        fps1 = 1/(end_time_1 - start_time + 0.001)
        end_time = time.time()
        fps = 1/(end_time - start_time + 0.001)
        start_time = end_time

        cv2.putText(frame, "{:.1f} {:.1f}".format(fps, fps1), (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow('Cam1', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
                break