import threading
import time

class MyTrackers:
    def __init__(self):
        self.__tracker_list = list()
        self.__success_list = list()
        self.__box_list = list()
        self.__track_done = 0

    def add(self, tracker, frame, box):
        tracker.init(frame, box)
        self.__tracker_list.append(tracker)
        self.__success_list.append(False)
        self.__box_list.append((0, 0, 0, 0))

    def update_after_track(self):
        self.__track_done += 1

    def __track_run(self, idx, tracker, frame):
        (success, box) = tracker.update(frame)
        self.__success_list[idx] = success
        if success:
            self.__box_list[idx] = box
        self.update_after_track()

    def update(self, frame):
        #TODO
        self.__track_done = 0
        for idx, tracker in enumerate(self.__tracker_list):
            processThread = threading.Thread(target=self.__track_run, args=(idx, tracker, frame))
            processThread.start()
        
        while self.__track_done != len(self.__tracker_list):
            time.sleep(0.001)

        success = True
        for obj in self.__success_list:
            if not obj:
                success = False
                break
        
        return success, self.__box_list