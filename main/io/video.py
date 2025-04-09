from queue import Queue
import threading

import cv2


class VideoCapture:
    def __init__(self, target: int = 0, cam_width = None, cam_height = None, drop_if_full = True):
        self.stopped = False
        self.drop_if_full = drop_if_full
        self.lock = threading.Lock()
        self.stop_cam_thread = threading.Event()
        self.Q = Queue(maxsize=2)
        self.cap = cv2.VideoCapture(target)
        if not self.cap.isOpened():
            print(f'could not open {target}')
            return
        if cam_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        if cam_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
        self.t = threading.Thread(target=self._reader)
        self.t.start()

    def _reader(self):
        while not self.stop_cam_thread.is_set():
            if self.drop_if_full or not self.Q.full():
                if self.Q.full():
                    _ = self.Q.get()  # remove value for new ones

                with self.lock:
                    ret, image = self.cap.read()
                    if not ret:
                        print('could not get image from VideoCapture')
                        self.stop_cam_thread.set()
                        self.cap.release()

                self.Q.put(image)

    def read(self):
        if self.stop_cam_thread.is_set():
            ret = False
            img = None
        elif not self.Q.empty():
            ret = True
            img = self.Q.get(timeout=2)
        else:
            ret = True
            img = None
        return ret, img

    def isOpened(self):
        with self.lock:
            ret = self.cap.isOpened()
        return ret

    def set(self, *args):
        with self.lock:
            self.cap.set(*args)

    def release(self):
        if self.isOpened():
            self.stop_cam_thread.set()
            self.t.join()
        self.cap.release()
