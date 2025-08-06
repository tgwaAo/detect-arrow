from queue import Queue
import threading

import cv2

from typing import Optional as Opt
from typing import Any
import numpy as np
import numpy.typing as npt


class VideoCapture:
    def __init__(
        self,
        target: int = 0,
        cam_width: Opt[int] = None,
        cam_height: Opt[int] = None,
        drop_if_full:bool = True
    ) -> None:
        self.stopped = False
        self.drop_if_full = drop_if_full
        self.lock = threading.Lock()
        self.stop_cam_thread = threading.Event()
        self.queue: Queue = Queue(maxsize=2)
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

    def _reader(self) -> None:
        while not self.stop_cam_thread.is_set():
            if self.drop_if_full or not self.queue.full():
                if self.queue.full():
                    _ = self.queue.get()  # remove value for new ones

                with self.lock:
                    ret, image = self.cap.read()
                    if not ret:
                        print('could not get image from VideoCapture')
                        self.stop_cam_thread.set()
                        self.cap.release()

                self.queue.put(image)

    def read(self) -> tuple[bool, Opt[npt.NDArray[np.uint8]]]:
        if self.stop_cam_thread.is_set():
            ret = False
            img = None
        elif not self.queue.empty():
            ret = True
            img = self.queue.get(timeout=2)
        else:
            ret = True
            img = None
        return ret, img

    def is_opened(self) -> bool:
        with self.lock:
            ret = self.cap.isOpened()
        return ret

    def set(self, *args: Any) -> None:
        with self.lock:
            self.cap.set(*args)

    def release(self) -> None:
        if self.is_opened():
            self.stop_cam_thread.set()
            self.t.join()
        self.cap.release()
