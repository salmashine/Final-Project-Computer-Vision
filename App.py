import cv2
from ultralytics import YOLO
import time

model = YOLO("best.pt")

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

while True:
    ret, video = cam.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    _time_mulai = time.time()
    result = model.predict(video, show=True, conf=0.5)

    print("Preprocessing time:", time.time() - _time_mulai, "seconds")
    _key = cv2.waitkey(1)
    if _key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()