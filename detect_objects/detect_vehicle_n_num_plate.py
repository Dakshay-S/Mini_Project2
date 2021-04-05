import numpy as np
import cv2
from imutils.video import FPS
from queue import Queue
from threading import Thread
import time

VIDEO_PATH = 'videos/white_car.mp4'
CLASS_FILE = 'weights_n_config/obj.names'
CONFIG_FILE = 'weights_n_config/obj.cfg'
YOLO_WEIGHTS_FILE = 'weights_n_config/obj.weights'
MIN_CONFIDENCE = 0.4
SHOW_DETECTION = True


# reduce BLOB_SIZE if GPU is not available
BLOB_SIZE = 544
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# thread queue
MAX_THREAD_Q_SZ = 10
frame_queue = Queue(MAX_THREAD_Q_SZ)
blob_queue = Queue(MAX_THREAD_Q_SZ)
dnn_output_queue = Queue(MAX_THREAD_Q_SZ)
non_max_supp_queue = Queue(MAX_THREAD_Q_SZ)

GPU_THERE = (cv2.cuda.getCudaEnabledDeviceCount() > 0)


if GPU_THERE:
    BACKEND = cv2.dnn.DNN_BACKEND_CUDA
    TARGET = cv2.dnn.DNN_TARGET_CUDA
    print("OpenCV can work with GPU ")
else:
    BACKEND = cv2.dnn.DNN_BACKEND_OPENCV
    TARGET = cv2.dnn.DNN_TARGET_CPU
    print("OpenCV is not installed with GPU support ")


LABELS = open(CLASS_FILE).read().strip().split("\n")

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                           dtype="uint8")

DNN = cv2.dnn.readNetFromDarknet(CONFIG_FILE, YOLO_WEIGHTS_FILE)

DNN.setPreferableBackend(BACKEND)

DNN.setPreferableTarget(TARGET)

video_source = cv2.VideoCapture(VIDEO_PATH)
# video_source = cv2.VideoCapture(0)

fps = FPS()

# wont work for webcam
good_frame, f = video_source.read()
if good_frame:
    HEIGHT = f.shape[0]
    WIDTH = f.shape[1]
else:
    raise Exception("Can't read video")


layers_name = DNN.getLayerNames()
out_layers_name = [layers_name[i[0] - 1] for i in DNN.getUnconnectedOutLayers()]


# todo: delete below variable
num_plates = []


def make_blob_from_frame(video_feed):
    while True:
        (valid, frame) = video_feed.read()
        if not valid:
            break

        frame_queue.put(frame)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (BLOB_SIZE, BLOB_SIZE),
                                     swapRB=True, crop=False)
        blob_queue.put(blob)


def all_processing(net):
    while True:
        try:
            curr_frame = frame_queue.get(timeout=1)
            blob = blob_queue.get(timeout=1)
        except:
            break

        net.setInput(blob)
        layerOutputs = net.forward(out_layers_name)

        rects = []
        probabilities = []
        classCodes = []

        for layer_output in layerOutputs:

            for detected_obj in layer_output:
                if detected_obj[4] < 0.7:
                    continue

                obj_confidence = detected_obj[5:]
                class_code = np.argmax(obj_confidence)
                max_confidence = obj_confidence[class_code]

                if max_confidence > MIN_CONFIDENCE:
                    rect = detected_obj[0:4] * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
                    (centerX, centerY, width, height) = rect.astype("int")

                    top_left_x = int(centerX - (width / 2))
                    top_left_y = int(centerY - (height / 2))

                    rects.append([top_left_x, top_left_y, int(width), int(height)])
                    probabilities.append(float(max_confidence))
                    classCodes.append(class_code)

        indices = cv2.dnn.NMSBoxes(rects, probabilities, MIN_CONFIDENCE,
                                   MIN_CONFIDENCE)

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (rects[i][0], rects[i][1])
                (w, h) = (rects[i][2], rects[i][3])

                obj_type = LABELS[classCodes[i]]

                if obj_type == 'carLP' or obj_type == 'truckLP' or obj_type == 'busLP' or obj_type == 'motorcycleLP' or obj_type == 'autoLP':
                    pass
                    num_plate = curr_frame[y:y + h, x:x + w]#.copy()
                    num_plates.append(num_plate)
                # cv2.imshow(f'{x + y + h + w}', num_plate)

                if SHOW_DETECTION:
                    bgr = [int(c) for c in COLORS[classCodes[i]]]
                    cv2.rectangle(curr_frame, (x, y), (x + w, y + h), bgr, 2)
                    class_name = "{}: {:.4f}".format(obj_type, probabilities[i])
                    cv2.putText(curr_frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, bgr, 2)

        del_y = 30
        cv2.putText(curr_frame, f"frame_q : {frame_queue.qsize()}", (20, 1 * del_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
        cv2.putText(curr_frame, f"blob_q : {blob_queue.qsize()}", (20, 2 * del_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        # todo: give as input to  gui
        cv2.imshow("output", cv2.resize(curr_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
        fps.update()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


blob_thread = Thread(target=make_blob_from_frame, args=(video_source,), daemon=True)
blob_thread.start()

fps = FPS().start()
all_processing_thread = Thread(target=all_processing, args=(DNN,), daemon=True)
all_processing_thread.start()
blob_thread.join()

fps.stop()
print("Elasped time: {:.2f}".format(fps.elapsed()))
print("Approx. FPS: {:.2f}".format(fps.fps()))

print("the end")

# cv2.destroyWindow("DETECTIONS")

cv2.destroyAllWindows()

for plate in num_plates:

    cv2.imshow('plate', plate)
    key = cv2.waitKey(1000) & 0xFF
    if key == ord("n"):
        continue


video_source.release()
