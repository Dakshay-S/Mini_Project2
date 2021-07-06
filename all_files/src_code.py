import numpy as np
import cv2
from imutils.video import FPS
from queue import Queue
from threading import Thread
import time
import imagehash
from PIL import Image
import sqlite3

VIDEO_PATH = '../test_videos/white_car.mp4'
DETECTOR_CLASS_FILE = '../cnn_labels/detector.names'
READER_CLASS_FILE = '../cnn_labels/plate_reader.names'

DETECTOR_CONFIG_FILE = '../cfgs/detector.cfg'
READER_CONFIG_FILE = '../cfgs/plate_reader.cfg'

DETECTOR_WEIGHTS_FILE = '../cnn_weights/detector.weights'
READER_WEIGHTS_FILE = '../cnn_weights/plate_reader.weights'

DATABASE_FILE = '../database/vehicles.db'

DETECTOR_MIN_CONFIDENCE = 0.4
READER_MIN_CONFIDENCE = 0.4

SHOW_DETECTION = True

# reduce BLOB_SIZE if GPU is not available
BLOB_SIZE = 544
PLATE_BLOB_SIZE = 416
DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 800

# thread queue
DETECTOR_MAX_THREAD_Q_SZ = 10
detector_frame_queue = Queue(DETECTOR_MAX_THREAD_Q_SZ)
detector_blob_queue = Queue(DETECTOR_MAX_THREAD_Q_SZ)

reader_frame_queue = Queue()
reader_blob_queue = Queue()

DETECTOR_LABELS = open(DETECTOR_CLASS_FILE).read().strip().split("\n")
READER_LABELS = open(READER_CLASS_FILE).read().strip().split("\n")

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(DETECTOR_LABELS), 3),
                           dtype="uint8")

num_plate_detector_DNN = cv2.dnn.readNetFromDarknet(DETECTOR_CONFIG_FILE, DETECTOR_WEIGHTS_FILE)
num_plate_reader_DNN = cv2.dnn.readNetFromDarknet(READER_CONFIG_FILE, READER_WEIGHTS_FILE)

# if using CPU change to: cv2.dnn.DNN_BACKEND_OPENCV
num_plate_detector_DNN.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
num_plate_reader_DNN.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)

# if using CPU change to : cv2.dnn.DNN_TARGET_CPU
num_plate_detector_DNN.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
num_plate_reader_DNN.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

DATABASE_CONNECTION = None  # instantiated in reader thread
DATABASE_CURSOR = None  # instantiated in reader thread
command_create_table = "CREATE TABLE IF NOT EXISTS VEHICLE_INFO(" \
                       "TIMESTAMP TEXT PRIMARY KEY, LICENSE_NUM TEXT, BEST_IMAGE BLOB,  DIRECTION TEXT CHECK (DIRECTION IN ('IN','OUT')) )"

video_source = cv2.VideoCapture(VIDEO_PATH)
# video_source = cv2.VideoCapture(0)

fps = FPS()

layers_name = num_plate_detector_DNN.getLayerNames()
detector_out_layers_name = [layers_name[i[0] - 1] for i in num_plate_detector_DNN.getUnconnectedOutLayers()]

layers_name = num_plate_reader_DNN.getLayerNames()
reader_out_layers_name = [layers_name[i[0] - 1] for i in num_plate_reader_DNN.getUnconnectedOutLayers()]

# wont work for webcam
good_frame, f = video_source.read()
if good_frame:
    HEIGHT = f.shape[0]
    WIDTH = f.shape[1]
else:
    raise Exception("Can't read video")

# todo: delete below variable
num_plates = []


def make_blob_from_frame(video_feed):
    while True:
        (valid, frame) = video_feed.read()
        if not valid:
            break

        detector_frame_queue.put(frame)

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (BLOB_SIZE, BLOB_SIZE),
                                     swapRB=True, crop=False)
        detector_blob_queue.put(blob)


def all_processing(net):
    global tracker
    while True:
        try:
            curr_frame = detector_frame_queue.get(timeout=1)
            blob = detector_blob_queue.get(timeout=1)
        except:
            break

        net.setInput(blob)
        layerOutputs = net.forward(detector_out_layers_name)

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

                if max_confidence > DETECTOR_MIN_CONFIDENCE:
                    rect = detected_obj[0:4] * np.array([WIDTH, HEIGHT, WIDTH, HEIGHT])
                    (centerX, centerY, width, height) = rect.astype("int")

                    top_left_x = int(centerX - (width / 2))
                    top_left_y = int(centerY - (height / 2))

                    rects.append([top_left_x, top_left_y, int(width), int(height)])
                    probabilities.append(float(max_confidence))
                    classCodes.append(class_code)

        indices = cv2.dnn.NMSBoxes(rects, probabilities, DETECTOR_MIN_CONFIDENCE,
                                   DETECTOR_MIN_CONFIDENCE)

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (rects[i][0], rects[i][1])
                (w, h) = (rects[i][2], rects[i][3])

                obj_type = DETECTOR_LABELS[classCodes[i]]

                if obj_type == 'carLP' or obj_type == 'truckLP' or obj_type == 'busLP' or obj_type == 'motorcycleLP' or obj_type == 'autoLP':
                    # todo: is copying image needed
                    num_plate = curr_frame[y:y + h, x:x + w].copy()

                    num_plates.append(num_plate)
                    reader_frame_queue.put(((x, y), num_plate))
                # cv2.imshow(f'{x + y + h + w}', num_plate)

                if SHOW_DETECTION:
                    bgr = [int(c) for c in COLORS[classCodes[i]]]
                    cv2.rectangle(curr_frame, (x, y), (x + w, y + h), bgr, 2)
                    class_name = "{}: {:.4f}".format(obj_type, probabilities[i])
                    cv2.putText(curr_frame, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, bgr, 2)

        del_y = 30
        cv2.putText(curr_frame, f"frame_q : {detector_frame_queue.qsize()}", (20, 1 * del_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)
        cv2.putText(curr_frame, f"blob_q : {detector_blob_queue.qsize()}", (20, 2 * del_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        # todo: give as input to  gui
        cv2.imshow("output", cv2.resize(curr_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
        fps.update()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


def make_plate_blob():
    while True:
        pos, plate = reader_frame_queue.get()

        blob = cv2.dnn.blobFromImage(plate, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        reader_blob_queue.put((pos, blob))


def clean_up_old_plates(all_current_plates):
    tobe_deleted = []
    for tracking_plates in all_current_plates:
        if time.time() - tracking_plates[1] > 5:
            tobe_deleted.append(tracking_plates)

    for i in tobe_deleted:
        write_to_database(i)
        all_current_plates.remove(i)
        print('removed')


def read_plate(net):
    all_current_plates = list()

    global DATABASE_CONNECTION
    DATABASE_CONNECTION = sqlite3.connect(DATABASE_FILE)
    global DATABASE_CURSOR
    DATABASE_CURSOR = DATABASE_CONNECTION.cursor()
    global command_create_table

    DATABASE_CURSOR.execute(command_create_table)
    DATABASE_CONNECTION.commit()

    while True:
        try:
            pos, plate = reader_frame_queue.get(timeout=10)
        except:
            clean_up_old_plates(all_current_plates)
            continue

        # (cx, cy), blob = reader_blob_queue.get()

        closest = get_closest_poly(all_current_plates, plate)

        blob = cv2.dnn.blobFromImage(plate, 1 / 255.0, (PLATE_BLOB_SIZE, PLATE_BLOB_SIZE),
                                     swapRB=True, crop=False)

        if closest is None:
            num, conf = get_num_with_conf(net, blob)
            all_current_plates.append((plate, time.time(), num, conf, pos, 0))


        else:
            num, conf = get_num_with_conf(net, blob)
            _, _, _, prv_conf, _, _ = closest

            if prv_conf < conf:
                all_current_plates.remove(closest)
                diff = closest[5] + closest[4][1] - pos[1]
                all_current_plates.append((plate, time.time(), num, conf, pos, diff))
                # print(num, " diff = ", diff)



def get_num_with_conf(net, blob):
    net.setInput(blob)
    layerOutputs = net.forward(reader_out_layers_name)

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

            if max_confidence > READER_MIN_CONFIDENCE:
                rect = detected_obj[0:4] * np.array([416, 416, 416, 416])
                (centerX, centerY, width, height) = rect.astype("int")

                top_left_x = int(centerX - (width / 2))
                top_left_y = int(centerY - (height / 2))

                rects.append([top_left_x, top_left_y, int(width), int(height)])
                probabilities.append(float(max_confidence))
                classCodes.append(class_code)

    indices = cv2.dnn.NMSBoxes(rects, probabilities, READER_MIN_CONFIDENCE,
                               READER_MIN_CONFIDENCE)

    detections_in_order = []
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (rects[i][0], rects[i][1])

            obj_type = READER_LABELS[classCodes[i]]

            detections_in_order.append((x, y, obj_type, probabilities[i]))

    detections_in_order = sorted(detections_in_order, key=lambda p: (round(2 * p[1] / PLATE_BLOB_SIZE), p[0]))

    num = ""
    conf = 0

    for char in detections_in_order:
        num += char[2]
        conf += char[3]

    return num, conf


def get_closest_poly(all_curr_plates, plate):
    plate_hash = imagehash.average_hash(Image.fromarray(plate))
    for obj in all_curr_plates:
        obj_plate, _, _, _, _, _ = obj

        obj_plate_hash = imagehash.average_hash(Image.fromarray(obj_plate))

        if (obj_plate_hash - plate_hash) < 40:
            return obj

    return None


def write_to_database(plate_info):
    plate_img = plate_info[0]

    is_success, im_buf_arr = cv2.imencode(".jpg", plate_img)
    plate_img = im_buf_arr.tobytes()

    timestamp = time.asctime(time.localtime(plate_info[1]))
    license_num = plate_info[2]
    direction = "IN" if plate_info[5] < 0 else "OUT"

    # command_insert_data = f"INSERT INTO VEHICLE_INFO VALUES ({timestamp}, {license_num}, {plate_img}, {confidence}, {direction})"
    command_insert_data = f"INSERT INTO VEHICLE_INFO VALUES (?,?,?,?)"
    # data_tuple = (f"'{timestamp}'", f"'{license_num}'", sqlite3.Binary(plate_img), f"'{direction}'")

    data_tuple = (f"'{timestamp}'", f"'{license_num}'", sqlite3.Binary(plate_img), direction)

    print(data_tuple)

    DATABASE_CURSOR.execute(command_insert_data, data_tuple)
    DATABASE_CONNECTION.commit()


blob_thread = Thread(target=make_blob_from_frame, args=(video_source,), daemon=True)
blob_thread.start()

fps = FPS().start()
all_processing_thread = Thread(target=all_processing, args=(num_plate_detector_DNN,), daemon=True)
all_processing_thread.start()


reader_thread = Thread(target=read_plate, args=(num_plate_reader_DNN,), daemon=True)
reader_thread.start()

blob_thread.join()
reader_thread.join()

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

print("the end")
print(len(num_plates))

cv2.destroyAllWindows()

video_source.release()