from datetime import datetime

import cv2
import pytesseract
import numpy as np

# todo: replace the path with appropriate path on the running system
pytesseract.pytesseract.tesseract_cmd = r'D:\PROGRAMS\Tesseract-OCR\tesseract'


def resize_num_plate(img):
    input_height, input_width = img.shape

    scaling_factor = 600 / input_width
    output_height, output_width = int(scaling_factor * input_height), int(scaling_factor * input_width)

    return cv2.resize(img, dsize=(output_width, output_height), interpolation=cv2.INTER_CUBIC)


def get_license_num(img):
    HEIGHT, WIDTH = img.shape

    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img2 = np.full(img.shape, 255, dtype=np.uint8)

    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        if ((w < 5) or (h * w < 1000) or (h / HEIGHT < 0.25) or (w / WIDTH > 0.3) or (h / HEIGHT > 0.9) or (
                h / w < 0.5)):
            continue
        else:
            img2 = cv2.drawContours(img2, contours, idx, (0, 0, 0), thickness=-1)

    img2 = cv2.bitwise_or(img, img2)

    cv2.imshow('Just numbers ', img2)

    prediction = pytesseract.image_to_string(img2, lang='eng', output_type=pytesseract.Output.DICT,
                                             config='--psm 4 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    predicted_num = prediction['text'].replace('\x0c', '')
    predicted_num = predicted_num.replace('\n', '')

    return predicted_num


def peicewise_transformation(img):
    gamma = 1.5
    r1, s1 = 20, 0
    r2, s2 = 200, 255

    lookup_table = np.array([255 if (i > r2) else 0 if (i < r1) else ((i / 255.0) ** gamma) * 255 for i in
                             np.arange(0, 256)]).astype('uint8')

    return cv2.LUT(img, lookup_table)


def pre_process_num_plate(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize_num_plate(img)

    # cv2.imshow('Enlarged image', img)

    img = peicewise_transformation(img)

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)

    return img


def print_time():
    format_string = '%H:%M:%S.%f seconds'
    a_datetime_datetime = datetime.now()
    current_time_string = a_datetime_datetime.strftime(format_string)
    print(current_time_string)


if __name__ == '__main__':
    print_time()

    image = cv2.imread('../yellow_truck.png')

    # cv2.imshow('small cropped image', image)

    image = pre_process_num_plate(image)

    # cv2.imshow('Preprocessed image', image)

    number = get_license_num(image)

    print("Number is : ", number)

    print_time()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
