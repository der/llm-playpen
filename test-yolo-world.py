from ultralytics import YOLO
from ultralytics import YOLOE
import cv2
import requests
from io import BytesIO
from PIL import Image
import time

classes = ["person", "teddy bear", "book", "bookshelf", "chair", "door", "doorframe", "windowed door", "stool", "pc", "sofa", "table", "tv"]
url = "http://marvin.local:8080/still"

def test_yolo_world():
    model = YOLO("yolov8x-worldv2.pt")  # or choose yolov8m/l-world.pt
    model.set_classes(classes)
    results = model.predict("data/marvin-512.png")
    results[0].show()

def test_yoloe():
    model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes
    # Set text prompt. You only need to do this once after you load the model.
    model.set_classes(classes, model.get_text_pe(classes))
    results = model.predict("data/marvin-512.png")
    results[0].show()

def get_image():
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
    except IOError as e:
        print(f"Error processing image: {e}")

def test_loop():
    model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes
    model.set_classes(classes, model.get_text_pe(classes))
    window_name = "YOLO-E Test Loop"
    cv2.namedWindow(window_name)
    cv2.waitKey(10)
    while(True):
        image = get_image()
        print("calling predict")
        results = model.predict(image, classes=[1])
        print("Results: {results}")
        annotated_frame = results[0].plot()
        cv2.imshow(window_name, annotated_frame)
        cv2.waitKey(10)
        time.sleep(.5)

test_loop()
