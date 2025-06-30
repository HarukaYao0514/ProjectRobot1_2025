import sensor
import image
import lcd
import time
import gc
import uhashlib
import ubinascii
from fpioa_manager import fm
from board import board_info
from maix import KPU, GPIO
from modules import ybserial

# Initialize serial communication
serial = ybserial()

# Initialize the LCD and camera
lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time=100)
clock = time.clock()

# Initialize Boot key
fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS0)
key = GPIO(GPIO.GPIOHS0, GPIO.IN)

# Load face detection model
face_detect = KPU()
face_detect.load_kmodel("/sd/KPU/yolo_face_detect/face_detect_320x240.kmodel")
anchor = (0.1075, 0.126875, 0.126875, 0.175, 0.1465625, 0.2246875,
          0.1953125, 0.25375, 0.2440625, 0.351875, 0.341875, 0.4721875,
          0.5078125, 0.6696875, 0.8984375, 1.099687, 2.129062, 2.425937)
face_detect.init_yolo2(anchor, anchor_num=9, img_w=320, img_h=240,
                       net_w=320, net_h=240, layer_w=10, layer_h=8,
                       threshold=0.7, nms_value=0.2, classes=1)

landmark = KPU()
landmark.load_kmodel("/sd/KPU/face_recognization/ld5.kmodel")
feature_extract = KPU()
feature_extract.load_kmodel("/sd/KPU/face_recognization/feature_extraction.kmodel")

# Facial alignment target points
dst_point = [(22, 29), (42, 29), (32, 40), (24, 50), (40, 50)]

# Feature simplification and hashing
SIMPLIFY_DIM = 8
MATCH_THRESHOLD = 0.45  # Allows small variation
registered_hashes = []  # Each item is (vec[:SIMPLIFY_DIM], index)

def simplify_feature(feature):
    return feature[:SIMPLIFY_DIM]

def vector_distance(v1, v2):
    return sum((a - b) ** 2 for a, b in zip(v1, v2)) ** 0.5

def align_face_and_extract(img, face_box):
    x, y, w, h = face_box
    cut = img.cut(x, y, w, h)
    face = cut.resize(128, 128)
    face.pix_to_ai()
    out = landmark.run_with_output(face, getlist=True)
    keypoints = [(int(KPU.sigmoid(out[i*2])*w + x),
                  int(KPU.sigmoid(out[i*2+1])*h + y)) for i in range(5)]
    aligned = image.Image(size=(64, 64))
    aligned.pix_to_ai()
    T = image.get_affine_transform(keypoints, dst_point)
    image.warp_affine_ai(img, aligned, T)
    feature = feature_extract.run_with_output(aligned, get_feature=True)
    return feature

# Main loop
last_msg = None
while True:
    gc.collect()
    clock.tick()
    img = sensor.snapshot()
    face_detect.run_with_output(img)
    faces = face_detect.regionlayer_yolo2()
    current_msg = "N"

    if len(faces) > 0:
        face = faces[0]
        x, y, w, h = face[:4]
        feature = align_face_and_extract(img, (x, y, w, h))
        vec = simplify_feature(feature)

        if key.value() == 0:
            registered_hashes.append((vec, len(registered_hashes)+1))
            print("Registered #%d" % len(registered_hashes))
            time.sleep(0.5)
            img.draw_rectangle(x, y, w, h, color=(255, 255, 255))
            current_msg = "R"

        elif registered_hashes:
            matched = False
            for reg_vec, reg_id in registered_hashes:
                dist = float(vector_distance(vec, reg_vec))
                if dist < MATCH_THRESHOLD:
                    img.draw_string(0, 200, "Match #%d dist=%.2f" % (reg_id, dist), scale=2)
                    img.draw_rectangle(x, y, w, h, color=(0, 255, 0))
                    current_msg = str(reg_id)
                    matched = True
                    break
            if not matched:
                img.draw_string(0, 200, "Unknown", scale=2)
                img.draw_rectangle(x, y, w, h, color=(255, 255, 255))
                current_msg = "U"
        else:
            current_msg = "U"
            img.draw_rectangle(x, y, w, h, color=(255, 255, 255))

    if current_msg != last_msg:
        serial.send(current_msg + ",")
        time.sleep_ms(1)
        print("Send to microbit:", current_msg + ",")
        last_msg = current_msg

    img.draw_string(0, 0, "%2.1f FPS" % clock.fps(), scale=2)
    lcd.display(img)

# Release models
face_detect.deinit()
landmark.deinit()
feature_extract.deinit()