import asyncio
import os
import cv2
import time
import json
import hashlib
import numpy as np
import uuid
from time import strftime, gmtime
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from supabase import create_client
from ultralytics import YOLO
import torch
import torch.nn as nn
from torchvision import transforms

SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"

VIDEO_PATH = 'raw_videos/'
MOTION_PATH = 'motion_videos/'
PERSON_PATH = 'person_images/'
ANOMALY_PATH = 'anomaly_videos/'

os.makedirs(VIDEO_PATH, exist_ok=True)
os.makedirs(MOTION_PATH, exist_ok=True)
os.makedirs(PERSON_PATH, exist_ok=True)
os.makedirs(ANOMALY_PATH, exist_ok=True)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def simulate_pqc_key_exchange():
    return get_random_bytes(16)

with open("aes_key.bin", "rb") as f:
    shared_key = f.read()

def aes_encrypt(data_bytes, key):
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(data_bytes)
    return cipher.nonce + tag + ciphertext

def unique_filename(name):
    base, ext = os.path.splitext(name)
    suffix = uuid.uuid4().hex[:8]
    return f"{base}_{suffix}{ext}"

class SimpleBlockchain:
    def __init__(self):
        self.chain = []
        self.create_block(nonce=1, previous_hash='0')

    def create_block(self, nonce, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'nonce': nonce,
            'previous_hash': previous_hash,
            'data': []
        }
        self.chain.append(block)
        return block

    def add_data(self, data):
        self.chain[-1]['data'].append(data)

    def get_last_block(self):
        return self.chain[-1]

    def hash(self, block):
        encoded = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def proof_of_work(self, previous_nonce):
        new_nonce = 1
        while True:
            guess = str(new_nonce**2 - previous_nonce**2).encode()
            guess_hash = hashlib.sha256(guess).hexdigest()
            if guess_hash[:2] == '00':
                return new_nonce
            new_nonce += 1

blockchain = SimpleBlockchain()

async def upload_to_supabase(bucket, file_name, data):
    unique_name = unique_filename(file_name)
    storage = supabase.storage.from_(bucket)
    try:
        response = storage.upload(unique_name, data)
        if hasattr(response, 'path') and response.path:
            url = f"{supabase.url}/storage/v1/object/public/{bucket}/{unique_name}"
            print(f"Uploaded {unique_name} to {bucket} bucket.")
            return url
        else:
            print(f"Upload failed: {response}")
            return None
    except Exception as e:
        print(f"Upload exception: {e}")
        return None

class DummyAutoencoder(nn.Module):
    def __init__(self): 
        super().__init__()
    def forward(self, x): 
        return x

autoencoder = DummyAutoencoder()
autoencoder.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - inter_area
    return inter_area / union_area if union_area else 0

pending_tasks = []

async def cleanup_tasks():
    global pending_tasks
    if pending_tasks:
        done_tasks = [task for task in pending_tasks if task.done()]
        for task in done_tasks:
            try:
                await task
            except Exception as e:
                print(f"Task error: {e}")
        pending_tasks = [task for task in pending_tasks if not task.done()]

async def main_loop():
    global pending_tasks

    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    last_raw_video_time = 0
    RAW_VIDEO_INTERVAL = 10
    current_raw_video_writer = None
    current_raw_video_filename = None

    last_person_save_time = 0
    PERSON_SAVE_DEBOUNCE_SECONDS = 5
    last_person_bbox = None
    IOU_THRESHOLD = 0.5

    motion_history = None
    motion_detected = False
    motion_video_writer = None
    motion_start_time = None
    MOTION_RECORD_SECONDS = 5

    anomaly_detected = False
    anomaly_video_writer = None
    anomaly_start_time = None
    ANOMALY_RECORD_SECONDS = 5
    ANOMALY_THRESHOLD = 0.02

    model = YOLO('yolov8n.pt')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_small = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            current_time = time.time()

            if len(pending_tasks) > 10:
                await cleanup_tasks()

            if current_time - last_raw_video_time > RAW_VIDEO_INTERVAL:
                if current_raw_video_writer:
                    current_raw_video_writer.release()
                    if current_raw_video_filename and os.path.exists(current_raw_video_filename):
                        async def upload_raw():
                            with open(current_raw_video_filename, "rb") as f:
                                encrypted = aes_encrypt(f.read(), shared_key)
                            url = await upload_to_supabase("raw_videos", os.path.basename(current_raw_video_filename), encrypted)
                            if url:
                                block = blockchain.get_last_block()
                                nonce = blockchain.proof_of_work(block['nonce'])
                                prev_hash = blockchain.hash(block)
                                blockchain.create_block(nonce, prev_hash)
                                blockchain.add_data({'type': 'raw_video', 'url': url, 'timestamp': time.time()})
                        pending_tasks.append(asyncio.create_task(upload_raw()))
                ts = strftime("%Y%m%d-%H%M%S", gmtime())
                current_raw_video_filename = os.path.join(VIDEO_PATH, f"video_{ts}.avi")
                current_raw_video_writer = cv2.VideoWriter(current_raw_video_filename, fourcc, 20.0, (640, 480))
                last_raw_video_time = current_time

            if current_raw_video_writer:
                current_raw_video_writer.write(frame_small)

            if motion_history is not None:
                diff = cv2.absdiff(gray, motion_history)
                motion_area = np.sum(diff > 50)
                if motion_area > 2000:
                    if not motion_detected:
                        motion_detected = True
                        motion_start_time = time.time()
                        motion_ts = strftime("%Y%m%d-%H%M%S", gmtime())
                        motion_vid_filename = os.path.join(MOTION_PATH, f"motion_{motion_ts}.avi")
                        motion_video_writer = cv2.VideoWriter(motion_vid_filename, fourcc, 20.0, (640, 480))
                    if motion_video_writer:
                        motion_video_writer.write(frame_small)
                elif motion_detected and motion_video_writer and (time.time() - motion_start_time) > MOTION_RECORD_SECONDS:
                    motion_detected = False
                    motion_video_writer.release()
                    motion_video_writer = None
                    async def upload_motion():
                        with open(motion_vid_filename, "rb") as f:
                            encrypted = aes_encrypt(f.read(), shared_key)
                        url = await upload_to_supabase("motion_videos", os.path.basename(motion_vid_filename), encrypted)
                        if url:
                            block = blockchain.get_last_block()
                            nonce = blockchain.proof_of_work(block['nonce'])
                            prev_hash = blockchain.hash(block)
                            blockchain.create_block(nonce, prev_hash)
                            blockchain.add_data({'type': 'motion', 'url': url, 'timestamp': time.time()})
                    pending_tasks.append(asyncio.create_task(upload_motion()))
            motion_history = gray

            results = model(frame_small)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                for i, box in enumerate(boxes):
                    cls_id = int(class_ids[i])
                    if cls_id == 0:
                        x1, y1, x2, y2 = box.astype(int)
                        w, h = x2 - x1, y2 - y1
                        if w < 60 or h < 120:
                            continue
                        cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        person_bbox = [x1, y1, w, h]
                        save_image = False
                        if last_person_save_time == 0 or (current_time - last_person_save_time > PERSON_SAVE_DEBOUNCE_SECONDS):
                            if last_person_bbox is None or iou(person_bbox, last_person_bbox) < IOU_THRESHOLD:
                                save_image = True
                        if save_image:
                            crop = frame_small[y1:y2, x1:x2]
                            person_ts = strftime("%Y%m%d-%H%M%S", gmtime())
                            fname = os.path.join(PERSON_PATH, f"person_{person_ts}_{x1}_{y1}.jpg")
                            cv2.imwrite(fname, crop)
                            blockchain.add_data({
                                'type': 'person',
                                'image': fname,
                                'bounding_box': [x1, y1, w, h],
                                'timestamp': current_time
                            })
                            last_person_save_time = current_time
                            last_person_bbox = person_bbox

            input_tensor = transform(gray).unsqueeze(0)
            with torch.no_grad():
                output = autoencoder(input_tensor)
            loss = torch.mean((output - input_tensor) ** 2).item()
            if loss > ANOMALY_THRESHOLD:
                if not anomaly_detected:
                    anomaly_detected = True
                    anomaly_start_time = time.time()
                    anomaly_ts = strftime("%Y%m%d-%H%M%S", gmtime())
                    anomaly_vid_filename = os.path.join(ANOMALY_PATH, f"anomaly_{anomaly_ts}.avi")
                    anomaly_video_writer = cv2.VideoWriter(anomaly_vid_filename, fourcc, 20.0, (640, 480))
                if anomaly_video_writer:
                    anomaly_video_writer.write(frame_small)
            elif anomaly_detected and anomaly_video_writer and (time.time() - anomaly_start_time) > ANOMALY_RECORD_SECONDS:
                anomaly_detected = False
                anomaly_video_writer.release()
                anomaly_video_writer = None
                async def upload_anomaly():
                    with open(anomaly_vid_filename, "rb") as f:
                        encrypted = aes_encrypt(f.read(), shared_key)
                    url = await upload_to_supabase("anomaly_videos", os.path.basename(anomaly_vid_filename), encrypted)
                    if url:
                        block = blockchain.get_last_block()
                        nonce = blockchain.proof_of_work(block['nonce'])
                        prev_hash = blockchain.hash(block)
                        blockchain.create_block(nonce, prev_hash)
                        blockchain.add_data({'type': 'anomaly', 'url': url, 'loss': loss, 'timestamp': time.time()})
                pending_tasks.append(asyncio.create_task(upload_anomaly()))

            cv2.imshow('Surveillance', frame_small)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        if current_raw_video_writer:
            current_raw_video_writer.release()
        if motion_video_writer:
            motion_video_writer.release()
        if anomaly_video_writer:
            anomaly_video_writer.release()
        if pending_tasks:
            print("Waiting for pending upload tasks to finish...")
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        print("Surveillance stopped gracefully.")

if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("Program stopped by user.")
