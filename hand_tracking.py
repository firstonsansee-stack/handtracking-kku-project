import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import numpy as np
import time

# 1. ตรวจสอบและดาวน์โหลดโมเดล AI สำหรับตรวจจับมืออัตโนมัติ
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading AI model (Hand Landmarker)...")
    url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    try:
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")

# 2. ตั้งค่า MediaPipe Tasks API รองรับการตรวจจับ 2 มือพร้อมกัน
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)

# 3. เปิดใช้งานกล้องเว็บแคมผ่าน DirectShow สำหรับระบบ Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ตัวแปรควบคุมระบบ
canvas = None
current_mode = "DRAW"
filter_style = 0
last_clear_time = 0
last_gesture_switch_time = 0

# ตัวแปรสำหรับจำแนกและติดตามมือแต่ละข้างในโหมดวาด
prev_hand_dict = {}
next_hand_id = 0

# ตัวแปรสำหรับระบบหน่วงความนุ่มนวล (Smoothing)
smooth_idx_l = None
smooth_idx_r = None
smooth_thumb_r = None
smooth_thumb_l = None
smooth_circle_center = None
smooth_circle_radius = None
alpha = 0.35

def is_fist(landmarks, w, h):
    wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])
    mcp_middle = np.array([landmarks[9].x * w, landmarks[9].y * h])
    palm_size = np.linalg.norm(wrist - mcp_middle)
    
    tips = [8, 12, 16, 20]
    closed_count = 0
    for tip_id in tips:
        tip = np.array([landmarks[tip_id].x * w, landmarks[tip_id].y * h])
        if np.linalg.norm(tip - wrist) < palm_size * 1.3:
            closed_count += 1
    return closed_count >= 3

def is_pinky_folded(landmarks, w, h):
    wrist = np.array([landmarks[0].x * w, landmarks[0].y * h])
    pinky_tip = np.array([landmarks[20].x * w, landmarks[20].y * h])
    pinky_mcp = np.array([landmarks[17].x * w, landmarks[17].y * h])
    
    dist_tip = np.linalg.norm(pinky_tip - wrist)
    dist_mcp = np.linalg.norm(pinky_mcp - wrist)
    return dist_tip < dist_mcp * 1.25

print("Dual-Mode Hand Tracking System Initialized.")
print("Controls:")
print(" - Press 'F': Switch between DRAW and FRAME modes")
print(" - DRAW Mode: Pinch index and thumb to draw (Dark blue line) | Make a fist to clear canvas")
print(" - FRAME Mode: Single hand creates circle filter | Two hands create polygon window filter | Fold pinkies to switch filters")
print(" - Press 'Q': Exit program")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from camera.")
        break

    frame = cv2.flip(frame, 1)
    
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)
    h, w, _ = frame.shape

    if current_mode == "DRAW":
        current_frame_hands = []
        if detection_result.hand_landmarks:
            for hand_lms in detection_result.hand_landmarks:
                wrist = (hand_lms[0].x * w, hand_lms[0].y * h)
                index_tip = (int(hand_lms[8].x * w), int(hand_lms[8].y * h))
                thumb_tip = (int(hand_lms[4].x * w), int(hand_lms[4].y * h))
                pinch_dist = np.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
                
                if is_fist(hand_lms, w, h):
                    if time.time() - last_clear_time > 1.0:
                        canvas = np.zeros_like(frame)
                        last_clear_time = time.time()
                        print("Fist detected: Canvas cleared.")

                for lm in hand_lms:
                    lx, ly = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (lx, ly), 2, (0, 200, 255), -1)

                current_frame_hands.append({
                    'wrist': wrist,
                    'index_tip': index_tip,
                    'pinching': pinch_dist < 35
                })

        matched_hands = []
        for h_curr in current_frame_hands:
            best_match = None
            min_dist = float('inf')
            for prev_id, p_data in prev_hand_dict.items():
                dist = np.hypot(h_curr['wrist'][0] - p_data['wrist'][0], h_curr['wrist'][1] - p_data['wrist'][1])
                if dist < 120 and dist < min_dist:
                    min_dist = dist
                    best_match = prev_id
            
            if best_match is not None:
                matched_hands.append((best_match, h_curr))
            else:
                new_id = next_hand_id
                next_hand_id += 1
                matched_hands.append((new_id, h_curr))

        new_hand_dict = {}
        for hand_id, h_curr in matched_hands:
            idx_pt = h_curr['index_tip']
            if h_curr['pinching']:
                cv2.circle(frame, idx_pt, 5, (255, 0, 0), cv2.FILLED)
                if hand_id in prev_hand_dict and prev_hand_dict[hand_id]['pinching']:
                    prev_pt = prev_hand_dict[hand_id]['last_pt']
                    cv2.line(canvas, prev_pt, idx_pt, (255, 0, 0), 4)
                new_hand_dict[hand_id] = {
                    'wrist': h_curr['wrist'],
                    'pinching': True,
                    'last_pt': idx_pt
                }
            else:
                cv2.circle(frame, idx_pt, 5, (200, 50, 0), cv2.FILLED)
                new_hand_dict[hand_id] = {
                    'wrist': h_curr['wrist'],
                    'pinching': False,
                    'last_pt': idx_pt
                }
        prev_hand_dict = new_hand_dict

        frame = cv2.addWeighted(frame, 1, canvas, 0.85, 0)

    elif current_mode == "FRAME":
        if detection_result.hand_landmarks:
            num_hands = len(detection_result.hand_landmarks)
            
            if num_hands >= 2:
                both_pinkies_folded = all(is_pinky_folded(hlms, w, h) for hlms in detection_result.hand_landmarks)
                if both_pinkies_folded and (time.time() - last_gesture_switch_time > 1.2):
                    filter_style = (filter_style + 1) % 6
                    last_gesture_switch_time = time.time()
                    print(f"Pinky folded: Filter style changed to {filter_style}")

            if num_hands == 1:
                hand_lms = detection_result.hand_landmarks[0]
                
                thumb_pt = np.array([hand_lms[4].x * w, hand_lms[4].y * h])
                index_pt = np.array([hand_lms[8].x * w, hand_lms[8].y * h])
                pinch_dist = np.linalg.norm(thumb_pt - index_pt)

                if pinch_dist > 35:
                    palm_pts = [0, 5, 9, 13, 17]
                    palm_x = np.mean([hand_lms[i].x * w for i in palm_pts])
                    palm_y = np.mean([hand_lms[i].y * h for i in palm_pts])
                    raw_center = np.array([palm_x, palm_y], dtype=np.float32)

                    raw_radius = pinch_dist * 1.5

                    if smooth_circle_center is None:
                        smooth_circle_center = raw_center
                        smooth_circle_radius = raw_radius
                    else:
                        smooth_circle_center = alpha * raw_center + (1 - alpha) * smooth_circle_center
                        smooth_circle_radius = alpha * raw_radius + (1 - alpha) * smooth_circle_radius

                    center = (int(smooth_circle_center[0]), int(smooth_circle_center[1]))
                    radius = int(smooth_circle_radius)

                    if radius > 10:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.circle(mask, center, radius, 255, -1)

                        filtered_frame = frame.copy()
                        if filter_style == 0:
                            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
                            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)
                        elif filter_style == 1:
                            filtered_frame = cv2.bitwise_not(filtered_frame)
                        elif filter_style == 2:
                            filtered_frame = cv2.Canny(filtered_frame, 100, 200)
                            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)
                        elif filter_style == 3:
                            hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
                            hsv[:, :, 0] = (hsv[:, :, 0] + int(time.time() * 60) % 180)
                            filtered_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        elif filter_style == 4:
                            small = cv2.resize(filtered_frame, (w // 16, h // 16), interpolation=cv2.INTER_LINEAR)
                            filtered_frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                        elif filter_style == 5:
                            sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                                     [0.349, 0.686, 0.168],
                                                     [0.393, 0.769, 0.189]])
                            filtered_frame = cv2.transform(filtered_frame, sepia_kernel)
                            filtered_frame = np.clip(filtered_frame, 0, 255).astype(np.uint8)

                        inverse_mask = cv2.bitwise_not(mask)
                        bg = cv2.bitwise_and(frame, frame, mask=inverse_mask)
                        fg = cv2.bitwise_and(filtered_frame, filtered_frame, mask=mask)
                        frame = cv2.add(bg, fg)

                        cv2.circle(frame, center, radius, (0, 165, 255), 3)

            elif num_hands >= 2:
                left_hand = None
                right_hand = None
                
                for hand_lms, hand_class in zip(detection_result.hand_landmarks, detection_result.handedness):
                    label = hand_class[0].category_name
                    if label == "Left":
                        left_hand = hand_lms
                    elif label == "Right":
                        right_hand = hand_lms

                if left_hand and right_hand:
                    raw_idx_l = np.array([left_hand[8].x * w, left_hand[8].y * h], dtype=np.float32)
                    raw_idx_r = np.array([right_hand[8].x * w, right_hand[8].y * h], dtype=np.float32)
                    raw_thumb_r = np.array([right_hand[4].x * w, right_hand[4].y * h], dtype=np.float32)
                    raw_thumb_l = np.array([left_hand[4].x * w, left_hand[4].y * h], dtype=np.float32)

                    if smooth_idx_l is None:
                        smooth_idx_l, smooth_idx_r = raw_idx_l, raw_idx_r
                        smooth_thumb_r, smooth_thumb_l = raw_thumb_r, raw_thumb_l
                    else:
                        smooth_idx_l = alpha * raw_idx_l + (1 - alpha) * smooth_idx_l
                        smooth_idx_r = alpha * raw_idx_r + (1 - alpha) * smooth_idx_r
                        smooth_thumb_r = alpha * raw_thumb_r + (1 - alpha) * smooth_thumb_r
                        smooth_thumb_l = alpha * raw_thumb_l + (1 - alpha) * smooth_thumb_l

                    idx_l = tuple(smooth_idx_l.astype(int))
                    idx_r = tuple(smooth_idx_r.astype(int))
                    thumb_r = tuple(smooth_thumb_r.astype(int))
                    thumb_l = tuple(smooth_thumb_l.astype(int))

                    polygon_pts = np.array([idx_l, idx_r, thumb_r, thumb_l], dtype=np.int32)
                    frame_area = cv2.contourArea(polygon_pts)

                    if frame_area > 3000:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(mask, [polygon_pts], 255)

                        filtered_frame = frame.copy()
                        if filter_style == 0:
                            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2GRAY)
                            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)
                        elif filter_style == 1:
                            filtered_frame = cv2.bitwise_not(filtered_frame)
                        elif filter_style == 2:
                            filtered_frame = cv2.Canny(filtered_frame, 100, 200)
                            filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)
                        elif filter_style == 3:
                            hsv = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2HSV)
                            hsv[:, :, 0] = (hsv[:, :, 0] + int(time.time() * 60) % 180)
                            filtered_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        elif filter_style == 4:
                            small = cv2.resize(filtered_frame, (w // 16, h // 16), interpolation=cv2.INTER_LINEAR)
                            filtered_frame = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
                        elif filter_style == 5:
                            sepia_kernel = np.array([[0.272, 0.534, 0.131],
                                                     [0.349, 0.686, 0.168],
                                                     [0.393, 0.769, 0.189]])
                            filtered_frame = cv2.transform(filtered_frame, sepia_kernel)
                            filtered_frame = np.clip(filtered_frame, 0, 255).astype(np.uint8)

                        inverse_mask = cv2.bitwise_not(mask)
                        bg = cv2.bitwise_and(frame, frame, mask=inverse_mask)
                        fg = cv2.bitwise_and(filtered_frame, filtered_frame, mask=mask)
                        frame = cv2.add(bg, fg)

                    cv2.line(frame, idx_l, idx_r, (0, 255, 255), 3)     
                    cv2.line(frame, thumb_l, thumb_r, (255, 0, 255), 3) 
                    cv2.line(frame, idx_l, thumb_l, (255, 255, 255), 3) 
                    cv2.line(frame, idx_r, thumb_r, (255, 255, 255), 3) 

                    cv2.circle(frame, idx_l, 6, (0, 255, 0), -1)
                    cv2.circle(frame, idx_r, 6, (0, 255, 0), -1)
                    cv2.circle(frame, thumb_l, 6, (255, 0, 0), -1)
                    cv2.circle(frame, thumb_r, 6, (255, 0, 0), -1)

    cv2.rectangle(frame, (10, 10), (450, 80), (10, 10, 10), -1)
    cv2.rectangle(frame, (10, 10), (450, 80), (0, 255, 255), 1)
    
    cv2.putText(frame, f"MODE [F]: {current_mode}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    if current_mode == "DRAW":
        cv2.putText(frame, "VFX Draw Mode | [Make a Fist] Clear", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    else:
        cv2.putText(frame, f"Filter Style [{filter_style}] (0-5) | Fold Pinkies to Switch", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    cv2.imshow('Dual-Mode Hand Tracking - KKU AI', frame)

    key = cv2.waitKey(25) & 0xFF
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('f'), ord('F')]:
        current_mode = "FRAME" if current_mode == "DRAW" else "DRAW"
        print(f"Mode switched to: {current_mode}")
    elif key in [ord('v'), ord('V')] and current_mode == "FRAME":
        filter_style = (filter_style + 1) % 6
        print(f"Filter style changed to: {filter_style}")

cap.release()
cv2.destroyAllWindows()