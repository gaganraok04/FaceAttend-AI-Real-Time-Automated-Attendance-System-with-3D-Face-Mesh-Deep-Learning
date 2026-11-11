"""
Real-Time Face Recognition Attendance System with Light-Brown Geometric Wireframe
Combines MediaPipe 3D facial wireframe mesh with OpenCV LBPH face recognition
to automatically identify students, display live wireframe tracking, and mark attendance.
"""

import os
import sys
import csv
import time
from datetime import datetime
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

# ============================================================
# 1. FILE PATHS & SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "face_model.yml")
LABEL_FILE = os.path.join(BASE_DIR, "face_labels.txt")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
FACE_MODEL_PATH = os.path.join(BASE_DIR, "face_landmarker.task")

FACE_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# LBPH Distance Threshold (lower = stricter match)
RECOGNITION_THRESHOLD = 75

# Update Last Seen timestamp every 2 seconds
UPDATE_INTERVAL = 2

# ============================================================
# 2. VERIFY REQUIRED MODELS & DOWNLOAD IF MISSING
# ============================================================

def ensure_face_landmarker_task():
    if not os.path.exists(FACE_MODEL_PATH):
        print("Downloading MediaPipe face landmarker model...")
        try:
            urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
            print("MediaPipe model downloaded successfully.")
        except Exception as e:
            print("ERROR: Could not download face_landmarker.task:", e)
            sys.exit(1)

ensure_face_landmarker_task()

if not os.path.exists(MODEL_FILE):
    print("ERROR: face_model.yml not found. Please train the model first.")
    sys.exit(1)

if not os.path.exists(LABEL_FILE):
    print("ERROR: face_labels.txt not found. Please train the model first.")
    sys.exit(1)

# ============================================================
# 3. LOAD LBPH RECOGNIZER & LABELS
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

labels = {}
with open(LABEL_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    for row in reader:
        labels[int(row["Label"])] = {
            "id": row["Student ID"],
            "name": row["Student Name"]
        }

# Haar face detector for LBPH face cropping
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ============================================================
# 4. INITIALIZE MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=4,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# ============================================================
# 5. LIGHT-BROWN THEME & STRUCTURAL WIREFRAME LOOPS
# ============================================================

def _build_loop(connections):
    adj = {}
    for c in connections:
        adj.setdefault(c.start, []).append(c.end)
        adj.setdefault(c.end, []).append(c.start)
    start = list(adj.keys())[0]
    loop = [start]
    prev, curr = None, start
    while True:
        nxts = [n for n in adj[curr] if n != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, curr = curr, nxt
    return loop

OVAL_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
LEFT_EYE_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
LEFT_EYEBROW_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
RIGHT_EYEBROW_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
OUTER_LIP_LOOP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
NOSE_WINGS = [98, 97, 2, 326, 327]

STRUCTURAL_MESH_INDICES = sorted(list(set(
    OVAL_LOOP[::2] +
    LEFT_EYE_LOOP[::2] +
    RIGHT_EYE_LOOP[::2] +
    LEFT_EYEBROW_LOOP +
    RIGHT_EYEBROW_LOOP +
    OUTER_LIP_LOOP[::2] +
    NOSE_BRIDGE + NOSE_WINGS +
    [10, 67, 109, 108, 151, 337, 338, 297] +
    [50, 118, 123, 187, 205] +
    [280, 347, 352, 411, 425] +
    [152, 175, 199, 200]
)))

# Colors (OpenCV BGR Format)
COLOR_WIRE_MESH = (135, 180, 225)       # Spacious light brown wireframe
COLOR_CONTOUR = (85, 145, 225)         # Rich light golden-brown contour
COLOR_EYES = (110, 185, 245)           # Golden amber for eyes
COLOR_LIPS = (90, 140, 225)            # Warm caramel for lips
COLOR_SUCCESS = (60, 210, 120)         # Verified green for Present status
COLOR_UNKNOWN = (60, 60, 220)          # Soft red for Unknown face

# ============================================================
# 6. ATTENDANCE LOGGING FUNCTION
# ============================================================

def mark_attendance(student_id, student_name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    rows = []
    found = False

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("Student ID") == student_id and row.get("Date") == date:
                    row["Last Seen"] = current_time
                    row["Status"] = "Present"
                    found = True
                rows.append(row)

    if not found:
        rows.append({
            "Student ID": student_id,
            "Student Name": student_name,
            "Date": date,
            "First Seen": current_time,
            "Last Seen": current_time,
            "Status": "Present"
        })

    with open(ATTENDANCE_FILE, "w", newline="", encoding="utf-8") as file:
        fieldnames = ["Student ID", "Student Name", "Date", "First Seen", "Last Seen", "Status"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# ============================================================
# 7. DRAW LIGHT-BROWN GEOMETRIC WIREFRAME
# ============================================================

def draw_light_brown_wireframe(image, face_landmarks):
    height, width = image.shape[:2]

    all_pts = []
    for lm in face_landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        all_pts.append((x, y))

    total_pts = len(all_pts)
    if total_pts < 468:
        return image

    left_eye_poly = np.array([all_pts[i] for i in LEFT_EYE_LOOP], dtype=np.int32)
    right_eye_poly = np.array([all_pts[i] for i in RIGHT_EYE_LOOP], dtype=np.int32)
    mouth_poly = np.array([all_pts[i] for i in OUTER_LIP_LOOP], dtype=np.int32)
    face_hull = cv2.convexHull(np.array([all_pts[i] for i in OVAL_LOOP], dtype=np.int32))

    # Spacious Delaunay Mesh
    mesh_points = []
    for idx in STRUCTURAL_MESH_INDICES:
        if idx < total_pts:
            pt = all_pts[idx]
            if 0 <= pt[0] < width and 0 <= pt[1] < height:
                mesh_points.append(pt)

    if len(mesh_points) >= 30:
        subdiv = cv2.Subdiv2D((0, 0, width, height))
        for pt in mesh_points:
            try:
                subdiv.insert((float(pt[0]), float(pt[1])))
            except Exception:
                pass

        triangle_list = subdiv.getTriangleList()
        if triangle_list is not None:
            max_edge = max(40, int(width * 0.16))

            for t in triangle_list:
                x1, y1 = int(t[0]), int(t[1])
                x2, y2 = int(t[2]), int(t[3])
                x3, y3 = int(t[4]), int(t[5])

                cx = (x1 + x2 + x3) // 3
                cy = (y1 + y2 + y3) // 3

                if cv2.pointPolygonTest(face_hull, (cx, cy), False) < 0:
                    continue
                if cv2.pointPolygonTest(left_eye_poly, (cx, cy), False) >= 0:
                    continue
                if cv2.pointPolygonTest(right_eye_poly, (cx, cy), False) >= 0:
                    continue
                if cv2.pointPolygonTest(mouth_poly, (cx, cy), False) >= 0:
                    continue

                d12 = np.hypot(x2 - x1, y2 - y1)
                d23 = np.hypot(x3 - x2, y3 - y2)
                d31 = np.hypot(x1 - x3, y1 - y3)
                if d12 > max_edge or d23 > max_edge or d31 > max_edge:
                    continue

                cv2.line(image, (x1, y1), (x2, y2), COLOR_WIRE_MESH, 1, cv2.LINE_AA)
                cv2.line(image, (x2, y2), (x3, y3), COLOR_WIRE_MESH, 1, cv2.LINE_AA)
                cv2.line(image, (x3, y3), (x1, y1), COLOR_WIRE_MESH, 1, cv2.LINE_AA)

    # Feature Outlines (Eyes, Lips, Eyebrows, Nose, Oval)
    cv2.polylines(image, [left_eye_poly], True, COLOR_EYES, 2, cv2.LINE_AA)
    cv2.polylines(image, [right_eye_poly], True, COLOR_EYES, 2, cv2.LINE_AA)

    lbrow_pts = np.array([all_pts[i] for i in LEFT_EYEBROW_LOOP], dtype=np.int32)
    rbrow_pts = np.array([all_pts[i] for i in RIGHT_EYEBROW_LOOP], dtype=np.int32)
    cv2.polylines(image, [lbrow_pts], False, COLOR_CONTOUR, 2, cv2.LINE_AA)
    cv2.polylines(image, [rbrow_pts], False, COLOR_CONTOUR, 2, cv2.LINE_AA)

    cv2.polylines(image, [mouth_poly], True, COLOR_LIPS, 2, cv2.LINE_AA)

    for k in range(len(NOSE_BRIDGE) - 1):
        cv2.line(image, all_pts[NOSE_BRIDGE[k]], all_pts[NOSE_BRIDGE[k + 1]], COLOR_CONTOUR, 2, cv2.LINE_AA)

    oval_pts = np.array([all_pts[i] for i in OVAL_LOOP], dtype=np.int32)
    cv2.polylines(image, [oval_pts], True, COLOR_CONTOUR, 2, cv2.LINE_AA)

    return image

# ============================================================
# 8. MAIN CAMERA ATTENDANCE LOOP
# ============================================================

def main():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "FaceAttend AI - Live Attendance with Wireframe"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print()
    print("==================================================")
    print("      FACEATTEND AI — REAL-TIME ATTENDANCE")
    print("==================================================")
    print("Light-brown facial wireframe: ACTIVE")
    print("Automated attendance recording: ACTIVE")
    print("Press Q or ESC to exit.")
    print("==================================================")

    last_attendance_update = 0
    video_timestamp = 0

    fps_value = 0
    frame_counter = 0
    last_fps_time = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        video_timestamp += 1
        output = frame.copy()
        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # 1. MediaPipe Facial Wireframe Tracking
        # ----------------------------------------------------
        proc_w = 640
        proc_h = int(height * proc_w / width)
        proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        rgb_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_proc)

        detected_faces = []
        try:
            face_result = face_landmarker.detect_for_video(mp_img, video_timestamp)
            if face_result.face_landmarks:
                detected_faces = face_result.face_landmarks
        except Exception:
            pass

        # Draw light-brown geometric wireframe on each face
        for face_landmarks in detected_faces:
            output = draw_light_brown_wireframe(output, face_landmarks)

        # ----------------------------------------------------
        # 2. Face Recognition & Attendance Marking
        # ----------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        haar_faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(90, 90)
        )

        recognized_student = None
        recognized_status = "SEARCHING FACE..."
        status_color = (140, 180, 210)

        for (x, y, w, h) in haar_faces:
            face_roi = gray[y:y + h, x:x + w]
            face_roi = cv2.resize(face_roi, (200, 200))

            label, distance = recognizer.predict(face_roi)

            if label in labels and distance <= RECOGNITION_THRESHOLD:
                s_id = labels[label]["id"]
                s_name = labels[label]["name"]
                recognized_student = (s_id, s_name)

                # Periodic CSV update
                cur_time = time.time()
                if cur_time - last_attendance_update >= UPDATE_INTERVAL:
                    mark_attendance(s_id, s_name)
                    last_attendance_update = cur_time

                # Student recognized overlay
                recognized_status = f"PRESENT: {s_name} ({s_id})"
                status_color = COLOR_SUCCESS

                badge_y = max(30, y - 24)
                badge_text = f"PRESENT | {s_id} - {s_name} (Dist: {distance:.1f})"

                # Background capsule
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output, (x - 6, badge_y - th - 8), (x + tw + 10, badge_y + 6), (22, 28, 38), -1)
                cv2.rectangle(output, (x - 6, badge_y - th - 8), (x + tw + 10, badge_y + 6), COLOR_SUCCESS, 2)
                cv2.putText(output, badge_text, (x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 250, 255), 2, cv2.LINE_AA)
            else:
                recognized_status = "UNKNOWN FACE DETECTED"
                status_color = COLOR_UNKNOWN

                badge_y = max(30, y - 24)
                badge_text = f"UNKNOWN | (Dist: {distance:.1f})"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(output, (x - 6, badge_y - th - 8), (x + tw + 10, badge_y + 6), (22, 28, 38), -1)
                cv2.rectangle(output, (x - 6, badge_y - th - 8), (x + tw + 10, badge_y + 6), COLOR_UNKNOWN, 2)
                cv2.putText(output, badge_text, (x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 255), 2, cv2.LINE_AA)

        # ----------------------------------------------------
        # 3. Executive UI Overlays (Top Bar & Bottom Bar)
        # ----------------------------------------------------
        frame_counter += 1
        curr_t = time.time()
        if curr_t - last_fps_time >= 1.0:
            fps_value = int(frame_counter / (curr_t - last_fps_time))
            frame_counter = 0
            last_fps_time = curr_t

        # Top Bar
        cv2.rectangle(output, (0, 0), (width, 76), (24, 20, 17), -1)
        cv2.line(output, (0, 76), (width, 76), (108, 154, 196), 2, cv2.LINE_AA)

        cv2.putText(output, "FACEATTEND AI", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (246, 237, 228), 2, cv2.LINE_AA)
        cv2.putText(output, "LIVE ATTENDANCE & GEOMETRIC WIREFRAME", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (115, 163, 212), 1, cv2.LINE_AA)

        now_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(output, now_str, (width - 290, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 225, 215), 1, cv2.LINE_AA)
        cv2.putText(output, f"FPS: {fps_value}  |  Faces: {len(detected_faces)}", (width - 230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (115, 163, 212), 1, cv2.LINE_AA)

        # Bottom Bar
        cv2.rectangle(output, (0, height - 52), (width, height), (24, 20, 17), -1)
        cv2.line(output, (0, height - 52), (width, height - 52), (108, 154, 196), 2, cv2.LINE_AA)

        cv2.putText(output, recognized_status, (24, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
        cv2.putText(output, "Q / ESC: EXIT ATTENDANCE", (width - 250, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 210, 200), 1, cv2.LINE_AA)

        cv2.imshow(window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break

    camera.release()
    face_landmarker.close()
    cv2.destroyAllWindows()
    print("Attendance session ended.")

if __name__ == "__main__":
    main()
