from ultralytics import YOLO

import cv2
import threading
import time
import os
import urllib.request
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# FACEATTEND AI
# LIGHT BROWN FACIAL MESH, EYES, MOUTH & LANDMARKS
# ============================================================


# ============================================================
# 1. YOLO MODEL
# ============================================================

YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"

print("Loading YOLOv8 model...")

model = YOLO(YOLO_MODEL_PATH)

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# 2. MEDIAPIPE MODEL
# ============================================================

FACE_MODEL_PATH = "face_landmarker.task"

FACE_MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def download_face_model():

    if os.path.exists(FACE_MODEL_PATH):

        print("Face landmark model found.")

        return

    print()
    print("Face landmark model not found.")
    print("Downloading automatically...")
    print()

    try:

        urllib.request.urlretrieve(
            FACE_MODEL_URL,
            FACE_MODEL_PATH
        )

        print("Face landmark model downloaded.")

    except Exception as e:

        print("ERROR: Could not download face landmark model.")
        print(e)

        exit()


download_face_model()


# ============================================================
# 3. MEDIAPIPE FACE LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=FACE_MODEL_PATH
)

face_options = vision.FaceLandmarkerOptions(

    base_options=base_options,

    running_mode=vision.RunningMode.VIDEO,

    num_faces=5,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    output_face_blendshapes=False,

    output_facial_transformation_matrixes=False
)

face_landmarker = vision.FaceLandmarker.create_from_options(
    face_options
)


# ============================================================
# 4. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


# ============================================================
# 5. CAMERA SETTINGS
# ============================================================

# Request 1280x720.
# This is a good balance between quality and FPS.

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

# MJPEG
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(
        "M",
        "J",
        "P",
        "G"
    )
)

# Reduce camera buffer
cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)

# Autofocus
cap.set(
    cv2.CAP_PROP_AUTOFOCUS,
    1
)


# ============================================================
# 6. CHECK CAMERA
# ============================================================

if not cap.isOpened():

    print()
    print("ERROR: Webcam could not be opened.")
    print()

    exit()


# ============================================================
# 7. ACTUAL CAMERA RESOLUTION
# ============================================================

actual_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

actual_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print()
print(
    "Camera resolution:",
    actual_width,
    "x",
    actual_height
)
print()


# ============================================================
# 8. PROGRAM VARIABLES
# ============================================================

running = True

latest_frame = None

result_frame = None

lock = threading.Lock()

face_count = 0

fps_value = 0

video_timestamp = 0

show_mesh = True

# ============================================================
# 9. LIGHT BROWN THEME & SPACIOUS STRUCTURAL LANDMARKS
# ============================================================

from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections

# Helper to reconstruct sequential loop of landmark indices
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

# Ordered loops for facial features
OVAL_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
LEFT_EYE_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
RIGHT_EYE_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
LEFT_EYEBROW_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
RIGHT_EYEBROW_LOOP = _build_loop(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)

# Outer lip sequential contour
OUTER_LIP_LOOP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

# Nose bridge & nostrils indices
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]
NOSE_WINGS = [98, 97, 2, 326, 327]

# Selected well-spaced structural mesh vertices (~80 key points for clean, professional spacing)
STRUCTURAL_MESH_INDICES = sorted(list(set(
    OVAL_LOOP[::2] +                         # 18 points along face contour
    LEFT_EYE_LOOP[::2] +                     # 8 points around left eye
    RIGHT_EYE_LOOP[::2] +                    # 8 points around right eye
    LEFT_EYEBROW_LOOP +                      # 5 points on left brow
    RIGHT_EYEBROW_LOOP +                     # 5 points on right brow
    OUTER_LIP_LOOP[::2] +                    # 10 points on outer lips
    NOSE_BRIDGE + NOSE_WINGS +               # Nose points
    [10, 67, 109, 108, 151, 337, 338, 297] + # Upper forehead anchors
    [50, 118, 123, 187, 205] +               # Left cheek anchors
    [280, 347, 352, 411, 425] +              # Right cheek anchors
    [152, 175, 199, 200]                     # Chin / jaw interior
)))

# Elegant Light Brown Theme Palette (OpenCV BGR Format)
COLOR_LIGHT_BROWN_MESH = (135, 180, 225)      # Clean, spacious light brown wireframe
COLOR_LIGHT_BROWN_CONTOUR = (85, 145, 225)    # Rich light golden-brown contour
COLOR_EYES = (110, 185, 245)                  # Luminous golden light brown for eyes
COLOR_LIPS = (90, 140, 225)                   # Warm amber-caramel for lips/mouth


# ============================================================
# 10. DRAW CLEAN PROFESSIONAL SPACIOUS MESH & CONTOURS
# ============================================================

def draw_light_brown_face(image, face_landmarks):
    height, width = image.shape[:2]

    # Convert normalized landmarks to pixel coordinates
    all_pts = []
    for landmark in face_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        all_pts.append((x, y))

    total_pts = len(all_pts)
    if total_pts < 468:
        return image

    # 1. Prepare feature polygons for cutout exclusion
    left_eye_poly = np.array([all_pts[i] for i in LEFT_EYE_LOOP], dtype=np.int32)
    right_eye_poly = np.array([all_pts[i] for i in RIGHT_EYE_LOOP], dtype=np.int32)
    mouth_poly = np.array([all_pts[i] for i in OUTER_LIP_LOOP], dtype=np.int32)
    face_hull = cv2.convexHull(np.array([all_pts[i] for i in OVAL_LOOP], dtype=np.int32))

    # 2. Spacious, Professional Delaunay Facial Mesh
    if show_mesh:
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

                    p1, p2, p3 = (x1, y1), (x2, y2), (x3, y3)

                    # Triangle center
                    cx = (x1 + x2 + x3) // 3
                    cy = (y1 + y2 + y3) // 3

                    # Filter: Must be inside face boundary
                    if cv2.pointPolygonTest(face_hull, (cx, cy), False) < 0:
                        continue

                    # Filter: Leave eyes completely open and clear!
                    if cv2.pointPolygonTest(left_eye_poly, (cx, cy), False) >= 0:
                        continue
                    if cv2.pointPolygonTest(right_eye_poly, (cx, cy), False) >= 0:
                        continue

                    # Filter: Leave mouth completely open and clear!
                    if cv2.pointPolygonTest(mouth_poly, (cx, cy), False) >= 0:
                        continue

                    # Filter: Avoid stretched exterior triangles
                    d12 = np.hypot(x2 - x1, y2 - y1)
                    d23 = np.hypot(x3 - x2, y3 - y2)
                    d31 = np.hypot(x1 - x3, y1 - y3)
                    if d12 > max_edge or d23 > max_edge or d31 > max_edge:
                        continue

                    # Draw clean, spacious light brown triangle facets
                    cv2.line(image, p1, p2, COLOR_LIGHT_BROWN_MESH, 1, cv2.LINE_AA)
                    cv2.line(image, p2, p3, COLOR_LIGHT_BROWN_MESH, 1, cv2.LINE_AA)
                    cv2.line(image, p3, p1, COLOR_LIGHT_BROWN_MESH, 1, cv2.LINE_AA)

    # 3. Clean, Prominent Feature Contours (Eyes, Eyebrows, Lips, Nose, Oval)
    # Eyes (open, clean outline)
    cv2.polylines(image, [left_eye_poly], True, COLOR_EYES, 2, cv2.LINE_AA)
    cv2.polylines(image, [right_eye_poly], True, COLOR_EYES, 2, cv2.LINE_AA)

    # Eyebrows
    lbrow_pts = np.array([all_pts[i] for i in LEFT_EYEBROW_LOOP], dtype=np.int32)
    rbrow_pts = np.array([all_pts[i] for i in RIGHT_EYEBROW_LOOP], dtype=np.int32)
    cv2.polylines(image, [lbrow_pts], False, COLOR_LIGHT_BROWN_CONTOUR, 2, cv2.LINE_AA)
    cv2.polylines(image, [rbrow_pts], False, COLOR_LIGHT_BROWN_CONTOUR, 2, cv2.LINE_AA)

    # Lips (crisp outer contour)
    cv2.polylines(image, [mouth_poly], True, COLOR_LIPS, 2, cv2.LINE_AA)

    # Nose Bridge
    for k in range(len(NOSE_BRIDGE) - 1):
        p_a = all_pts[NOSE_BRIDGE[k]]
        p_b = all_pts[NOSE_BRIDGE[k + 1]]
        cv2.line(image, p_a, p_b, COLOR_LIGHT_BROWN_CONTOUR, 2, cv2.LINE_AA)

    # Face Oval (outer jawline & temples silhouette)
    oval_pts = np.array([all_pts[i] for i in OVAL_LOOP], dtype=np.int32)
    cv2.polylines(image, [oval_pts], True, COLOR_LIGHT_BROWN_CONTOUR, 2, cv2.LINE_AA)

    return image


# ============================================================
# 11. CAMERA THREAD
# ============================================================

def camera_thread():

    global latest_frame
    global running

    while running:

        ret, frame = cap.read()

        if ret:

            with lock:

                latest_frame = frame

        else:

            time.sleep(
                0.01
            )


# ============================================================
# 12. AI PROCESSING THREAD
# ============================================================

def ai_thread():

    global latest_frame
    global result_frame
    global running
    global face_count
    global fps_value
    global video_timestamp

    last_fps_time = time.time()

    frame_counter = 0

    while running:

        # ----------------------------------------------------
        # Get latest camera frame
        # ----------------------------------------------------

        with lock:

            if latest_frame is None:

                continue

            frame = latest_frame.copy()

        # ----------------------------------------------------
        # Keep original camera image
        # ----------------------------------------------------

        output = frame.copy()

        # ----------------------------------------------------
        # MediaPipe processing resolution
        #
        # The webcam display remains high quality.
        # Only AI processing is reduced.
        # ----------------------------------------------------

        process_width = 640

        process_height = int(
            frame.shape[0]
            * process_width
            / frame.shape[1]
        )

        process_frame = cv2.resize(
            frame,
            (
                process_width,
                process_height
            ),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # Convert BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            process_frame,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # MediaPipe image
        # ----------------------------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        video_timestamp += 1

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        try:

            face_result = (
                face_landmarker.detect_for_video(
                    mp_image,
                    video_timestamp
                )
            )

        except Exception as e:

            print(
                "MediaPipe error:",
                e
            )

            continue

        # ----------------------------------------------------
        # Faces
        # ----------------------------------------------------

        if face_result.face_landmarks:

            detected_faces = (
                face_result.face_landmarks
            )

        else:

            detected_faces = []

        face_count = len(
            detected_faces
        )

        # ----------------------------------------------------
        # Draw light brown facial mesh & features
        # ----------------------------------------------------

        for face_landmarks in detected_faces:

            output = draw_light_brown_face(
                output,
                face_landmarks
            )

        # ====================================================
        # YOLOv8
        # ====================================================

        try:

            yolo_frame = cv2.resize(
                frame,
                (320, 240)
            )

            model.predict(
                yolo_frame,
                conf=CONFIDENCE_THRESHOLD,
                imgsz=96,
                verbose=False
            )

        except Exception as e:

            print(
                "YOLO error:",
                e
            )

        # ====================================================
        # FPS
        # ====================================================

        frame_counter += 1

        current_time = time.time()

        elapsed = (
            current_time
            - last_fps_time
        )

        if elapsed >= 1.0:

            fps_value = int(
                frame_counter
                / elapsed
            )

            frame_counter = 0

            last_fps_time = (
                current_time
            )

        # ====================================================
        # USER INTERFACE
        # ====================================================

        height, width = output.shape[:2]

        # ----------------------------------------------------
        # TOP BAR (Dark Espresso with Light Brown Accent)
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (0, 0),
            (width, 82),
            (22, 28, 38),
            -1
        )
        cv2.line(
            output,
            (0, 82),
            (width, 82),
            (115, 175, 225),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        cv2.putText(
            output,
            "FACEATTEND AI",
            (28, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (235, 242, 248),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # SUBTITLE
        # ----------------------------------------------------

        cv2.putText(
            output,
            "LIGHT BROWN FACIAL SCAN",
            (30, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 180, 230),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # FACE COUNT
        # ----------------------------------------------------

        cv2.putText(
            output,
            f"Faces: {face_count}",
            (
                width - 280,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (130, 190, 240),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        cv2.putText(
            output,
            f"FPS: {fps_value}",
            (
                width - 120,
                40
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (110, 195, 245),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # BOTTOM BAR (Dark Espresso with Light Brown Accent)
        # ----------------------------------------------------

        cv2.rectangle(
            output,
            (
                0,
                height - 58
            ),
            (
                width,
                height
            ),
            (22, 28, 38),
            -1
        )
        cv2.line(
            output,
            (0, height - 58),
            (width, height - 58),
            (115, 175, 225),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if face_count > 0:

            status = (
                "FACE DETECTED"
            )

            status_color = (
                110,
                195,
                245
            )

        else:

            status = (
                "SEARCHING FACE..."
            )

            status_color = (
                140,
                180,
                210
            )

        cv2.putText(
            output,
            status,
            (
                25,
                height - 22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # SHORTCUTS / EXIT
        # ----------------------------------------------------

        cv2.putText(
            output,
            "M: TOGGLE MESH   |   Q / ESC: EXIT",
            (
                width - 340,
                height - 22
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (215, 230, 245),
            1,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Save output
        # ----------------------------------------------------

        with lock:

            result_frame = output


# ============================================================
# 13. START CAMERA THREAD
# ============================================================

camera = threading.Thread(
    target=camera_thread,
    daemon=True
)

camera.start()


# ============================================================
# 14. START AI THREAD
# ============================================================

processor = threading.Thread(
    target=ai_thread,
    daemon=True
)

processor.start()


# ============================================================
# 15. START MESSAGE
# ============================================================

print()
print("==========================================")
print("          FACEATTEND AI")
print("==========================================")
print()
print("High quality webcam: ON")
print("Light brown facial mesh & contours: ON")
print("MediaPipe Face Landmarker: ON")
print("YOLOv8: ON")
print()
print(
    "Camera:",
    actual_width,
    "x",
    actual_height
)
print()
print("Controls:")
print("  Press M to toggle face mesh wireframe")
print("  Press Q or ESC to exit")
print()
print("==========================================")


# ============================================================
# 16. WINDOW
# ============================================================

window_name = (
    "FaceAttend AI - Light Brown Facial Scan"
)

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    1280,
    720
)


# ============================================================
# 17. DISPLAY LOOP
# ============================================================

while running:

    with lock:

        if result_frame is not None:

            display = (
                result_frame.copy()
            )

        elif latest_frame is not None:

            display = (
                latest_frame.copy()
            )

        else:

            continue

    # --------------------------------------------------------
    # Show webcam
    # --------------------------------------------------------

    cv2.imshow(
        window_name,
        display
    )

    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    # M / m - Toggle mesh lines
    if key == ord("m") or key == ord("M"):

        show_mesh = not show_mesh

        print(
            f"Face mesh: {'ON' if show_mesh else 'OFF'}"
        )

    # Q
    if key == ord("q"):

        print(
            "Q pressed."
        )

        running = False

        break

    # Q uppercase
    if key == ord("Q"):

        print(
            "Q pressed."
        )

        running = False

        break

    # ESC
    if key == 27:

        print(
            "ESC pressed."
        )

        running = False

        break

    # --------------------------------------------------------
    # Window X button
    # --------------------------------------------------------

    try:

        if cv2.getWindowProperty(
            window_name,
            cv2.WND_PROP_VISIBLE
        ) < 1:

            print(
                "Window closed."
            )

            running = False

            break

    except:

        running = False

        break


# ============================================================
# 18. CLEANUP
# ============================================================

print()
print("Closing FaceAttend AI...")

running = False

time.sleep(
    0.5
)

cap.release()

face_landmarker.close()

cv2.destroyAllWindows()

print()
print("Camera closed.")
print("Program finished.")