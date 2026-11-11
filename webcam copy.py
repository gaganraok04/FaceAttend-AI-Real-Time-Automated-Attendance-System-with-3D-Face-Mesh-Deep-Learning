from ultralytics import YOLO
import cv2
import threading
import time

# ============================================================
# 1. LOAD TRAINED YOLOv8 MODEL
# ============================================================

model = YOLO("runs/detect/train/weights/best.pt")


# ============================================================
# 2. CONFIDENCE THRESHOLD
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# 3. OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Webcam resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Reduce camera buffer
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# Check webcam
if not cap.isOpened():
    print("ERROR: Webcam could not be opened.")
    exit()


# ============================================================
# 4. PROGRAM VARIABLES
# ============================================================

running = True

latest_frame = None
result_frame = None

lock = threading.Lock()


# ============================================================
# 5. CAMERA THREAD
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

            time.sleep(0.01)


# ============================================================
# 6. YOLO DETECTION THREAD
# ============================================================

def yolo_thread():

    global latest_frame
    global result_frame
    global running

    while running:

        # Get latest camera frame
        with lock:

            if latest_frame is None:
                continue

            frame = latest_frame.copy()


        # ----------------------------------------------------
        # Resize image for faster CPU processing
        # ----------------------------------------------------

        small_frame = cv2.resize(
            frame,
            (320, 240)
        )


        # ----------------------------------------------------
        # YOLO prediction
        # ----------------------------------------------------

        results = model.predict(

            small_frame,

            conf=CONFIDENCE_THRESHOLD,

            imgsz=96,

            verbose=False
        )


        # ----------------------------------------------------
        # Draw detection boxes and names
        # ----------------------------------------------------

        output = results[0].plot()


        # ----------------------------------------------------
        # Resize result for larger display
        # ----------------------------------------------------

        output = cv2.resize(
            output,
            (900, 700)
        )


        # Save result
        with lock:
            result_frame = output


# ============================================================
# 7. START CAMERA THREAD
# ============================================================

camera = threading.Thread(

    target=camera_thread,

    daemon=True
)

camera.start()


# ============================================================
# 8. START YOLO THREAD
# ============================================================

detector = threading.Thread(

    target=yolo_thread,

    daemon=True
)

detector.start()


# ============================================================
# 9. START MESSAGE
# ============================================================

print("======================================")
print(" STUDENT ATTENDANCE SYSTEM")
print("======================================")

print("Webcam started!")
print("YOLO detection started!")
print("Confidence threshold:", CONFIDENCE_THRESHOLD)

print("Press Q or ESC to quit.")


# ============================================================
# 10. CREATE LARGE WINDOW
# ============================================================

window_name = "Student Attendance - YOLOv8"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    900,
    700
)


# ============================================================
# 11. MAIN DISPLAY LOOP
# ============================================================

while running:

    # Get latest result
    with lock:

        if result_frame is not None:

            display = result_frame.copy()

        elif latest_frame is not None:

            display = cv2.resize(
                latest_frame.copy(),
                (900, 700)
            )

        else:

            continue


    # Show webcam
    cv2.imshow(
        window_name,
        display
    )


    # Keyboard
    key = cv2.pollKey()


    # --------------------------------------------------------
    # Q / q / ESC = EXIT
    # --------------------------------------------------------

    if key == ord("q") or key == ord("Q") or key == 27:

        print("Exit key pressed.")

        running = False

        break


    # --------------------------------------------------------
    # X BUTTON = EXIT
    # --------------------------------------------------------

    try:

        if cv2.getWindowProperty(
            window_name,
            cv2.WND_PROP_VISIBLE
        ) < 1:

            print("Window closed.")

            running = False

            break

    except:

        running = False

        break


# ============================================================
# 12. CLEANUP
# ============================================================

running = False

time.sleep(0.5)

cap.release()

cv2.destroyAllWindows()

print("Webcam closed successfully!")
print("Program finished.")