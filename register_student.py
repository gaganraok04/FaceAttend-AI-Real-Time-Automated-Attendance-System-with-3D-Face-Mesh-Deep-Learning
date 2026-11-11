import cv2
import os
import csv

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "registered_faces"
STUDENT_FILE = "students.csv"

# Number of face images to capture
IMAGE_COUNT = 20

# Create folder
os.makedirs(DATASET_PATH, exist_ok=True)


# ==============================
# STUDENT DETAILS
# ==============================

student_id = input("Enter Student ID: ").strip()
student_name = input("Enter Student Name: ").strip()

if student_id == "" or student_name == "":
    print("Student ID and Name are required!")
    exit()


# ==============================
# CREATE STUDENT FOLDER
# ==============================

student_folder = os.path.join(
    DATASET_PATH,
    student_id
)

os.makedirs(student_folder, exist_ok=True)


# ==============================
# SAVE STUDENT INFORMATION
# ==============================

file_exists = os.path.exists(STUDENT_FILE)

with open(
    STUDENT_FILE,
    "a",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    if not file_exists:
        writer.writerow([
            "Student ID",
            "Student Name"
        ])

    writer.writerow([
        student_id,
        student_name
    ])


# ==============================
# LOAD FACE DETECTOR
# ==============================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("ERROR: Face detector could not be loaded.")
    exit()


# ==============================
# OPEN WEBCAM
# ==============================

camera = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()


print()
print("======================================")
print("       STUDENT REGISTRATION")
print("======================================")
print("Student ID:", student_id)
print("Student Name:", student_name)
print()
print("Look at the camera.")
print("Face the camera directly.")
print("Move your face slightly.")
print("Press Q to cancel.")
print()


# ==============================
# CAPTURE FACE IMAGES
# ==============================

count = 0

while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not read webcam.")
        break


    # Convert to grayscale
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )


    # Draw face boxes
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Face detected",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # Information on screen
    cv2.putText(
        frame,
        f"Student: {student_name}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Captured: {count}/{IMAGE_COUNT}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to cancel",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Show webcam
    cv2.imshow(
        "Student Registration",
        frame
    )


    # Automatically capture when face is detected
    if len(faces) == 1:

        x, y, w, h = faces[0]

        face_image = gray[
            y:y + h,
            x:x + w
        ]


        filename = os.path.join(
            student_folder,
            f"{count + 1}.jpg"
        )


        cv2.imwrite(
            filename,
            face_image
        )


        count += 1

        print(
            f"Captured image {count}/{IMAGE_COUNT}"
        )


        # Small delay between captures
        cv2.waitKey(150)


    # Finish
    if count >= IMAGE_COUNT:

        print()
        print("======================================")
        print("Registration completed!")
        print("Student ID:", student_id)
        print("Student Name:", student_name)
        print("Images captured:", count)
        print("======================================")

        break


    # Q = cancel
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):

        print("Registration cancelled.")
        break


# ==============================
# CLEANUP
# ==============================

camera.release()
cv2.destroyAllWindows()
