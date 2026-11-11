import cv2
import os
import csv
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = "registered_faces"
STUDENT_FILE = "students.csv"

MODEL_FILE = "face_model.yml"
LABEL_FILE = "face_labels.txt"


# ============================================================
# CHECK LBPH
# ============================================================

if not hasattr(cv2, "face"):
    print("ERROR: OpenCV Face module is not available.")
    print("Install opencv-contrib-python.")
    exit()


# ============================================================
# LOAD STUDENT INFORMATION
# ============================================================

students = {}

if os.path.exists(STUDENT_FILE):

    with open(
        STUDENT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            student_id = row["Student ID"]
            student_name = row["Student Name"]

            students[student_id] = student_name

else:

    print("ERROR: students.csv not found.")
    exit()


# ============================================================
# PREPARE TRAINING DATA
# ============================================================

faces = []
labels = []

label_map = {}

current_label = 0


print()
print("======================================")
print("       TRAINING FACE MODEL")
print("======================================")
print()


# ============================================================
# READ STUDENT FOLDERS
# ============================================================

for student_id in os.listdir(DATASET_PATH):

    student_folder = os.path.join(
        DATASET_PATH,
        student_id
    )

    # Skip files
    if not os.path.isdir(student_folder):
        continue

    # Check student exists in CSV
    if student_id not in students:

        print(
            "WARNING: Student ID",
            student_id,
            "not found in students.csv"
        )

        continue


    student_name = students[student_id]


    print(
        f"Student: {student_name} "
        f"(ID: {student_id})"
    )


    # Assign numeric label
    label_map[current_label] = (
        student_id,
        student_name
    )


    image_count = 0


    # ========================================================
    # READ IMAGES
    # ========================================================

    for filename in os.listdir(student_folder):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue


        image_path = os.path.join(
            student_folder,
            filename
        )


        # Read grayscale image
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )


        if image is None:

            print(
                "Could not read:",
                image_path
            )

            continue


        # Resize all images to same size
        image = cv2.resize(
            image,
            (200, 200)
        )


        faces.append(image)
        labels.append(current_label)

        image_count += 1


    print(
        "Images loaded:",
        image_count
    )

    print()


    current_label += 1


# ============================================================
# CHECK TRAINING DATA
# ============================================================

if len(faces) == 0:

    print("ERROR: No face images found.")
    exit()


print("======================================")
print("Training images:", len(faces))
print("Students:", len(label_map))
print("======================================")
print()


# ============================================================
# CREATE LBPH RECOGNIZER
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create(

    radius=1,

    neighbors=8,

    grid_x=8,

    grid_y=8
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("Training started...")
print("Please wait...")
print()


recognizer.train(
    faces,
    np.array(
        labels,
        dtype=np.int32
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

recognizer.write(
    MODEL_FILE
)


# ============================================================
# SAVE LABEL INFORMATION
# ============================================================

with open(
    LABEL_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "Label,Student ID,Student Name\n"
    )

    for label, data in label_map.items():

        student_id, student_name = data

        file.write(
            f"{label},{student_id},{student_name}\n"
        )


# ============================================================
# TRAINING COMPLETED
# ============================================================

print()
print("======================================")
print("       TRAINING COMPLETED!")
print("======================================")

print(
    "Students trained:",
    len(label_map)
)

print(
    "Images trained:",
    len(faces)
)

print()
print("Model saved as:")
print(MODEL_FILE)

print()
print("Labels saved as:")
print(LABEL_FILE)

print("======================================")