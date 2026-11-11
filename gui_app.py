"""
FaceAttend AI - Desktop Management Dashboard
A unified Graphical User Interface (GUI) for the Student Attendance System.
Enables 1-click execution for registration, model training, live attendance,
AI face mesh scanning, and record management without typing terminal commands.
"""

import os
import sys
import csv
import threading
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np

# ============================================================
# CONSTANTS & FILE PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STUDENT_FILE = os.path.join(BASE_DIR, "students.csv")
ATTENDANCE_FILE = os.path.join(BASE_DIR, "attendance.csv")
MODEL_FILE = os.path.join(BASE_DIR, "face_model.yml")
LABEL_FILE = os.path.join(BASE_DIR, "face_labels.txt")
DATASET_PATH = os.path.join(BASE_DIR, "registered_faces")

PYTHON_EXE = sys.executable

# ============================================================
# THEME PALETTE (Executive Light Brown & Warm Espresso)
# ============================================================

BG_MAIN = "#191512"            # Deep Roasted Espresso main background
BG_HEADER = "#241d18"          # Warm Mocha header bar
BG_CARD = "#29211b"            # Card surface (warm walnut)
BG_CARD_HOVER = "#382e26"      # Hover highlight
BG_INPUT = "#15110e"           # Deep espresso input fields

BORDER_COLOR = "#4d3d32"       # Warm light-brown card borders
BORDER_ACCENT = "#c49a6c"      # Luminous golden-brown accent line

TEXT_MAIN = "#f6ede4"          # Clean warm ivory/cream
TEXT_MUTED = "#bfaea0"         # Warm light mocha for labels/subtitles

ACCENT_BROWN = "#d4a373"       # Primary light golden brown
ACCENT_LIGHT = "#faedcd"       # Warm champagne highlight
ACCENT_WARM = "#e8c39e"        # Soft light caramel
ACCENT_GOLD = "#e9c46a"        # Luminous warm amber

# Professional Cohesive Light Brown & Bronze Buttons
BTN_HERO_BG = "#c48344"        # Primary Action: Radiant Light Caramel Brown
BTN_HERO_HOVER = "#db9856"

BTN_MESH_BG = "#a26e3c"        # Light Bronze Brown
BTN_MESH_HOVER = "#b88048"

BTN_REGISTER_BG = "#b07a48"    # Golden Chestnut Brown
BTN_REGISTER_HOVER = "#c58c57"

BTN_TRAIN_BG = "#935f37"       # Warm Terracotta Bronze
BTN_TRAIN_HOVER = "#a87143"

BTN_SECONDARY_BG = "#3d3027"   # Deep Warm Mocha
BTN_SECONDARY_HOVER = "#4f3f33"

FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_SUBTITLE = ("Segoe UI", 10)
FONT_HEADING = ("Segoe UI", 11, "bold")
FONT_BODY = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_SMALL = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)


class FaceAttendGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceAttend AI — Automated Attendance Dashboard")
        self.root.geometry("1180x750")
        self.root.minsize(1050, 680)
        self.root.configure(bg=BG_MAIN)

        # Active process tracking
        self.active_process = None

        self._configure_styles()
        self._build_header()
        self._build_layout()
        self._refresh_stats()
        self._load_attendance_table()
        self._load_students_table()
        self.log("FaceAttend AI Dashboard initialized and ready.")

    # ========================================================
    # STYLES & SETUP
    # ========================================================

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        # Global Treeview styling (Warm Roasted Walnut & Light Brown)
        style.configure(
            "Custom.Treeview",
            background=BG_CARD,
            foreground=TEXT_MAIN,
            fieldbackground=BG_CARD,
            rowheight=30,
            font=FONT_BODY,
            borderwidth=0
        )
        style.configure(
            "Custom.Treeview.Heading",
            background="#201915",
            foreground=ACCENT_BROWN,
            font=FONT_BOLD,
            borderwidth=1,
            relief="flat"
        )
        style.map(
            "Custom.Treeview",
            background=[("selected", "#544133")],
            foreground=[("selected", "#ffffff")]
        )

        # Notebook tabs (Warm Coffee Palette)
        style.configure(
            "Custom.TNotebook",
            background=BG_MAIN,
            borderwidth=0
        )
        style.configure(
            "Custom.TNotebook.Tab",
            background="#241d18",
            foreground=TEXT_MUTED,
            font=FONT_BOLD,
            padding=[18, 9]
        )
        style.map(
            "Custom.TNotebook.Tab",
            background=[("selected", "#382c23")],
            foreground=[("selected", ACCENT_LIGHT)]
        )

    # ========================================================
    # HEADER & METRIC CHIPS
    # ========================================================

    def _build_header(self):
        header_frame = tk.Frame(self.root, bg=BG_HEADER, height=76)
        header_frame.pack(fill=tk.X, side=tk.TOP)
        header_frame.pack_propagate(False)

        # Luminous Light Brown Bottom Accent Border Line
        divider = tk.Frame(header_frame, bg=BORDER_ACCENT, height=2)
        divider.pack(fill=tk.X, side=tk.BOTTOM)

        # Title block
        title_box = tk.Frame(header_frame, bg=BG_HEADER)
        title_box.pack(side=tk.LEFT, padx=22, pady=12)

        lbl_title = tk.Label(
            title_box,
            text="🎓 FACEATTEND AI",
            font=FONT_TITLE,
            fg=TEXT_MAIN,
            bg=BG_HEADER
        )
        lbl_title.pack(anchor="w")

        lbl_sub = tk.Label(
            title_box,
            text="Automated Student Attendance & Facial Recognition System",
            font=FONT_SUBTITLE,
            fg=ACCENT_BROWN,
            bg=BG_HEADER
        )
        lbl_sub.pack(anchor="w")

        # Metric Stats Chips on the right
        stats_box = tk.Frame(header_frame, bg=BG_HEADER)
        stats_box.pack(side=tk.RIGHT, padx=22, pady=12)

        self.chip_model = self._create_stat_chip(stats_box, "MODEL STATUS", "Checking...", ACCENT_GOLD)
        self.chip_model.pack(side=tk.LEFT, padx=8)

        self.chip_students = self._create_stat_chip(stats_box, "REGISTERED STUDENTS", "0", TEXT_MAIN)
        self.chip_students.pack(side=tk.LEFT, padx=8)

        self.chip_today = self._create_stat_chip(stats_box, "TODAY'S ATTENDANCE", "0", ACCENT_BROWN)
        self.chip_today.pack(side=tk.LEFT, padx=8)

    def _create_stat_chip(self, parent, label, value, val_color):
        chip = tk.Frame(parent, bg="#1b1511", padx=14, pady=6, highlightbackground=BORDER_COLOR, highlightthickness=1)
        lbl = tk.Label(chip, text=label, font=("Segoe UI", 7, "bold"), fg=TEXT_MUTED, bg="#1b1511")
        lbl.pack(anchor="center")
        val = tk.Label(chip, text=value, font=("Segoe UI", 11, "bold"), fg=val_color, bg="#1b1511")
        val.pack(anchor="center")
        chip.val_lbl = val
        return chip

    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    def _build_layout(self):
        container = tk.Frame(self.root, bg=BG_MAIN)
        container.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)

        # LEFT SIDE: Action Controls (width 370)
        left_panel = tk.Frame(container, bg=BG_MAIN, width=370)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        left_panel.pack_propagate(False)

        # RIGHT SIDE: Data Tables & Live Logs
        right_panel = tk.Frame(container, bg=BG_MAIN)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls(left_panel)
        self._build_data_views(right_panel)

    # ========================================================
    # CONTROLS PANEL (LEFT)
    # ========================================================

    def _build_controls(self, parent):
        # 1. CORE WORKFLOW ACTIONS CARD
        card_actions = self._create_card(parent, "🚀 MAIN ACTIONS")
        card_actions.pack(fill=tk.X, pady=(0, 10))

        btn_attend = self._create_button(
            card_actions,
            text="▶  START LIVE ATTENDANCE",
            command=self.run_attendance,
            bg=BTN_HERO_BG,
            hover_bg=BTN_HERO_HOVER
        )
        btn_attend.pack(fill=tk.X, padx=12, pady=(10, 6))

        btn_test = self._create_button(
            card_actions,
            text="🔍  TEST RECOGNITION (DISTANCE)",
            command=self.run_recognize_test,
            bg=BTN_SECONDARY_BG,
            hover_bg=BTN_SECONDARY_HOVER,
            fg=ACCENT_LIGHT,
            border_color=BORDER_COLOR
        )
        btn_test.pack(fill=tk.X, padx=12, pady=(0, 10))

        # 2. STUDENT REGISTRATION CARD
        card_reg = self._create_card(parent, "👤 REGISTER NEW STUDENT")
        card_reg.pack(fill=tk.X, pady=(0, 10))

        form_frame = tk.Frame(card_reg, bg=BG_CARD)
        form_frame.pack(fill=tk.X, padx=12, pady=(6, 4))

        tk.Label(form_frame, text="Student ID:", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")
        self.entry_id = tk.Entry(form_frame, font=FONT_BODY, bg=BG_INPUT, fg=TEXT_MAIN, insertbackground=TEXT_MAIN, relief="flat", highlightbackground=BORDER_COLOR, highlightthickness=1)
        self.entry_id.pack(fill=tk.X, pady=(2, 6), ipady=4)

        tk.Label(form_frame, text="Student Name:", font=FONT_SMALL, fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")
        self.entry_name = tk.Entry(form_frame, font=FONT_BODY, bg=BG_INPUT, fg=TEXT_MAIN, insertbackground=TEXT_MAIN, relief="flat", highlightbackground=BORDER_COLOR, highlightthickness=1)
        self.entry_name.pack(fill=tk.X, pady=(2, 8), ipady=4)

        btn_register = self._create_button(
            card_reg,
            text="📸  CAPTURE FACE & REGISTER",
            command=self.run_register_student,
            bg=BTN_REGISTER_BG,
            hover_bg=BTN_REGISTER_HOVER
        )
        btn_register.pack(fill=tk.X, padx=12, pady=(0, 10))

        # 3. MODEL TRAINING CARD
        card_train = self._create_card(parent, "🧠 MODEL TRAINING")
        card_train.pack(fill=tk.X, pady=(0, 10))

        btn_train = self._create_button(
            card_train,
            text="⚡  TRAIN LBPH FACE MODEL",
            command=self.run_train_model,
            bg=BTN_TRAIN_BG,
            hover_bg=BTN_TRAIN_HOVER
        )
        btn_train.pack(fill=tk.X, padx=12, pady=(8, 10))

        # 4. YOLOv8 UTILITIES CARD
        card_yolo = self._create_card(parent, "🤖 YOLOv8 TOOLS")
        card_yolo.pack(fill=tk.X)

        btn_yolo_test = self._create_button(
            card_yolo,
            text="📊  EVALUATE YOLOv8 MODEL (test.py)",
            command=self.run_yolo_test,
            bg="#342821",
            hover_bg="#46372d",
            fg="#dfcbba",
            border_color="#524032"
        )
        btn_yolo_test.pack(fill=tk.X, padx=12, pady=(6, 4))

        btn_yolo_cam = self._create_button(
            card_yolo,
            text="🎥  SIMPLE YOLO WEBCAM (copy.py)",
            command=self.run_yolo_webcam,
            bg="#342821",
            hover_bg="#46372d",
            fg="#dfcbba",
            border_color="#524032"
        )
        btn_yolo_cam.pack(fill=tk.X, padx=12, pady=(0, 10))

    def _create_card(self, parent, title):
        card = tk.Frame(parent, bg=BG_CARD, highlightbackground=BORDER_COLOR, highlightthickness=1)
        header = tk.Label(card, text=title, font=FONT_HEADING, fg=ACCENT_LIGHT, bg=BG_CARD, anchor="w")
        header.pack(fill=tk.X, padx=12, pady=(8, 4))
        return card

    def _create_button(self, parent, text, command, bg, hover_bg, fg="#ffffff", border_color=None):
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            font=FONT_BOLD,
            fg=fg,
            bg=bg,
            activebackground=hover_bg,
            activeforeground=fg,
            relief="flat",
            cursor="hand2",
            padx=10,
            pady=7,
            highlightbackground=border_color if border_color else bg,
            highlightthickness=1 if border_color else 0
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=hover_bg))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg))
        return btn

    # ========================================================
    # DATA VIEWS & LOGS (RIGHT)
    # ========================================================

    def _build_data_views(self, parent):
        self.notebook = ttk.Notebook(parent, style="Custom.TNotebook")
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Attendance Ledger
        tab_attendance = tk.Frame(self.notebook, bg=BG_CARD)
        self.notebook.add(tab_attendance, text="📅 Live Attendance Records")
        self._build_attendance_tab(tab_attendance)

        # Tab 2: Registered Students Roster
        tab_students = tk.Frame(self.notebook, bg=BG_CARD)
        self.notebook.add(tab_students, text="👥 Registered Students")
        self._build_students_tab(tab_students)

        # Bottom: Activity Log Console
        log_card = tk.Frame(parent, bg=BG_CARD, height=140, highlightbackground=BORDER_COLOR, highlightthickness=1)
        log_card.pack(fill=tk.X, pady=(10, 0))
        log_card.pack_propagate(False)

        log_head = tk.Frame(log_card, bg=BG_CARD)
        log_head.pack(fill=tk.X, padx=12, pady=(6, 2))
        tk.Label(log_head, text="🖥️ SYSTEM ACTIVITY LOG", font=FONT_SMALL, fg=ACCENT_BROWN, bg=BG_CARD).pack(side=tk.LEFT)

        btn_clear = tk.Button(log_head, text="Clear Log", font=("Segoe UI", 8), fg=TEXT_MUTED, bg="#332821", activebackground="#45372e", activeforeground=TEXT_MAIN, relief="flat", padx=6, pady=2, command=self.clear_log)
        btn_clear.pack(side=tk.RIGHT)

        self.txt_log = tk.Text(log_card, font=FONT_MONO, bg=BG_INPUT, fg="#dfcfbe", insertbackground=TEXT_MAIN, relief="flat", wrap="word")
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

    def _build_attendance_tab(self, parent):
        # Top Right Action Bar (Above Table)
        top_bar = tk.Frame(parent, bg=BG_CARD)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 4))

        lbl_section = tk.Label(top_bar, text="TODAY'S LOGGED SESSIONS", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD)
        lbl_section.pack(side=tk.LEFT, pady=2)

        btn_open = tk.Button(
            top_bar,
            text="📂 Open CSV",
            font=("Segoe UI", 8),
            fg=ACCENT_LIGHT,
            bg="#3a2e26",
            activebackground="#4d3e33",
            activeforeground=ACCENT_LIGHT,
            relief="flat",
            padx=8,
            pady=2,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
            cursor="hand2",
            command=lambda: self.open_file(ATTENDANCE_FILE)
        )
        btn_open.pack(side=tk.RIGHT, padx=(4, 0))

        btn_refresh = tk.Button(
            top_bar,
            text="🔄 Refresh",
            font=("Segoe UI", 8),
            fg=ACCENT_LIGHT,
            bg="#3a2e26",
            activebackground="#4d3e33",
            activeforeground=ACCENT_LIGHT,
            relief="flat",
            padx=8,
            pady=2,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
            cursor="hand2",
            command=self._load_attendance_table
        )
        btn_refresh.pack(side=tk.RIGHT, padx=4)

        # Table & Scrollbar
        cols = ("id", "name", "date", "first_seen", "last_seen", "status")
        self.tree_attendance = ttk.Treeview(parent, columns=cols, show="headings", style="Custom.Treeview")

        self.tree_attendance.heading("id", text="Student ID")
        self.tree_attendance.heading("name", text="Student Name")
        self.tree_attendance.heading("date", text="Date")
        self.tree_attendance.heading("first_seen", text="First Seen")
        self.tree_attendance.heading("last_seen", text="Last Seen")
        self.tree_attendance.heading("status", text="Status")

        self.tree_attendance.column("id", width=90, anchor="center")
        self.tree_attendance.column("name", width=180, anchor="w")
        self.tree_attendance.column("date", width=100, anchor="center")
        self.tree_attendance.column("first_seen", width=100, anchor="center")
        self.tree_attendance.column("last_seen", width=100, anchor="center")
        self.tree_attendance.column("status", width=90, anchor="center")

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.tree_attendance.yview)
        self.tree_attendance.configure(yscrollcommand=scrollbar.set)

        self.tree_attendance.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0, 10))

    def _build_students_tab(self, parent):
        # Top Right Action Bar (Above Table)
        top_bar = tk.Frame(parent, bg=BG_CARD)
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 4))

        lbl_section = tk.Label(top_bar, text="ROSTER DIRECTORY", font=("Segoe UI", 8, "bold"), fg=TEXT_MUTED, bg=BG_CARD)
        lbl_section.pack(side=tk.LEFT, pady=2)

        btn_open = tk.Button(
            top_bar,
            text="📂 Faces Folder",
            font=("Segoe UI", 8),
            fg=ACCENT_LIGHT,
            bg="#3a2e26",
            activebackground="#4d3e33",
            activeforeground=ACCENT_LIGHT,
            relief="flat",
            padx=8,
            pady=2,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
            cursor="hand2",
            command=lambda: self.open_folder(DATASET_PATH)
        )
        btn_open.pack(side=tk.RIGHT, padx=(4, 0))

        btn_refresh = tk.Button(
            top_bar,
            text="🔄 Refresh",
            font=("Segoe UI", 8),
            fg=ACCENT_LIGHT,
            bg="#3a2e26",
            activebackground="#4d3e33",
            activeforeground=ACCENT_LIGHT,
            relief="flat",
            padx=8,
            pady=2,
            highlightbackground=BORDER_COLOR,
            highlightthickness=1,
            cursor="hand2",
            command=self._load_students_table
        )
        btn_refresh.pack(side=tk.RIGHT, padx=4)

        # Table & Scrollbar
        cols = ("id", "name", "samples", "folder")
        self.tree_students = ttk.Treeview(parent, columns=cols, show="headings", style="Custom.Treeview")

        self.tree_students.heading("id", text="Student ID")
        self.tree_students.heading("name", text="Student Name")
        self.tree_students.heading("samples", text="Captured Images")
        self.tree_students.heading("folder", text="Dataset Folder")

        self.tree_students.column("id", width=100, anchor="center")
        self.tree_students.column("name", width=220, anchor="w")
        self.tree_students.column("samples", width=130, anchor="center")
        self.tree_students.column("folder", width=240, anchor="w")

        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.tree_students.yview)
        self.tree_students.configure(yscrollcommand=scrollbar.set)

        self.tree_students.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0, 10))

    # ========================================================
    # LOGGING & REFRESH LOGIC
    # ========================================================

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{timestamp}] {message}\n")
        self.txt_log.see(tk.END)

    def clear_log(self):
        self.txt_log.delete("1.0", tk.END)

    def _refresh_stats(self):
        # 1. Model Status
        if os.path.exists(MODEL_FILE):
            size_mb = os.path.getsize(MODEL_FILE) / (1024 * 1024)
            self.chip_model.val_lbl.config(text=f"Ready ({size_mb:.1f} MB)", fg=ACCENT_GOLD)
        else:
            self.chip_model.val_lbl.config(text="Not Trained", fg="#d96b52")

        # 2. Total Students
        count = 0
        if os.path.exists(STUDENT_FILE):
            with open(STUDENT_FILE, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)
                count = max(0, len(rows) - 1)
        self.chip_students.val_lbl.config(text=str(count))

        # 3. Attendance Today
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_count = 0
        if os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("Date") == today_str:
                        today_count += 1
        self.chip_today.val_lbl.config(text=str(today_count))

    def _load_attendance_table(self):
        for item in self.tree_attendance.get_children():
            self.tree_attendance.delete(item)

        if not os.path.exists(ATTENDANCE_FILE):
            return

        with open(ATTENDANCE_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.tree_attendance.insert(
                    "",
                    tk.END,
                    values=(
                        row.get("Student ID", ""),
                        row.get("Student Name", ""),
                        row.get("Date", ""),
                        row.get("First Seen", ""),
                        row.get("Last Seen", ""),
                        row.get("Status", "")
                    )
                )
        self._refresh_stats()

    def _load_students_table(self):
        for item in self.tree_students.get_children():
            self.tree_students.delete(item)

        if not os.path.exists(STUDENT_FILE):
            return

        with open(STUDENT_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                s_id = row.get("Student ID", "")
                s_name = row.get("Student Name", "")

                folder = os.path.join(DATASET_PATH, s_id)
                num_images = 0
                if os.path.exists(folder):
                    num_images = len([fn for fn in os.listdir(folder) if fn.lower().endswith((".jpg", ".png"))])

                self.tree_students.insert(
                    "",
                    tk.END,
                    values=(
                        s_id,
                        s_name,
                        f"{num_images} images",
                        f"registered_faces/{s_id}"
                    )
                )
        self._refresh_stats()

    def open_file(self, path):
        if os.path.exists(path):
            os.startfile(path)
        else:
            messagebox.showinfo("File Not Found", f"{os.path.basename(path)} has not been created yet.")

    def open_folder(self, path):
        os.makedirs(path, exist_ok=True)
        os.startfile(path)

    # ========================================================
    # ACTIONS EXECUTION
    # ========================================================

    def run_in_background(self, target, *args):
        thread = threading.Thread(target=target, args=args, daemon=True)
        thread.start()

    # 1. LIVE ATTENDANCE
    def run_attendance(self):
        if not os.path.exists(MODEL_FILE):
            messagebox.showwarning(
                "Model Not Trained",
                "face_model.yml not found!\nPlease train the model first by clicking 'Train LBPH Face Model'."
            )
            return

        self.log("Starting Live Automated Attendance...")
        self.run_in_background(self._exec_script, "attendance.py", "Attendance System finished.")

    # 2. AI WEBCAM (LIGHT BROWN WIREFRAME)
    def run_ai_webcam(self):
        self.log("Starting AI Face Mesh Webcam (Light Brown Wireframe)...")
        self.run_in_background(self._exec_script, "webcam.py", "AI Face Mesh Webcam closed.")

    # 3. TEST RECOGNITION
    def run_recognize_test(self):
        if not os.path.exists(MODEL_FILE):
            messagebox.showwarning("Model Not Found", "Please train the face model before testing.")
            return

        self.log("Starting Face Recognition Test...")
        self.run_in_background(self._exec_script, "recognize_face.py", "Recognition test closed.")

    # 4. YOLO TOOLS
    def run_yolo_test(self):
        self.log("Running YOLOv8 validation (test.py)...")
        self.run_in_background(self._exec_script_capture, "test.py")

    def run_yolo_webcam(self):
        self.log("Starting Simple YOLO Webcam (webcam copy.py)...")
        self.run_in_background(self._exec_script, "webcam copy.py", "Simple YOLO webcam closed.")

    # 5. REGISTER STUDENT (Direct GUI capture)
    def run_register_student(self):
        s_id = self.entry_id.get().strip()
        s_name = self.entry_name.get().strip()

        if not s_id or not s_name:
            messagebox.showwarning("Required Fields", "Please enter both Student ID and Student Name.")
            return

        self.log(f"Registering student: {s_name} (ID: {s_id}). Opening camera...")
        self.run_in_background(self._capture_registration, s_id, s_name)

    def _capture_registration(self, student_id, student_name):
        student_folder = os.path.join(DATASET_PATH, student_id)
        os.makedirs(student_folder, exist_ok=True)

        # Save to students.csv
        file_exists = os.path.exists(STUDENT_FILE)
        with open(STUDENT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Student ID", "Student Name"])
            writer.writerow([student_id, student_name])

        face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Camera Error", "Could not open webcam."))
            return

        IMAGE_COUNT = 20
        count = 0

        while True:
            ret, frame = camera.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, f"Student: {student_name} ({student_id})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Captured: {count}/{IMAGE_COUNT}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Press Q to cancel", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Student Registration - FaceAttend AI", frame)

            if len(faces) == 1:
                x, y, w, h = faces[0]
                face_image = gray[y:y + h, x:x + w]
                filename = os.path.join(student_folder, f"{count + 1}.jpg")
                cv2.imwrite(filename, face_image)
                count += 1
                self.root.after(0, lambda c=count: self.log(f"Captured {c}/{IMAGE_COUNT} images for {student_name}"))
                cv2.waitKey(150)

            if count >= IMAGE_COUNT:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                self.root.after(0, lambda: self.log("Registration cancelled by user."))
                break

        camera.release()
        cv2.destroyAllWindows()

        if count >= IMAGE_COUNT:
            self.root.after(0, lambda: self._on_registration_complete(student_id, student_name, count))

    def _on_registration_complete(self, s_id, s_name, count):
        self.entry_id.delete(0, tk.END)
        self.entry_name.delete(0, tk.END)
        self._load_students_table()
        self.log(f"✅ Registration completed: {s_name} ({count} images captured).")
        messagebox.showinfo(
            "Registration Complete",
            f"Successfully registered:\nStudent ID: {s_id}\nName: {s_name}\n\nRemember to click 'Train LBPH Face Model' to update recognition!"
        )

    # 6. TRAIN MODEL
    def run_train_model(self):
        self.log("Training LBPH Face Recognition Model in background...")
        self.run_in_background(self._train_model_worker)

    def _train_model_worker(self):
        try:
            if not os.path.exists(STUDENT_FILE):
                self.root.after(0, lambda: messagebox.showerror("Error", "students.csv not found!"))
                return

            students = {}
            with open(STUDENT_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    students[row["Student ID"]] = row["Student Name"]

            faces, labels = [], []
            label_map = {}
            current_label = 0

            for student_id in os.listdir(DATASET_PATH):
                folder = os.path.join(DATASET_PATH, student_id)
                if not os.path.isdir(folder) or student_id not in students:
                    continue

                label_map[current_label] = (student_id, students[student_id])

                for fn in os.listdir(folder):
                    if fn.lower().endswith((".jpg", ".png")):
                        img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (200, 200))
                            faces.append(img)
                            labels.append(current_label)

                current_label += 1

            if not faces:
                self.root.after(0, lambda: messagebox.showerror("Training Error", "No face images found in registered_faces/!"))
                return

            recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
            recognizer.train(faces, np.array(labels, dtype=np.int32))
            recognizer.write(MODEL_FILE)

            with open(LABEL_FILE, "w", encoding="utf-8") as f:
                f.write("Label,Student ID,Student Name\n")
                for lbl, (s_id, s_name) in label_map.items():
                    f.write(f"{lbl},{s_id},{s_name}\n")

            self.root.after(0, lambda: self._on_training_complete(len(label_map), len(faces)))

        except Exception as e:
            self.root.after(0, lambda err=str(e): self.log(f"Training error: {err}"))
            self.root.after(0, lambda err=str(e): messagebox.showerror("Training Failed", err))

    def _on_training_complete(self, num_students, num_faces):
        self._refresh_stats()
        self.log(f"✅ Model training successful! {num_students} students, {num_faces} face samples.")
        messagebox.showinfo(
            "Training Successful",
            f"LBPH Face Recognition Model trained!\n\nStudents Trained: {num_students}\nImages Trained: {num_faces}\nModel File: face_model.yml"
        )

    # Helper to execute external scripts
    def _exec_script(self, script_name, end_message):
        script_path = os.path.join(BASE_DIR, script_name)
        subprocess.run([PYTHON_EXE, script_path], cwd=BASE_DIR)
        self.root.after(0, lambda: self.log(end_message))
        self.root.after(0, self._load_attendance_table)

    def _exec_script_capture(self, script_name):
        script_path = os.path.join(BASE_DIR, script_name)
        proc = subprocess.Popen([PYTHON_EXE, script_path], cwd=BASE_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            cleaned = line.strip()
            if cleaned:
                self.root.after(0, lambda msg=cleaned: self.log(msg))
        proc.wait()
        self.root.after(0, lambda: self.log(f"Finished {script_name}."))


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def main():
    root = tk.Tk()
    app = FaceAttendGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

