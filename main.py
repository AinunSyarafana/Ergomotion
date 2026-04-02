import cv2
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import threading
import mediapipe as mp
import numpy as np
import sys
import os
from utils import DLT_multi, get_projection_matrix, write_keypoints_to_disk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import math
import json
import zipfile
import shutil
import socket

# --- Configuration & Globals ---
# Use a dark style for matplotlib to match the UI theme
plt.style.use('dark_background')

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

frame_shape = [1280, 720]

pose_keypoints = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Head
    11, 12, 13, 14, 15, 16,  # Body/Arms
    17, 18, 19, 20, 21, 22,  # Fingers/Hands
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32  # Legs
]

MP_POSE_NAMES = [
    "Nose", "L_Eye_Inner", "L_Eye", "L_Eye_Outer", "R_Eye_Inner", "R_Eye", "R_Eye_Outer",
    "L_Ear", "R_Ear", "Mouth_L", "Mouth_R",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Pinky", "R_Pinky", "L_Index", "R_Index", "L_Thumb", "R_Thumb",
    "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    "L_Heel", "R_Heel", "L_Foot_Index", "R_Foot_Index"
]

POSE_KEYPOINT_NAMES = [MP_POSE_NAMES[i] for i in pose_keypoints]

# --- UI Helper Classes ---

class OptionDialog(ttk.Toplevel):
    def __init__(self, master, options, prompt="Select an option:", title="Select Option"):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)

        # Center the dialog
        self.place_window_center()

        self.selected_option = tk.StringVar()

        container = ttk.Frame(self, padding=20)
        container.pack(fill=BOTH, expand=True)

        ttk.Label(container, text=prompt, font=("Segoe UI", 11)).pack(pady=(0, 20))

        btn_frame = ttk.Frame(container)
        btn_frame.pack(fill=X, expand=True)

        for option in options:
            ttk.Button(
                btn_frame,
                text=option,
                command=lambda o=option: self.on_option_selected(o),
                bootstyle="primary-outline",
                padding=(10, 5)
            ).pack(side=LEFT, padx=5, expand=True)

            # Force the window to calculate its size based on its contents
            self.update_idletasks()

            # Center the window on the screen after it knows its own size
            self.place_window_center()

    def on_option_selected(self, option):
        self.selected_option.set(option)
        self.destroy()


# --- Main Application ---

class Application:
    def __init__(self, master):
        self.master = master
        self.master.title("Ergomotion (v2.0 UI)")

        # Window State
        w, h = self.master.winfo_screenwidth(), self.master.winfo_screenheight()
        self.master.geometry(f"{w}x{h}")
        # self.master.state('zoomed')

        # State Variables
        self.show_joint_labels = tk.BooleanVar(value=True)
        self.show_3d_joint_labels = tk.BooleanVar(value=False)
        self.show_spine_labels = tk.BooleanVar(value=False)
        self.show_spine_dots = tk.BooleanVar(value=False)
        self.view_rotations = [0, 0, 0]
        self.cam_ids = []
        self.frame_paths = []
        self.fullres_windows = {}
        self.fullres_images = {}

        # 3D View Settings (Starts at Front View)
        self.view_elev = 10
        self.ELEVATION_STEP = 10
        self.side_view_angles = [-90, 180, 90, 0]
        self.current_side_view_index = 0
        self.view_azim = -90

        # Processing State
        self.lock = threading.Lock()
        self.cap0 = self.cap1 = self.cap2 = None
        self.previous_kpts = None
        self.is_processing = False
        self.is_replaying = False
        self.fps = 30
        self.max_frames = 0

        # Recent Files
        self.MAX_RECENT_FILES = 5
        self.RECENT_FILES_PATH = "recent_files.json"
        self.recent_files = []

        # Logic Setup
        self.initialize_reba_tables()
        self.joint_names = {i: mp_pose.PoseLandmark(i).name for i in pose_keypoints}
        self.landmark_map = {val: i for i, val in enumerate(pose_keypoints)}

        # Networking
        self.client_socket = None
        self.network_lock = threading.Lock()
        threading.Thread(target=self.setup_networking, daemon=True).start()

        # UI Construction
        self.setup_menus()
        self.setup_widgets()

        # Load recent files AFTER UI is built
        self.load_recent_files()

        # Spine mode
        self.spine_mode = 6 # Default

    # ---------------- UI Construction ----------------

    def setup_menus(self):
        self.menubar = tk.Menu(self.master)
        self.master.config(menu=self.menubar)

        # File Menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Import Session Package (.zip)", command=self.import_session_package)
        self.file_menu.add_command(label="Load Session (.json)", command=self.load_session)
        self.file_menu.add_command(label="Save Session (.json)", command=self.save_session, state='disabled')
        self.file_menu.add_separator()

        self.recent_files_menu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Recent Videos", menu=self.recent_files_menu)
        self.update_recent_files_menu()
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Reload Application", command=self.reload_app)
        self.file_menu.add_command(label="Exit", command=self.exit_app)

        # Action Menu
        self.action_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Actions", menu=self.action_menu)
        self.action_menu.add_command(label="Upload Video", command=self.choose_file)
        self.action_menu.add_command(label="Connect USB Camera", command=self.connect_usb_camera)
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Start Detection", command=self.detect)
        self.action_menu.add_command(label="Stop Detection", command=self.stop_detection, state='disabled')
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Replay", command=self.replay, state='disabled')

        # View Menu
        self.view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=self.view_menu)
        self.view_menu.add_checkbutton(label="Show Joint Labels (2D)", variable=self.show_joint_labels,
                                       command=self.refresh_joint_labels)
        self.view_menu.add_checkbutton(label="Show Joint Names (3D)", variable=self.show_3d_joint_labels,
                                       command=self.refresh_3d_view)
        self.view_menu.add_checkbutton(label="Show Spine Names", variable=self.show_spine_labels,
                                       command=self.refresh_3d_view)
        self.view_menu.add_checkbutton(label="Show Spine Keypoints", variable=self.show_spine_dots,
                                       command=self.refresh_3d_view)

        # Export Menu
        self.export_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Export", menu=self.export_menu)
        self.export_menu.add_command(label="Export Session Package (.zip)", command=self.export_session_package,
                                     state='disabled')
        self.export_menu.add_separator()
        self.export_menu.add_command(label="Export 2D Video View", command=self.save_2d_video, state='disabled')
        self.export_menu.add_command(label="Export 3D Skeleton Video", command=self.save, state='disabled')
        self.export_menu.add_command(label="Export CSV Data", command=self.save_csv, state='disabled')

    def setup_widgets(self):
        # 1. Top Dashboard - Pack at the top first
        dash_frame = ttk.Frame(self.master, padding=10)
        dash_frame.pack(side=TOP, fill=X)

        self.reba_label = ttk.Label(dash_frame, text="REBA Score: -", font=("Segoe UI", 18, "bold"), bootstyle="danger")
        self.reba_label.pack(side=LEFT, padx=10)

        input_container = ttk.Frame(dash_frame)
        input_container.pack(side=RIGHT)

        ttk.Label(input_container, text="Force/Load:", font=("Segoe UI", 10)).pack(side=LEFT, padx=5)
        self.force_load_var = tk.StringVar()
        options = ['0 (<11 lbs)', '1 (11-22 lbs)', '2 (>22 lbs)', '3 (+Shock/Rapid)']
        self.force_load_combobox = ttk.Combobox(input_container, textvariable=self.force_load_var, values=options,
                                                state='readonly', width=15, bootstyle="primary")
        self.force_load_combobox.set(options[0])
        self.force_load_combobox.pack(side=LEFT)

        # 2. Bottom Controls - Pack these at the BOTTOM before the middle section
        # This "claims" the bottom space so it cannot be pushed away
        self.status_bar = ttk.Label(self.master, text="Ready", bootstyle="inverse-secondary", anchor=W, padding=3)
        self.status_bar.pack(side=BOTTOM, fill=X)

        self.slider_frame = ttk.Frame(self.master, padding=10)
        self.slider_frame.pack(side=BOTTOM, fill=X)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.slider_frame, orient=HORIZONTAL, mode='determinate',
                                            variable=self.progress_var, bootstyle="success-striped")
        self.progress_bar.pack(fill=X, pady=(0, 5))

        self.frame_slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient=HORIZONTAL,
                                      command=self.update_frame_display, state='disabled', bootstyle="success")
        self.frame_slider.pack(fill=X, pady=5)

        controls = ttk.Frame(self.slider_frame)
        controls.pack(pady=5)

        self.skip_backward_button = ttk.Button(controls, text="<< -10s", command=self.skip_backward, state='disabled',
                                               bootstyle="primary-outline")
        self.play_button = ttk.Button(controls, text="▶ Play", command=self.play_replay, state='disabled',
                                      bootstyle="success")
        self.pause_button = ttk.Button(controls, text="❚❚ Pause", command=self.pause_replay, state='disabled',
                                       bootstyle="warning")
        self.skip_forward_button = ttk.Button(controls, text=">> +10s", command=self.skip_forward, state='disabled',
                                              bootstyle="primary-outline")

        self.skip_backward_button.pack(side=LEFT, padx=5)
        self.play_button.pack(side=LEFT, padx=5)
        self.pause_button.pack(side=LEFT, padx=5)
        self.skip_forward_button.pack(side=LEFT, padx=5)

        # 3. Main Content - Pack this LAST with expand=True
        # It will now take up all the REMAINING space between the top and bottom
        self.main_pane = ttk.Panedwindow(self.master, orient=HORIZONTAL)
        self.main_pane.pack(fill=BOTH, expand=True, padx=10, pady=5)

        # --- LEFT: Data Table & Cameras ---
        left_panel = ttk.Frame(self.main_pane)
        self.main_pane.add(left_panel, weight=1)

        table_frame = ttk.Labelframe(left_panel, text="Biomechanical Data", padding=5)
        table_frame.pack(fill=BOTH, expand=True)

        cols = ('part', 'angle', 'score', 'velocity')
        table_container = ttk.Frame(table_frame)
        table_container.pack(fill=BOTH, expand=True)

        self.angle_table = ttk.Treeview(table_container, columns=cols, show='headings', bootstyle="info")
        vsb = ttk.Scrollbar(table_container, orient="vertical", command=self.angle_table.yview)
        self.angle_table.configure(yscrollcommand=vsb.set)

        self.angle_table.pack(side=LEFT, fill=BOTH, expand=True)
        vsb.pack(side=RIGHT, fill=Y)

        self.angle_table.heading('part', text='Body Part')
        self.angle_table.heading('angle', text='Angle (°)')
        self.angle_table.heading('score', text='Score')
        self.angle_table.heading('velocity', text='Vel (mm/s)')

        self.angle_table.column('part', width=140)
        self.angle_table.column('angle', width=80, anchor='center')
        self.angle_table.column('score', width=60, anchor='center')
        self.angle_table.column('velocity', width=100, anchor='center')

        self.angle_table_rows = {}
        parts = ["Neck", "Trunk", "L Knee", "R Knee", "L Upper Arm", "R Upper Arm", "L Lower Arm", "R Lower Arm",
                 "L Wrist", "R Wrist"]
        for part in parts:
            self.angle_table_rows[part] = self.angle_table.insert('', 'end', values=(part, '-', '-', '-'))

        # --- RIGHT: Visualization ---
        right_panel = ttk.Frame(self.main_pane)
        self.main_pane.add(right_panel, weight=3)

        view3d_container = ttk.Labelframe(right_panel, text="3D Skeleton Reconstruction", padding=2)
        view3d_container.pack(side=TOP, fill=BOTH, expand=True)

        self.fig_3d = plt.figure(figsize=(4, 4))
        self.fig_3d.patch.set_facecolor('#2B3E50')
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.ax_3d.set_facecolor('#2B3E50')

        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=view3d_container)
        self.canvas_3d.get_tk_widget().pack(fill=BOTH, expand=True)

        ctrl_frame = ttk.Frame(view3d_container)
        ctrl_frame.place(relx=0.98, rely=0.02, anchor='ne')

        btn_style = "secondary-outline"
        ttk.Button(ctrl_frame, text="↑", command=lambda: self.rotate_3d_elevation(1), width=3,
                   bootstyle=btn_style).grid(row=0, column=1)
        ttk.Button(ctrl_frame, text="↶", command=lambda: self.cycle_3d_azimuth(-1), width=3, bootstyle=btn_style).grid(
            row=1, column=0)
        ttk.Button(ctrl_frame, text="⌂", command=self.reset_3d_view, width=3, bootstyle="warning-outline").grid(row=1,
                                                                                                                column=1)
        ttk.Button(ctrl_frame, text="↷", command=lambda: self.cycle_3d_azimuth(1), width=3, bootstyle=btn_style).grid(
            row=1, column=2)
        ttk.Button(ctrl_frame, text="↓", command=lambda: self.rotate_3d_elevation(-1), width=3,
                   bootstyle=btn_style).grid(row=2, column=1)

        # Camera Feeds Frame
        # 1. Create a Labelframe to act as the outer container
        cams_outer_frame = ttk.Labelframe(left_panel, text="Camera Feeds", padding=5)
        cams_outer_frame.pack(side=BOTTOM, fill=BOTH, expand=True, pady=(5, 0))

        # 2. Create a Canvas and Scrollbar
        canvasCam = tk.Canvas(cams_outer_frame, borderwidth=10, highlightthickness=0, background='#2B3E50')
        v_scrollbar = ttk.Scrollbar(cams_outer_frame, orient=VERTICAL, command=canvasCam.yview)

        # 3. This is the frame that actually holds the cameras
        self.scrollable_cams_inner = ttk.Frame(canvasCam)

        # 4. Configure canvas to scroll the inner frame
        self.scrollable_cams_inner.bind(
            "<Configure>",
            lambda e: canvasCam.configure(scrollregion=canvasCam.bbox("all"))
        )

        # Create a window inside the canvas to display the inner frame
        canvas_frame = canvasCam.create_window((0, 0), window=self.scrollable_cams_inner, anchor="nw")

        # Make sure the inner frame matches the canvas width
        def configure_inner_frame(event):
            canvasCam.itemconfig(canvas_frame, width=event.width)

        canvasCam.bind("<Configure>", configure_inner_frame)

        canvasCam.configure(yscrollcommand=v_scrollbar.set)

        # Pack the canvas and scrollbar
        canvasCam.pack(side=LEFT, fill=BOTH, expand=True, padx=10)
        v_scrollbar.pack(side=RIGHT, fill=Y)

        # Mouse wheel support for scrolling
        def _on_mousewheel(event):
            canvasCam.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvasCam.bind_all("<MouseWheel>", _on_mousewheel)

        self.labels_2d = []
        self.black_image_tk = ImageTk.PhotoImage(image=Image.new('RGB', (200, 150), (20, 20, 20)))

        for i in range(3):
            # We use a Labelframe for each camera to keep the look consistent
            container = ttk.Labelframe(self.scrollable_cams_inner, text=f"Cam {i}", padding=2)
            container.pack(side=TOP, fill=BOTH, padx=10, pady=5)

            # Header inside the labelframe for rotation
            header = ttk.Frame(container)
            header.pack(fill=X)
            ttk.Button(header, text="↻", command=lambda c=i: self.rotate_view_button(c), width=2,
                       bootstyle="link").pack(side=RIGHT)

            lbl = ttk.Label(container, image=self.black_image_tk, anchor="center")
            lbl.image = self.black_image_tk
            lbl.pack(fill=BOTH, expand=True, pady=10)
            lbl.bind("<Double-Button-1>", lambda e, c=i: self.open_fullres_view(c))
            self.labels_2d.append(lbl)

    # ---------------- Networking ----------------
    def setup_networking(self):
        host = '127.0.0.1'
        port = 65432
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen()

        while True:
            try:
                client, addr = server_socket.accept()
                with self.network_lock:
                    self.client_socket = client

                while True:
                    data = client.recv(1, socket.MSG_PEEK)
                    if not data: break
            except:
                pass
            with self.network_lock:
                if self.client_socket:
                    try:
                        self.client_socket.close()
                    except:
                        pass
                self.client_socket = None

    def send_data_to_client(self, kpts_3d_frame):
        with self.network_lock:
            if self.client_socket:
                try:
                    data = kpts_3d_frame.tolist()
                    message = json.dumps(data) + "\n"
                    self.client_socket.sendall(message.encode('utf-8'))
                except:
                    pass

    # ---------------- Application Logic ----------------

    def exit_app(self):
        self.is_processing = False
        self.is_replaying = False
        self.master.quit()

    def reload_app(self):
        os.execl(sys.executable, sys.executable, *sys.argv)

    def _reset_ui_for_new_video(self):
        self.action_menu.entryconfig("Replay", state="disabled")
        self.export_menu.entryconfig("Export Session Package (.zip)", state="disabled")
        self.export_menu.entryconfig("Export 2D Video View", state="disabled")
        self.export_menu.entryconfig("Export 3D Skeleton Video", state="disabled")
        self.export_menu.entryconfig("Export CSV Data", state="disabled")
        self.file_menu.entryconfig("Save Session (.json)", state="disabled")

    def _open_video_file(self, filepath):
        if not filepath or not os.path.exists(filepath): return

        prompt = f"Assign '{os.path.basename(filepath)}' to which view?"
        dialog = OptionDialog(self.master, ["Cam0", "Cam1", "Cam2"], prompt=prompt)
        self.master.wait_window(dialog)
        input_source = dialog.selected_option.get()
        if not input_source: return

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            tk.messagebox.showerror("Error", f"Failed to open video file: {filepath}")
            return

        cam_id = int(input_source.replace("Cam", ""))
        target_label = self.labels_2d[cam_id]

        if cam_id == 0:
            self.cap0 = cap
        elif cam_id == 1:
            self.cap1 = cap
        elif cam_id == 2:
            self.cap2 = cap

        if target_label:
            ret, frame = cap.read()
            if ret:
                frame_fit = self._fit_image_to_label(frame, target_label)
                img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_fit, cv2.COLOR_BGR2RGB)))
                target_label.config(image=img_tk)
                target_label.image = img_tk
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.status_bar.config(text=f"{input_source}: {os.path.basename(filepath)} loaded.")
        self.add_to_recent_files(filepath)

    def _fit_image_to_label(self, cv_img, label_widget):
        parent = label_widget.master
        target_w = parent.winfo_width()
        target_h = parent.winfo_height()

        # Safety for initial load
        if target_w < 50: target_w = 200
        if target_h < 50: target_h = 150
        target_h -= 25  # Header space

        h0, w0 = cv_img.shape[:2]
        if w0 == 0 or h0 == 0: return cv_img

        scale = min(target_w / w0, target_h / h0)
        new_w = max(1, int(w0 * scale))
        new_h = max(1, int(h0 * scale))
        return cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def choose_file(self):
        self._reset_ui_for_new_video()
        file_path = tk.filedialog.askopenfilename(filetypes=[("Video files", "*.avi;*.mp4;*.mov")])
        if file_path:
            self._open_video_file(file_path)

    def connect_usb_camera(self):
        self._reset_ui_for_new_video()
        dialog = OptionDialog(self.master, ["Cam0", "Cam1", "Cam2"], prompt="Select View for USB Camera:")
        self.master.wait_window(dialog)
        input_source = dialog.selected_option.get()
        if not input_source: return

        cam_index = tk.simpledialog.askinteger("Camera Index", "Enter camera index (e.g., 0, 1):", parent=self.master)
        if cam_index is None: return

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            tk.messagebox.showerror("Error", "Could not open camera.")
            return

        cam_id = int(input_source.replace("Cam", ""))
        if cam_id == 0:
            self.cap0 = cap
        elif cam_id == 1:
            self.cap1 = cap
        elif cam_id == 2:
            self.cap2 = cap
        self.status_bar.config(text=f"{input_source} connected to USB {cam_index}.")

    def stop_detection(self):
        self.is_processing = False

    def cleanup_images(self):
        if os.path.exists('./images'):
            for f in os.listdir('./images'):
                try:
                    os.unlink(os.path.join('./images', f))
                except:
                    pass
        else:
            os.makedirs('./images')

    def detect(self):
        if self.is_processing: return
        num_caps = len([c for c in [self.cap0, self.cap1, self.cap2] if c is not None])
        if num_caps < 1:
            tk.messagebox.showwarning('No Input', 'Please upload at least 1 video.')
            return

        # 2. Selection for Spine Keypoints
        dialog = OptionDialog(self.master, ["6 keypoints", "7 keypoints"],
                                  prompt="Select the number of spine keypoints to use")
        self.master.wait_window(dialog)
        selection = dialog.selected_option.get()
        if not selection: return  # User closed dialog

        self.spine_mode = 6 if "6" in selection else 7

        self.cleanup_images()
        self.is_processing = True

            # ... (rest of method same as before)
        self.cleanup_images()
        self.is_processing = True

        self.action_menu.entryconfig("Upload Video", state="disabled")
        self.action_menu.entryconfig("Stop Detection", state="normal")
        self.progress_bar.configure(mode='determinate')
        self.status_bar.config(text="Processing...")
        self.progress_var.set(0)

        threading.Thread(target=self.play_video, daemon=True).start()

    def replay(self):
        if hasattr(self, 'all_kpts_3d') and len(self.all_kpts_3d) > 0:
            self.frame_slider.configure(state='normal', to=len(self.all_kpts_3d) - 1)
            self.play_button.config(state='normal')
            self.pause_button.config(state='disabled')
            self.skip_backward_button.config(state='normal')
            self.skip_forward_button.config(state='normal')
            self.frame_slider.set(0)
            self.update_frame_display(0)

    # ---------------- Core Processing ----------------

    def extract_world_keypoints(self, result):
        if result.pose_world_landmarks:
            w = result.pose_world_landmarks.landmark
            return np.array([[w[p].x, w[p].y, w[p].z] for p in pose_keypoints], dtype=np.float32)
        return np.full((len(pose_keypoints), 3), -1.0, dtype=np.float32)

    def extract_keypoints(self, result, shape):
        if result.pose_landmarks:
            all_landmarks = result.pose_landmarks.landmark
            return [[int(round(all_landmarks[p_idx].x * shape[1])),
                     int(round(all_landmarks[p_idx].y * shape[0]))] for p_idx in pose_keypoints]
        return [[-1, -1]] * len(pose_keypoints)

    def triangulate_points(self, P_list, kpts_2d):
        return np.array(
            [DLT_multi(P_list, [cam_kpts[i] for cam_kpts in kpts_2d]) for i in range(len(pose_keypoints))]
        ).reshape((len(pose_keypoints), 3))

    def play_video(self):
        cam_info = []
        if self.cap0: cam_info.append((0, self.cap0, self.labels_2d[0]))
        if self.cap1: cam_info.append((1, self.cap1, self.labels_2d[1]))
        if self.cap2: cam_info.append((2, self.cap2, self.labels_2d[2]))

        self.cam_ids = [info[0] for info in cam_info]
        caps = [info[1] for info in cam_info]
        self.fps = caps[0].get(cv2.CAP_PROP_FPS) or 30

        use_single_cam_world3d = False
        if len(cam_info) == 1:
            dialog = OptionDialog(self.master, ["2D only (Z=0)", "MediaPipe World 3D"], prompt="Single Video Mode:")
            self.master.wait_window(dialog)
            if dialog.selected_option.get() == "MediaPipe World 3D":
                use_single_cam_world3d = True

        P_list = [get_projection_matrix(cid) for cid, _, _ in cam_info] if len(cam_info) >= 2 else None

        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        is_live = any(fc <= 0 for fc in frame_counts)
        self.max_frames = min(frame_counts) if not is_live else 0

        if is_live: self.progress_bar.configure(mode='indeterminate'); self.progress_bar.start(10)

        poses = {cid: mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) for cid, _, _ in cam_info}
        self.all_kpts_3d = []
        self.frame_2d_paths = [[] for _ in range(3)]
        skeleton_conn = self.get_skeleton_definition()
        last_good_kpts = None
        index = 0

        while self.is_processing:
            rets, frames = zip(*[cap.read() for cap in caps])
            if not all(rets): break

            current_frame_2d_kpts = []
            results = {}

            for i, (cam_id, _, label) in enumerate(cam_info):
                result = poses[cam_id].process(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
                results[cam_id] = result
                current_frame_2d_kpts.append(self.extract_keypoints(result, frames[i].shape))
                self.display_2d_frame(frames[i], result, label, index, cam_id)

            if len(cam_info) >= 2:
                frame_p3ds = self.triangulate_points(P_list, current_frame_2d_kpts)
            else:
                cid = cam_info[0][0]
                if use_single_cam_world3d:
                    frame_p3ds = self.extract_world_keypoints(results[cid])
                else:
                    k2d = np.array(current_frame_2d_kpts[0], dtype=np.float32)
                    frame_p3ds = np.concatenate([k2d, np.zeros((len(pose_keypoints), 1), dtype=np.float32)], axis=1)

            if last_good_kpts is not None:
                for j in range(len(frame_p3ds)):
                    if np.all(frame_p3ds[j] == -1): frame_p3ds[j] = last_good_kpts[j]

            if last_good_kpts is None:
                last_good_kpts = np.copy(frame_p3ds)
            else:
                good_indices = ~np.all(frame_p3ds == -1, axis=1)
                last_good_kpts[good_indices] = frame_p3ds[good_indices]

            valid_kpts = frame_p3ds[~np.all(frame_p3ds == -1, axis=1)]
            frame_p3ds_centered = frame_p3ds - np.mean(valid_kpts, axis=0) if len(valid_kpts) > 0 else frame_p3ds

            # --- CALCULATE MIDPOINTS & SPINE ---
            # Index 11=L_Shoulder, 12=R_Shoulder | 23=L_Hip, 24=R_Hip
            mid_shoulder = (frame_p3ds_centered[11] + frame_p3ds_centered[12]) / 2
            mid_hip = (frame_p3ds_centered[23] + frame_p3ds_centered[24]) / 2

            # Spine 2: Midpoint between shoulder and hip
            spine_2 = (mid_shoulder + mid_hip) / 2
            # Spine 1: Midpoint between shoulder and spine 2
            spine_1 = (mid_shoulder + spine_2) / 2
            # Spine 4: Midpoint between spine 2 and hip
            spine_4 = (spine_2 + mid_hip) / 2
            # Spine 3: Midpoint between spine 2 and spine 4
            spine_3 = (spine_2 + spine_4) / 2
            # Spine 5: Midpoint between spine 4 and hip
            spine_5 = (spine_4 + mid_hip) / 2

            # Index Map for Reference
            # Stack them in order: 33=Mid_Sh, 34=Mid_Hip, 35=Spine1, 36=Spine2, 37=Spine3, 38=Spine4, 39=Spine5
            if self.spine_mode == 7:
                spine_stack = [mid_shoulder, mid_hip, spine_1, spine_2, spine_3, spine_4, spine_5]
            else:
                spine_stack = [mid_shoulder, mid_hip, spine_1, spine_2, spine_3, spine_4]

            frame_p3ds_final = np.vstack([frame_p3ds_centered] + spine_stack)
            self.all_kpts_3d.append(frame_p3ds_final)

            # --- Print Keypoint Coordinates to PyCharm Console ---
            #print(f"\n--- Frame {index} ---")
            #for i, kpt in enumerate(frame_p3ds_final):
            #    name = POSE_KEYPOINT_NAMES[i]
                # Format: Name: [X, Y, Z]
            #    print(f"{name:<15}: x = {kpt[0]:.2f}, y = {kpt[1]:.2f}, z = {kpt[2]:.2f}")
            #    if (name=="L_Shoulder"):
            #        print(f"{name:<15}: x = {kpt[0]:.2f}, y = {kpt[1]:.2f}, z = {kpt[2]:.2f}")
            #    elif (name=="R_Shoulder"):
            #        print(f"{name:<15}: x = {kpt[0]:.2f}, y = {kpt[1]:.2f}, z = {kpt[2]:.2f}")
            #    elif (name=="L_Hip"):
            #        print(f"{name:<15}: x = {kpt[0]:.2f}, y = {kpt[1]:.2f}, z = {kpt[2]:.2f}")
            #    elif (name=="R_Hip"):
            #        print(f"{name:<15}: x = {kpt[0]:.2f}, y = {kpt[1]:.2f}, z = {kpt[2]:.2f}")

            # Update the print to include new midpoints
            #print(f"\n--- Frame {index} ---")
            #print(f"Mid-Shoulder (Idx 33): x = {mid_shoulder[0]:.2f}, y = {mid_shoulder[1]:.2f}, z = {mid_shoulder[2]:.2f}")
            #print(f"Mid-Hip (Idx 34):      x = {mid_hip[0]:.2f}, y = {mid_hip[1]:.2f}, z = {mid_hip[2]:.2f}")

            # --- Print Hip Coordinates in one line ---
            l_hip = frame_p3ds_final[23]
            r_hip = frame_p3ds_final[24]
            m_hip = frame_p3ds_final[34]

            print(f"Frame {index} | "
                  f"Left hip: x = {l_hip[0]:.2f}, y = {l_hip[1]:.2f}, z = {l_hip[2]:.2f}; "
                  f"Right hip: x = {r_hip[0]:.2f}, y = {r_hip[1]:.2f}, z = {r_hip[2]:.2f}; "
                  f"Mid hip: x = {m_hip[0]:.2f}, y = {m_hip[1]:.2f}, z = {m_hip[2]:.2f}")

            self.send_data_to_client(frame_p3ds_final)
            self.draw_3d_skeleton(frame_p3ds_final, skeleton_conn)

            self.fig_3d.savefig(f'./images/{str(index).zfill(8)}.png', bbox_inches='tight', pad_inches=0, dpi=100)
            self.canvas_3d.draw()

            if not is_live and self.max_frames > 0:
                self.progress_var.set((index / self.max_frames) * 100)
                self.status_bar.config(text=f"Processing Frame: {index + 1}/{self.max_frames}")
                if index >= self.max_frames - 1: self.is_processing = False

            index += 1

        if is_live: self.progress_bar.stop()
        self.finalize_processing()

    def finalize_processing(self):
        for cap in [self.cap0, self.cap1, self.cap2]:
            if cap: cap.release()
        self.cap0 = self.cap1 = self.cap2 = None

        self.action_menu.entryconfig("Upload Video", state="normal")
        self.action_menu.entryconfig("Stop Detection", state="disabled")
        self.status_bar.config(text="Processing Complete.")

        if hasattr(self, 'all_kpts_3d') and len(self.all_kpts_3d) > 0:
            self.max_frames = len(self.all_kpts_3d)
            write_keypoints_to_disk('kpts_3d.dat', np.array(self.all_kpts_3d))
            self.action_menu.entryconfig("Replay", state="normal")
            self.file_menu.entryconfig("Save Session (.json)", state="normal")
            self.export_menu.entryconfig("Export Session Package (.zip)", state="normal")
            self.export_menu.entryconfig("Export 2D Video View", state="normal")
            self.export_menu.entryconfig("Export 3D Skeleton Video", state="normal")
            self.export_menu.entryconfig("Export CSV Data", state="normal")

        self.is_processing = False

    # ---------------- Visualization ----------------

    def draw_3d_skeleton(self, kpts, skeleton_connections):
        self.ax_3d.clear()
        bg_color = '#2B3E50'
        self.ax_3d.set_facecolor(bg_color)
        self.ax_3d.grid(color='lightgrey', linestyle='--', linewidth=0.5, alpha=0.3)

        # Mapping coordinates
        xs = kpts[:, 0]
        ys = kpts[:, 2]
        zs = -kpts[:, 1]

        self.ax_3d.view_init(elev=self.view_elev, azim=self.view_azim)

        # 1. Draw Skeleton Lines
        for p1, p2 in skeleton_connections:
            if p1 < len(kpts) and p2 < len(kpts):
                if not np.all(kpts[p1] == -1) and not np.all(kpts[p2] == -1):
                    self.ax_3d.plot([xs[p1], xs[p2]], [ys[p1], ys[p2]], [zs[p1], zs[p2]], linewidth=2, c='#FF4444')

        # 2. Draw the Pelvis Circle (Aligned to Hip Orientation)
        if len(kpts) > 34:
            l_hip = kpts[23]
            r_hip = kpts[24]
            mid_hip = kpts[34]

            # Vector from Left Hip to Right Hip
            hip_vector = r_hip - l_hip
            radius = np.linalg.norm(hip_vector) / 2

            # Normalizing the hip vector to find direction
            hip_dir = hip_vector / np.linalg.norm(hip_vector)

            # Generate base circle points on a flat plane (XY)
            theta = np.linspace(0, 2 * np.pi, 50)
            # Base circle in 3D (lying on XY plane before rotation)
            circle_base = np.array([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])

            # Rotation Logic: Align base 'X' axis with our 'hip_dir'
            # We calculate the angle of the hip line in the XZ plane
            angle = np.arctan2(hip_dir[2], hip_dir[0])

            # Simple rotation matrix around the Vertical (Y) axis
            # This ensures the circle's "diameter" always stays locked to the hip joints
            rotation_matrix = np.array([
                [np.cos(angle),  0, np.sin(angle)],
                [0,              1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])

            # Apply rotation to the circle points
            rotated_circle = rotation_matrix @ circle_base

            # Map to Matplotlib axes (xs=X, ys=Z, zs=-Y) and translate to Mid_Hip
            c_xs = rotated_circle[0, :] + mid_hip[0]
            c_ys = rotated_circle[2, :] + mid_hip[2]
            c_zs = -rotated_circle[1, :] - mid_hip[1]

            # Plot the aligned circle # Create circle at hip
            #self.ax_3d.plot(c_xs, c_ys, c_zs, color='lime', linewidth=2, alpha=0.8)
            print("Testing1")

            # Add center dot for confirmation # Create circle dot at hip
            #self.ax_3d.scatter([mid_hip[0]], [mid_hip[2]], [-mid_hip[1]], color='lime', s=30)

        # 2. Draw Spine Specific Visualization
        if self.show_spine_dots.get() and len(kpts) > 33:
            # Draw black dots for indices 33 to 38
            self.ax_3d.scatter(xs[33:39], ys[33:39], zs[33:39], color='black', s=10)

        #
        if self.show_spine_labels.get() and len(kpts) > 33:
            spine_names = {
                    33: "MID_SHOULDER", 34: "MID_HIP",
                35: "SPINE_1", 36: "SPINE_2",
                37: "SPINE_3", 38: "SPINE_4"
            }
            if self.spine_mode == 7:
                spine_names[39] = "SPINE_5"

            for idx, name in spine_names.items():
                if idx < len(kpts) and not np.all(kpts[idx] == -1):
                    self.ax_3d.text(xs[idx], ys[idx], zs[idx], name, fontsize=7, color='cyan')

        # 4. Draw Standard Joint Labels (Only for indices 0-32)
        if self.show_3d_joint_labels.get():
            for idx in range(min(len(kpts), 33)):
                if not np.all(kpts[idx] == -1):
                    mp_idx = pose_keypoints[idx]
                    name = self.joint_names.get(mp_idx, str(idx))
                    self.ax_3d.text(xs[idx], ys[idx], zs[idx], name, fontsize=7, color='cyan')

        # Scaling logic
        valid_mask = ~np.all(kpts == -1, axis=1)
        if np.any(valid_mask):
            v_xs, v_ys, v_zs = xs[valid_mask], ys[valid_mask], zs[valid_mask]
            max_range = np.array(
                [v_xs.max() - v_xs.min(), v_ys.max() - v_ys.min(), v_zs.max() - v_zs.min()]).max() / 2.0
            mid_x, mid_y, mid_z = np.mean(v_xs), np.mean(v_ys), np.mean(v_zs)
            self.ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

    def display_2d_frame(self, frame, result, label, index, cam_id):
        frame_vis = frame.copy()
        if result and result.pose_landmarks:
            mp_drawing.draw_landmarks(frame_vis, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w = frame_vis.shape[:2]
            lm = result.pose_landmarks.landmark

            # --- DYNAMIC 2D SPINE CALCULATION ---
            l_sh = np.array([lm[11].x, lm[11].y])
            r_sh = np.array([lm[12].x, lm[12].y])
            l_hp = np.array([lm[23].x, lm[23].y])
            r_hp = np.array([lm[24].x, lm[24].y])

            m_sh = (l_sh + r_sh) / 2
            m_hp = (l_hp + r_hp) / 2
            s2 = (m_sh + m_hp) / 2
            s1 = (m_sh + s2) / 2
            s4 = (s2 + m_hp) / 2
            s3 = (s2 + s4) / 2
            s5 = (s4 + m_hp) / 2

            # Define ordered list for the continuous spine line:
            # MID_SH -> SPINE_1 -> SPINE_2 -> SPINE_3 -> SPINE_4 (-> SPINE_5) -> MID_HIP
            spine_line_pts = [m_sh, s1, s2, s3, s4]
            if self.spine_mode == 7:
                spine_line_pts.append(s5)
            spine_line_pts.append(m_hp)

            # --- DRAW SPINE LINES ---
            for i in range(len(spine_line_pts) - 1):
                pt1 = (int(spine_line_pts[i][0] * w), int(spine_line_pts[i][1] * h))
                pt2 = (int(spine_line_pts[i + 1][0] * w), int(spine_line_pts[i + 1][1] * h))
                # Draw a red line to match your 3D skeleton style
                cv2.line(frame_vis, pt1, pt2, (0, 0, 255), 2)

                # --- DRAW SPINE DOTS ---
            for pt in spine_line_pts:
                px, py = int(pt[0] * w), int(pt[1] * h)
                cv2.circle(frame_vis, (px, py), 4, (0, 255, 255), -1)

            # --- LABELS ---
            if self.show_joint_labels.get():
                # Standard MediaPipe Labels
                for idx_in_list, mp_idx in enumerate(pose_keypoints):
                    point = lm[mp_idx]
                    cx, cy = int(point.x * w), int(point.y * h)
                    cv2.putText(frame_vis, POSE_KEYPOINT_NAMES[idx_in_list], (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Spine Labels (if checkbox is on)
                if self.show_spine_labels.get():
                    spine_labels = ["MID_SH", "S1", "S2", "S3", "S4"]
                    if self.spine_mode == 7: spine_labels.append("S5")
                    spine_labels.append("MID_HIP")

                    for i, pt in enumerate(spine_line_pts):
                        px, py = int(pt[0] * w), int(pt[1] * h)
                        cv2.putText(frame_vis, spine_labels[i], (px + 8, py + 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

        # Save and display logic remains the same
        path = f'./images/cam{cam_id}_frame{str(index).zfill(8)}.png'
        cv2.imwrite(path, frame_vis)
        self.frame_2d_paths[cam_id].append(path)

        rot = self.view_rotations[cam_id]
        if rot == 1:
            frame_vis = cv2.rotate(frame_vis, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 2:
            frame_vis = cv2.rotate(frame_vis, cv2.ROTATE_180)
        elif rot == 3:
            frame_vis = cv2.rotate(frame_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)

        frame_fit = self._fit_image_to_label(frame_vis, label)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_fit, cv2.COLOR_BGR2RGB)))

        with self.lock:
            label.config(image=img_tk)
            label.image = img_tk
        self.update_fullres_view(cam_id, frame_vis)

    def refresh_joint_labels(self):
        # Use current_frame_idx instead of slider value to avoid state errors
        if hasattr(self, 'all_kpts_3d') and len(self.all_kpts_3d) > 0:
            current_val = self.frame_slider.get()
            self.update_frame_display(current_val)

    def refresh_3d_view(self):
        if self.is_replaying and hasattr(self, 'all_kpts_3d'):
            frame_idx = int(self.frame_slider.get())
            self.draw_3d_skeleton(self.all_kpts_3d[frame_idx], self.get_skeleton_definition())
            self.canvas_3d.draw()

    # ---------------- Playback ----------------
    def update_frame_display(self, value):
        frame_idx = int(float(value))
        if not hasattr(self, 'all_kpts_3d') or frame_idx >= len(self.all_kpts_3d): return

        self.status_bar.config(text=f"Replay Frame: {frame_idx + 1}/{self.max_frames}")
        kpts = self.all_kpts_3d[frame_idx]
        self.send_data_to_client(kpts)

        prev = self.all_kpts_3d[frame_idx - 1] if frame_idx > 0 else kpts
        velocities = self.calculate_velocities(kpts, prev, self.fps)
        final_score, risk, angles, scores = self.calculate_reba_score(kpts)

        self.reba_label.config(text=f"REBA Score: {final_score} ({risk})")
        self.update_tables(angles, scores, velocities)

        self.draw_3d_skeleton(kpts, self.get_skeleton_definition())
        self.canvas_3d.draw()

        for i, label in enumerate(self.labels_2d):
            if i in self.cam_ids and frame_idx < len(self.frame_2d_paths[i]):
                try:
                    img = cv2.imread(self.frame_2d_paths[i][frame_idx])
                    rot = self.view_rotations[i]
                    if rot == 1:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif rot == 2:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    elif rot == 3:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    frame_fit = self._fit_image_to_label(img, label)
                    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_fit, cv2.COLOR_BGR2RGB)))
                    label.config(image=img_tk)
                    label.image = img_tk
                except:
                    pass

    def update_tables(self, angles, scores, velocities):
        for part, val in angles.items():
            if part in self.angle_table_rows: self.angle_table.set(self.angle_table_rows[part], 'angle', f"{val:.1f}")
        for part, val in scores.items():
            if part in self.angle_table_rows: self.angle_table.set(self.angle_table_rows[part], 'score', val)
        for part, val in velocities.items():
            if part in self.angle_table_rows: self.angle_table.set(self.angle_table_rows[part], 'velocity',
                                                                   f"{val:.2f}")

    # ---------------- Replay Controls ----------------
    def play_replay(self):
        if not self.is_replaying:
            self.is_replaying = True
            self.play_button.config(state='disabled')
            self.pause_button.config(state='normal')
            self.update_replay_loop()

    def pause_replay(self):
        self.is_replaying = False
        self.play_button.config(state='normal')
        self.pause_button.config(state='disabled')

    def update_replay_loop(self):
        if self.is_replaying:
            curr = self.frame_slider.get()
            if curr < self.max_frames - 1:
                self.frame_slider.set(curr + 1)
                delay = int(1000 / self.fps) if self.fps > 0 else 33
                self.master.after(delay, self.update_replay_loop)
            else:
                self.pause_replay()

    def skip_forward(self):
        self.frame_slider.set(min(self.frame_slider.get() + (10 * self.fps), self.max_frames - 1))

    def skip_backward(self):
        self.frame_slider.set(max(self.frame_slider.get() - (10 * self.fps), 0))

    # ---------------- Helpers: Math/REBA ----------------
    def get_skeleton_definition(self):
        connections_mp = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # Head
            (11, 12), (24, 23),  # Body
            (11, 13), (13, 15), (15, 17), (17, 19), (19, 15), (15, 21),
            (12, 14), (14, 16), (16, 18), (18, 20), (20, 16), (16, 22),
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
            (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
        ]

        # Dynamic Spine Connections
        if self.spine_mode == 7:
            # Sh -> S1 -> S2 -> S3 -> S4 -> S5 -> Hip
            connections_mp += [(33, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 34)]
        else:
            # Sh -> S1 -> S2 -> S3 -> S4 -> Hip
            connections_mp += [(33, 35), (35, 36), (36, 37), (37, 38), (38, 34)]

        full_connections = []
        for p1, p2 in connections_mp:
            idx1 = self.landmark_map[p1] if p1 < 33 else p1
            idx2 = self.landmark_map[p2] if p2 < 33 else p2
            full_connections.append((idx1, idx2))
        return full_connections

    def calculate_velocities(self, curr, prev, fps):
        velocities = {}
        if prev is None: return velocities
        m = self.landmark_map
        velocities['Neck'] = np.linalg.norm(((curr[m[7]] + curr[m[8]]) / 2) - ((prev[m[7]] + prev[m[8]]) / 2)) * fps
        velocities['Trunk'] = np.linalg.norm(
            ((curr[m[11]] + curr[m[12]]) / 2) - ((prev[m[23]] + prev[m[24]]) / 2)) * fps
        return velocities

    def calculate_angle(self, p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        dp = np.dot(v1, v2)
        m1, m2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if m1 == 0 or m2 == 0: return 0
        return math.degrees(math.acos(np.clip(dp / (m1 * m2), -1.0, 1.0)))

    def calculate_angle_vertical(self, p1, p2):
        return self.calculate_angle(p1, p2, p2 + np.array([0, 0, 1]))

    def calculate_reba_score(self, kpts):
        angles, scores = {}, {}
        m = self.landmark_map
        try:
            force_str = self.force_load_var.get()
            force_load_score = int(force_str.split(' ')[0]) if force_str else 0
        except:
            force_load_score = 0
        coupling_score, activity_score = 0, 0

        # Group A
        head_mid = (kpts[m[7]] + kpts[m[8]]) / 2
        sh_mid = (kpts[m[11]] + kpts[m[12]]) / 2
        hip_mid = (kpts[m[23]] + kpts[m[24]]) / 2

        angles["Neck"] = self.calculate_angle_vertical(head_mid, sh_mid)
        scores["Neck"] = 1 if angles["Neck"] <= 20 else 2

        angles["Trunk"] = self.calculate_angle_vertical(sh_mid, hip_mid)
        if angles["Trunk"] <= 5:
            scores["Trunk"] = 1
        elif angles["Trunk"] <= 20:
            scores["Trunk"] = 2
        elif angles["Trunk"] <= 60:
            scores["Trunk"] = 3
        else:
            scores["Trunk"] = 4

        l_knee = self.calculate_angle(kpts[m[23]], kpts[m[25]], kpts[m[27]])
        r_knee = self.calculate_angle(kpts[m[24]], kpts[m[26]], kpts[m[28]])
        angles["L Knee"], angles["R Knee"] = abs(180 - l_knee), abs(180 - r_knee)
        scores["L Knee"] = 1 if angles["L Knee"] < 30 else 2
        scores["R Knee"] = 1 if angles["R Knee"] < 30 else 2
        legs_score = max(scores["L Knee"], scores["R Knee"])
        scores["Legs"] = legs_score  # For export

        idx_trunk = max(0, min(scores["Trunk"] - 1, 3))
        idx_neck = max(0, min(scores["Neck"] - 1, 2))
        idx_leg = max(0, min(legs_score - 1, 3))
        score_a = self.TABLE_A[idx_trunk][idx_neck][idx_leg] + force_load_score
        scores["Score A"] = score_a

        # Group B
        angles["L Upper Arm"] = self.calculate_angle_vertical(kpts[m[13]], kpts[m[11]])
        angles["R Upper Arm"] = self.calculate_angle_vertical(kpts[m[14]], kpts[m[12]])

        def rate_upper(a):
            return 1 if a <= 20 else 2 if a <= 45 else 3 if a <= 90 else 4

        scores["L Upper Arm"], scores["R Upper Arm"] = rate_upper(angles["L Upper Arm"]), rate_upper(
            angles["R Upper Arm"])
        upper_score = max(scores["L Upper Arm"], scores["R Upper Arm"])
        scores["Upper Arm"] = upper_score

        angles["L Lower Arm"] = self.calculate_angle(kpts[m[11]], kpts[m[13]], kpts[m[15]])
        angles["R Lower Arm"] = self.calculate_angle(kpts[m[12]], kpts[m[14]], kpts[m[16]])

        def rate_lower(a):
            return 1 if 60 <= a <= 100 else 2

        scores["L Lower Arm"], scores["R Lower Arm"] = rate_lower(angles["L Lower Arm"]), rate_lower(
            angles["R Lower Arm"])
        lower_score = max(scores["L Lower Arm"], scores["R Lower Arm"])
        scores["Lower Arm"] = lower_score

        l_wrist = self.calculate_angle(kpts[m[13]], kpts[m[15]], kpts[m[19]])
        r_wrist = self.calculate_angle(kpts[m[14]], kpts[m[16]], kpts[m[20]])
        angles["L Wrist"], angles["R Wrist"] = abs(180 - l_wrist), abs(180 - r_wrist)
        scores["L Wrist"] = 1 if angles["L Wrist"] <= 15 else 2
        scores["R Wrist"] = 1 if angles["R Wrist"] <= 15 else 2
        wrist_score = max(scores["L Wrist"], scores["R Wrist"])
        scores["Wrist"] = wrist_score

        idx_u = max(0, min(upper_score - 1, 5))
        idx_l = max(0, min(lower_score - 1, 1))
        idx_w = max(0, min(wrist_score - 1, 2))
        score_b = self.TABLE_B[idx_u][idx_l][idx_w] + coupling_score
        scores["Score B"] = score_b

        # Final
        idx_a, idx_b = max(0, min(score_a - 1, 11)), max(0, min(score_b - 1, 11))
        final = self.TABLE_C[idx_a][idx_b] + activity_score

        if final <= 1:
            r = "Negligible"
        elif final <= 3:
            r = "Low"
        elif final <= 7:
            r = "Medium"
        elif final <= 10:
            r = "High"
        else:
            r = "Very High"

        return final, r, angles, scores

    def initialize_reba_tables(self):
        self.TABLE_A = np.array([
            [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]],
            [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
            [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
            [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]
        ])
        self.TABLE_B = np.array([
            [[1, 2, 2], [1, 2, 3]],
            [[2, 3, 4], [3, 4, 5]],
            [[3, 4, 5], [4, 5, 5]],
            [[4, 5, 6], [5, 6, 7]],
            [[5, 6, 7], [6, 7, 8]],
            [[6, 7, 8], [7, 8, 9]]
        ])
        self.TABLE_C = np.array([
            [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
            [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
            [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
            [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
            [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
            [5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10],
            [6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10, 10],
            [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
            [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
            [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
            [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
            [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12]
        ])

    # ---------------- Full Res & View Control ----------------
    def open_fullres_view(self, cam_id):
        if cam_id in self.fullres_windows and self.fullres_windows[cam_id][0].winfo_exists():
            self.fullres_windows[cam_id][0].lift()
            return
        win = tk.Toplevel(self.master)
        win.title(f"Cam{cam_id} Full Res")
        lbl = tk.Label(win)
        lbl.pack(fill=BOTH, expand=True)
        self.fullres_windows[cam_id] = (win, lbl)

    def update_fullres_view(self, cam_id, frame):
        if cam_id in self.fullres_windows and self.fullres_windows[cam_id][0].winfo_exists():
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.fullres_images[cam_id] = img
            self.fullres_windows[cam_id][1].config(image=img)

    def rotate_3d_elevation(self, d):
        self.view_elev += self.ELEVATION_STEP * d
        self.ax_3d.view_init(elev=self.view_elev, azim=self.view_azim)
        self.canvas_3d.draw()

    def cycle_3d_azimuth(self, d):
        self.current_side_view_index = (self.current_side_view_index + d) % len(self.side_view_angles)
        self.view_azim = self.side_view_angles[self.current_side_view_index]
        self.ax_3d.view_init(elev=self.view_elev, azim=self.view_azim)
        self.canvas_3d.draw()

    def reset_3d_view(self):
        self.view_elev = 10
        self.view_azim = -90
        self.ax_3d.view_init(elev=self.view_elev, azim=self.view_azim)
        self.canvas_3d.draw()

    def rotate_view_button(self, cam_id):
        self.view_rotations[cam_id] = (self.view_rotations[cam_id] + 1) % 4
        if self.is_replaying: self.update_frame_display(self.frame_slider.get())

    # ---------------- File IO & Exports ----------------
    def save_csv(self):
        if not hasattr(self, 'all_kpts_3d') or len(self.all_kpts_3d) == 0:
            tk.messagebox.showwarning('Export Error', 'No data available to export.')
            return

        dialog = OptionDialog(self.master, ["Raw Coordinates (XYZ)", "REBA Analysis (Angles/Scores)"],
                              prompt="Select export type:")
        self.master.wait_window(dialog)
        choice = dialog.selected_option.get()
        if not choice: return

        default_name = "3D_Keypoints.csv" if "XYZ" in choice else "REBA_Analysis.csv"
        save_path = tk.filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")],
                                                    initialfile=default_name)
        if not save_path: return

        try:
            if "XYZ" in choice:
                self._export_raw_xyz(save_path)
            else:
                self._export_reba_analysis(save_path)
            tk.messagebox.showinfo("Export Success", f"Saved to {save_path}")
        except Exception as e:
            tk.messagebox.showerror("Export Error", str(e))

    def _export_raw_xyz(self, save_path):
        data = []
        # Define the base names including standard midpoints
        base_spine_names = ["Mid_Shoulder", "Mid_Hip", "Spine_1", "Spine_2", "Spine_3", "Spine_4"]

        # Check if the session was processed with 7 keypoints (index 39 exists)
        # Total kpts = 33 (standard) + 6 (base spine) = 39.
        # If len is 40, then index 39 (Spine 5) exists.
        has_spine_5 = len(self.all_kpts_3d[0]) > 39

        full_column_names = POSE_KEYPOINT_NAMES + base_spine_names
        if has_spine_5:
            full_column_names.append("Spine_5")

        for f in range(len(self.all_kpts_3d)):
            row = {"Frame": f, "Time (s)": round(f / self.fps, 3)}
            kpts = self.all_kpts_3d[f]
            for i, name in enumerate(POSE_KEYPOINT_NAMES):
                if kpts[i][0] != -1:
                    row[f"{name}_x"] = kpts[i][0]
                    row[f"{name}_y"] = kpts[i][1]
                    row[f"{name}_z"] = kpts[i][2]
                else:
                    row[f"{name}_x"] = row[f"{name}_y"] = row[f"{name}_z"] = ""
            data.append(row)

        pd.DataFrame(data).to_csv(save_path, index=False)

    def _export_reba_analysis(self, save_path):
        data = []
        self.progress_var.set(0)
        total = len(self.all_kpts_3d)
        for f in range(total):
            final, risk, ang, sco = self.calculate_reba_score(self.all_kpts_3d[f])
            row = {
                "Frame": f, "Time": round(f / self.fps, 3), "Final Score": final, "Risk": risk,
                "Score A": sco["Score A"], "Score B": sco["Score B"],
                "Score Neck": sco["Neck"], "Score Trunk": sco["Trunk"],
                "Score L Leg": sco["L Knee"], "Score R Leg": sco["R Knee"],
                "Score L Upper": sco["L Upper Arm"], "Score R Upper": sco["R Upper Arm"],
                "Score L Lower": sco["L Lower Arm"], "Score R Lower": sco["R Lower Arm"],
                "Score L Wrist": sco["L Wrist"], "Score R Wrist": sco["R Wrist"],
                "Angle Neck": round(ang["Neck"], 1), "Angle Trunk": round(ang["Trunk"], 1),
                "Angle L Knee": round(ang["L Knee"], 1), "Angle R Knee": round(ang["R Knee"], 1),
                "Angle L Upper": round(ang["L Upper Arm"], 1), "Angle R Upper": round(ang["R Upper Arm"], 1),
                "Angle L Lower": round(ang["L Lower Arm"], 1), "Angle R Lower": round(ang["R Lower Arm"], 1),
                "Angle L Wrist": round(ang["L Wrist"], 1), "Angle R Wrist": round(ang["R Wrist"], 1)
            }
            data.append(row)
            if f % 50 == 0:
                self.progress_var.set((f / total) * 100)
                self.master.update_idletasks()
        self.progress_var.set(0)
        pd.DataFrame(data).to_csv(save_path, index=False)

    def save_2d_video(self):
        if not self.cam_ids: return
        dialog = OptionDialog(self.master, [f"Cam{cid}" for cid in self.cam_ids], "Select View:")
        self.master.wait_window(dialog)
        sel = dialog.selected_option.get()
        if not sel: return
        cid = int(sel.replace("Cam", ""))
        files = self.frame_2d_paths[cid]
        if not files: return

        path = tk.filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI", "*.avi")])
        if path:
            img = cv2.imread(files[0])
            h, w = img.shape[:2]
            vid = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), self.fps, (w, h))
            self.progress_var.set(0)
            for i, f in enumerate(files):
                vid.write(cv2.imread(f))
                if i % 10 == 0:
                    self.progress_var.set((i / len(files)) * 100)
                    self.master.update_idletasks()
            vid.release()
            self.progress_var.set(0)
            tk.messagebox.showinfo("Success", f"Saved to {path}")

    def save(self):
        import glob
        files = sorted(glob.glob('./images/????????.png'))
        if not files: return
        path = tk.filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI", "*.avi")])
        if path:
            img = cv2.imread(files[0])
            h, w = img.shape[:2]
            vid = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), 10, (w, h))
            self.progress_var.set(0)
            for i, f in enumerate(files):
                vid.write(cv2.imread(f))
                if i % 10 == 0:
                    self.progress_var.set((i / len(files)) * 100)
                    self.master.update_idletasks()
            vid.release()
            self.progress_var.set(0)
            tk.messagebox.showinfo("Success", f"Saved to {path}")

    def save_session(self):
        if not hasattr(self, 'all_kpts_3d'): return
        path = tk.filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            data = {
                "kpts_3d_path": os.path.abspath('kpts_3d.dat'),
                "cam_ids": self.cam_ids,
                "max_frames": self.max_frames,
                "image_dir": os.path.abspath('./images/'),
                "num_keypoints": len(pose_keypoints),
                "fps": self.fps
            }
            with open(path, 'w') as f: json.dump(data, f, indent=4)
            tk.messagebox.showinfo("Success", "Session saved.")

    def load_session(self):
        path = tk.filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path: return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            base = os.path.dirname(path)
            kp_path = data["kpts_3d_path"]
            if not os.path.isabs(kp_path): kp_path = os.path.join(base, kp_path)

            self.cam_ids = data["cam_ids"]
            self.max_frames = data["max_frames"]
            self.fps = data.get("fps", 30)

            self.all_kpts_3d = np.loadtxt(kp_path).reshape((self.max_frames, data["num_keypoints"], 3))

            img_dir = data["image_dir"]
            if img_dir and not os.path.isabs(img_dir): img_dir = os.path.join(base, img_dir)
            self.frame_2d_paths = [[] for _ in range(3)]
            for c in self.cam_ids:
                for i in range(self.max_frames):
                    self.frame_2d_paths[c].append(os.path.join(img_dir, f'cam{c}_frame{str(i).zfill(8)}.png'))

            self.status_bar.config(text="Session Loaded.")
            self.action_menu.entryconfig("Replay", state="normal")
            self.export_menu.entryconfig("Export Session Package (.zip)", state="normal")
            self.export_menu.entryconfig("Export 2D Video View", state="normal")
            self.export_menu.entryconfig("Export 3D Skeleton Video", state="normal")
            self.export_menu.entryconfig("Export CSV Data", state="normal")
            self.replay()
        except Exception as e:
            tk.messagebox.showerror("Error", str(e))

    def export_session_package(self):
        if not hasattr(self, 'all_kpts_3d'): return
        path = tk.filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("ZIP", "*.zip")])
        if path:
            with zipfile.ZipFile(path, 'w') as z:
                if os.path.exists('kpts_3d.dat'): z.write('kpts_3d.dat')
                if os.path.exists('images'):
                    for r, _, fs in os.walk('images'):
                        for f in fs: z.write(os.path.join(r, f), os.path.relpath(os.path.join(r, f), os.getcwd()))
                data = {
                    "kpts_3d_path": "kpts_3d.dat", "cam_ids": self.cam_ids, "max_frames": self.max_frames,
                    "image_dir": "images/", "num_keypoints": len(pose_keypoints), "fps": self.fps
                }
                z.writestr('session.json', json.dumps(data, indent=4))
            tk.messagebox.showinfo("Success", f"Packaged to {path}")

    def import_session_package(self):
        zip_path = tk.filedialog.askopenfilename(filetypes=[("ZIP", "*.zip")])
        if not zip_path: return
        ex_dir = "imported_session"
        if os.path.exists(ex_dir): shutil.rmtree(ex_dir)
        os.makedirs(ex_dir)
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(ex_dir)
            if os.path.exists(os.path.join(ex_dir, 'session.json')):
                tk.messagebox.showinfo("Import",
                                       "Extracted. Please load 'session.json' from 'imported_session' folder.")
                self.load_session()
        except Exception as e:
            tk.messagebox.showerror("Error", str(e))

    # ---------------- Recent Files Logic ----------------
    def load_recent_files(self):
        try:
            if os.path.exists(self.RECENT_FILES_PATH):
                with open(self.RECENT_FILES_PATH, 'r') as f:
                    self.recent_files = json.load(f)
            else:
                self.recent_files = []
        except:
            self.recent_files = []

    def add_to_recent_files(self, path):
        if path in self.recent_files: self.recent_files.remove(path)
        self.recent_files.insert(0, path)
        self.recent_files = self.recent_files[:self.MAX_RECENT_FILES]
        try:
            with open(self.RECENT_FILES_PATH, 'w') as f:
                json.dump(self.recent_files, f)
        except:
            pass
        self.update_recent_files_menu()

    def update_recent_files_menu(self):
        self.recent_files_menu.delete(0, tk.END)
        if not self.recent_files:
            self.recent_files_menu.add_command(label="(No recent videos)", state='disabled')
        else:
            for p in self.recent_files:
                self.recent_files_menu.add_command(label=os.path.basename(p),
                                                   command=lambda x=p: self._open_video_file(x))


if __name__ == '__main__':
    root = ttk.Window(themename="superhero")
    app = Application(root)
    root.mainloop()