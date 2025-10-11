import cv2
import tkinter as tk
from tkinter import ttk, simpledialog, filedialog
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
import threading
import mediapipe as mp
import numpy as np
import sys
import os
from utils import DLT_multi, get_projection_matrix, write_keypoints_to_disk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import math
import json

plt.style.use('seaborn-v0_8')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [1280, 720]

pose_keypoints = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Head (11 points)
    11, 12, 13, 14, 15, 16,  # Body and Arms (6 points)
    19, 20,  # Index Fingers (2 points)
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32  # Hips and Legs (10 points)
]


class OptionDialog(tk.Toplevel):
    def __init__(self, master, options):
        super().__init__(master)
        self.title("Browse File");
        self.geometry("400x100");
        self.resizable(width=False, height=False)
        x, y = (1200 - self.winfo_reqwidth()) // 2, (674 - self.winfo_reqheight()) // 2
        self.geometry(f"+{x}+{y}")
        self.selected_option = tk.StringVar()
        tk.Label(self, text="Select Video From:").pack(pady=10)
        for option in options:
            tk.Button(self, text=option, width=15, height=2, command=lambda o=option: self.on_option_selected(o)).pack(
                side=tk.LEFT, padx=5)

    def on_option_selected(self, option):
        self.selected_option.set(option);
        self.destroy()


class Application:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive 3D Motion Reconstruction with REBA")
        self.master.configure(bg="#cccccc")
        self.master.resizable(width=True, height=True)
        self.master.state('zoomed')

        style = ThemedStyle(root);
        style.set_theme("yaru")
        self.style = ttk.Style();
        self.style.configure("Custom.TFrame", background="#cccccc")

        self.setup_menus()
        self.setup_widgets()
        self.initialize_reba_tables()

        self.lock = threading.Lock()
        self.cap0 = self.cap1 = self.cap2 = None
        self.previous_kpts = None
        self.is_processing = False
        self.is_replaying = False
        self.fps = 30
        self.frame_paths = []

    def setup_menus(self):
        self.menubar = tk.Menu(self.master)
        self.master.config(menu=self.menubar)

        self.action_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.export_menu = tk.Menu(self.menubar, tearoff=0)

        self.menubar.add_cascade(label="Actions", menu=self.action_menu)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.menubar.add_cascade(label="Export", menu=self.export_menu)

        # Populate Actions Menu
        self.action_menu.add_command(label="Upload Video", command=self.choose_file)
        self.action_menu.add_command(label="Connect USB Camera", command=self.connect_usb_camera)
        self.action_menu.add_command(label="Detect Pose", command=self.detect)
        self.action_menu.add_command(label="Replay", command=self.replay, state='disabled')
        self.action_menu.add_separator()
        self.action_menu.add_command(label="Stop Detection", command=self.stop_detection, state='disabled')

        # Populate File Menu
        self.file_menu.add_command(label="Load Session", command=self.load_session)
        self.file_menu.add_command(label="Save Session", command=self.save_session, state='disabled')
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Reload Application", command=self.reload_app)
        self.file_menu.add_command(label="Exit", command=self.exit_app)

        # Populate Export Menu
        self.export_menu.add_command(label="Export 3D Skeleton Video", command=self.save, state='disabled')
        self.export_menu.add_separator()
        self.export_menu.add_command(label="Export 3D Keypoints CSV", command=self.save_csv, state='disabled')

    def setup_widgets(self):
        reba_input_frame = ttk.Frame(self.master, style="Custom.TFrame");
        reba_input_frame.pack(pady=5, fill='x')
        reba_input_frame.grid_columnconfigure(0, weight=1);
        reba_input_frame.grid_columnconfigure(1, weight=1)
        self.reba_label = tk.Label(reba_input_frame, text="REBA Score: -", font=("Helvetica", 14, "bold"),
                                   background="#cccccc")
        self.reba_label.grid(row=0, column=0, sticky="e", padx=(0, 20))
        input_frame_inner = ttk.Frame(reba_input_frame, style="Custom.TFrame")
        input_frame_inner.grid(row=0, column=1, sticky="w", padx=(20, 0))
        ttk.Label(input_frame_inner, text="Force/Load Score:", background="#cccccc").pack(side='left', padx=5)
        self.force_load_var = tk.StringVar()
        force_load_options = ['0 (<11 lbs)', '1 (11-22 lbs)', '2 (>22 lbs)', '3 (+Shock/Rapid)']
        self.force_load_combobox = ttk.Combobox(input_frame_inner, textvariable=self.force_load_var,
                                                values=force_load_options, state='readonly', width=15)
        self.force_load_combobox.set(force_load_options[0]);
        self.force_load_combobox.pack(side='left')

        content_frame = ttk.Frame(self.master, style="Custom.TFrame");
        content_frame.pack(fill='both', expand=True, padx=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)
        content_frame.grid_columnconfigure(2, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        table_frame = ttk.Frame(content_frame, style="Custom.TFrame");
        table_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        self.style.configure("Custom.Treeview", font=('Helvetica', 10));
        self.style.configure("Custom.Treeview.Heading", font=('Helvetica', 11, 'bold'))
        self.angle_table = ttk.Treeview(table_frame, columns=('part', 'angle', 'score', 'velocity'), show='headings',
                                        style="Custom.Treeview")
        self.angle_table.heading('part', text='Body Part');
        self.angle_table.heading('angle', text='Angle (°)')
        self.angle_table.heading('score', text='Score');
        self.angle_table.heading('velocity', text='Velocity (mm/s)')
        self.angle_table.column('part', width=150, anchor='w');
        self.angle_table.column('angle', width=100, anchor='center')
        self.angle_table.column('score', width=80, anchor='center');
        self.angle_table.column('velocity', width=100, anchor='center')
        self.angle_table_rows = {}
        parts = ["Neck", "Trunk", "L Knee", "R Knee", "L Upper Arm", "R Upper Arm", "L Lower Arm", "R Lower Arm",
                 "L Wrist", "R Wrist"]
        for part in parts: self.angle_table_rows[part] = self.angle_table.insert('', 'end',
                                                                                 values=(part, '-', '-', '-'))
        self.angle_table.pack(side='left', fill='both', expand=True)

        self.fig_3d = plt.figure(figsize=(3, 3));
        self.fig_3d.patch.set_facecolor('#cccccc')
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d');
        self.fig_3d.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=content_frame);
        self.canvas_3d.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        right_frame = ttk.Frame(content_frame, style="Custom.TFrame");
        right_frame.grid(row=0, column=2, sticky="nsew", padx=10)
        self.label1 = tk.Label(right_frame, background="#cccccc");
        self.label1.pack(pady=2, expand=True, fill='both')
        self.label2 = tk.Label(right_frame, background="#cccccc");
        self.label2.pack(pady=2, expand=True, fill='both')
        self.label3 = tk.Label(right_frame, background="#cccccc");
        self.label3.pack(pady=2, expand=True, fill='both')

        self.slider_frame = ttk.Frame(self.master, style="Custom.TFrame");
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.slider_frame, orient=tk.HORIZONTAL, mode='determinate',
                                            variable=self.progress_var)
        self.frame_slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      command=self.update_frame_display)
        self.progress_bar.pack(fill='x');

        # --- NEW: Playback control buttons ---
        self.controls_frame = ttk.Frame(self.slider_frame, style="Custom.TFrame")
        self.play_button = ttk.Button(self.controls_frame, text="▶ Play", command=self.play_replay, state='disabled')
        self.pause_button = ttk.Button(self.controls_frame, text="❚❚ Pause", command=self.pause_replay,
                                       state='disabled')
        self.skip_backward_button = ttk.Button(self.controls_frame, text="<< -10s", command=self.skip_backward,
                                               state='disabled')
        self.skip_forward_button = ttk.Button(self.controls_frame, text=">> +10s", command=self.skip_forward,
                                              state='disabled')

        self.skip_backward_button.pack(side=tk.LEFT, padx=5)
        self.play_button.pack(side=tk.LEFT, padx=5)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.skip_forward_button.pack(side=tk.LEFT, padx=5)
        self.controls_frame.pack_forget()

        self.status_bar = tk.Label(self.master, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)

        self.status_bar.pack(side='bottom', fill='x')
        self.slider_frame.pack(pady=5, fill='x', padx=20, side='bottom')

    def exit_app(self):
        self.is_processing = False
        self.is_replaying = False
        self.master.quit()

    def reload_app(self):
        os.execl(sys.executable, sys.executable, *sys.argv)

    def choose_file(self):
        self.action_menu.entryconfig("Replay", state="disabled")
        self.export_menu.entryconfig("Export 3D Skeleton Video", state="disabled")
        self.export_menu.entryconfig("Export 3D Keypoints CSV", state="disabled")
        self.frame_2d_paths = [[] for _ in range(3)]
        self.frame_paths = []
        dialog = OptionDialog(self.master, ["Cam0 (Front View)", "Cam1 (Side View)", "Cam2 (Third View)"])
        self.master.wait_window(dialog);
        input_source = dialog.selected_option.get()
        if not input_source: return
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi;*.mp4;*.mov")])
        if not file_path: return
        if "Cam0" in input_source:
            self.cap0 = cv2.VideoCapture(file_path)
        elif "Cam1" in input_source:
            self.cap1 = cv2.VideoCapture(file_path)
        elif "Cam2" in input_source:
            self.cap2 = cv2.VideoCapture(file_path)
        self.status_bar.config(text=f"{input_source}: {os.path.basename(file_path)} successfully uploaded.")

    def connect_usb_camera(self):
        self.action_menu.entryconfig("Replay", state="disabled")
        self.export_menu.entryconfig("Export 3D Skeleton Video", state="disabled")
        self.export_menu.entryconfig("Export 3D Keypoints CSV", state="disabled")

        dialog = OptionDialog(self.master, ["Cam0 (Front View)", "Cam1 (Side View)", "Cam2 (Third View)"])
        self.master.wait_window(dialog)
        input_source = dialog.selected_option.get()
        if not input_source:
            return

        cam_index = simpledialog.askinteger("Camera Index", "Enter camera index (e.g., 0, 1, 2):", parent=self.master)
        if cam_index is None:
            return

        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            tk.messagebox.showerror("Connection Failed", f"Could not open camera at index {cam_index}.")
            return

        if "Cam0" in input_source:
            if self.cap0: self.cap0.release()
            self.cap0 = cap
        elif "Cam1" in input_source:
            if self.cap1: self.cap1.release()
            self.cap1 = cap
        elif "Cam2" in input_source:
            if self.cap2: self.cap2.release()
            self.cap2 = cap

        self.status_bar.config(text=f"{input_source} connected to USB camera at index {cam_index}.")

    def stop_detection(self):
        self.is_processing = False

    def cleanup_images(self):
        """Clears the images directory before a new session."""
        if os.path.exists('./images'):
            for filename in os.listdir('./images'):
                file_path = os.path.join('./images', filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def detect(self):
        if self.is_processing: tk.messagebox.showinfo('Processing', 'Detection is already running.'); return
        if len([c for c in [self.cap0, self.cap1, self.cap2] if c is not None]) < 2: tk.messagebox.showwarning(
            'No Input', 'Please upload at least 2 video files to detect.'); return

        self.cleanup_images()
        self.is_processing = True
        self.progress_bar.pack(fill='x')
        self.frame_slider.pack_forget()
        self.controls_frame.pack_forget()

        self.action_menu.entryconfig("Upload Video", state="disabled")
        self.action_menu.entryconfig("Connect USB Camera", state="disabled")
        self.action_menu.entryconfig("Stop Detection", state="normal")
        self.action_menu.entryconfig("Replay", state="disabled")
        self.export_menu.entryconfig("Export 3D Skeleton Video", state="disabled")
        self.export_menu.entryconfig("Export 3D Keypoints CSV", state="disabled")
        self.file_menu.entryconfig("Save Session", state="disabled")

        self.status_bar.config(text="Processing...");
        self.progress_var.set(0)
        threading.Thread(target=self.play_video).start()

    def replay(self):
        if hasattr(self, 'all_kpts_3d') and self.all_kpts_3d:
            self.progress_bar.pack_forget()
            self.frame_slider.pack(fill='x', pady=5)
            self.controls_frame.pack(pady=5)

            self.frame_slider.configure(state='normal', to=len(self.all_kpts_3d) - 1)
            self.play_button.config(state='normal')
            self.pause_button.config(state='disabled')
            self.skip_backward_button.config(state='normal')
            self.skip_forward_button.config(state='normal')

            self.frame_slider.set(0)
            self.update_frame_display(0)

    def play_video(self):
        cam_info = []
        if self.cap0: cam_info.append((0, self.cap0, self.label1))
        if self.cap1: cam_info.append((1, self.cap1, self.label2))
        if self.cap2: cam_info.append((2, self.cap2, self.label3))
        self.cam_ids = [info[0] for info in cam_info];
        caps = [info[1] for info in cam_info]
        self.fps = caps[0].get(cv2.CAP_PROP_FPS);
        if self.fps == 0: self.fps = 30
        P_list = [get_projection_matrix(cam_id) for cam_id, _, _ in cam_info]

        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        is_live = any(fc <= 0 for fc in frame_counts)

        max_frames = 0
        if not is_live:
            max_frames = min(frame_counts) if frame_counts else 0
            self.max_frames = max_frames
        else:
            self.progress_bar.config(mode='indeterminate')
            self.progress_bar.start(10)

        poses = {id: mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) for id, _, _ in cam_info}
        kpts_2d = {cam_id: [] for cam_id, _, _ in cam_info};
        self.all_kpts_3d = []
        self.frame_2d_paths = [[] for _ in range(3)]
        skeleton_connections = self.get_skeleton_definition();
        last_good_kpts = None

        index = 0
        while self.is_processing:
            rets, frames = zip(*[cap.read() for cap in caps])
            if not all(rets):
                self.is_processing = False
                break

            current_frame_2d_kpts = []
            for i, (cam_id, _, label) in enumerate(cam_info):
                result = poses[cam_id].process(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
                frame_keypoints = self.extract_keypoints(result, frames[i].shape)
                kpts_2d[cam_id].append(frame_keypoints);
                current_frame_2d_kpts.append(frame_keypoints)
                self.display_2d_frame(frames[i], result, label, index, cam_id)
            frame_p3ds = self.triangulate_points(P_list, current_frame_2d_kpts)
            if last_good_kpts is not None:
                for i in range(len(frame_p3ds)):
                    if np.all(frame_p3ds[i] == -1): frame_p3ds[i] = last_good_kpts[i]
            if last_good_kpts is None:
                last_good_kpts = np.copy(frame_p3ds)
            else:
                good_indices = ~np.all(frame_p3ds == -1, axis=1)
                last_good_kpts[good_indices] = frame_p3ds[good_indices]
            valid_kpts = frame_p3ds[~np.all(frame_p3ds == -1, axis=1)]
            frame_p3ds_centered = frame_p3ds - np.mean(valid_kpts, axis=0) if len(valid_kpts) > 0 else frame_p3ds
            self.all_kpts_3d.append(frame_p3ds_centered)
            self.draw_3d_skeleton(frame_p3ds_centered, skeleton_connections)

            self.fig_3d.savefig(f'./images/{str(index).zfill(8)}.png', bbox_inches='tight', pad_inches=0, dpi=100)
            self.canvas_3d.draw()

            if is_live:
                self.status_bar.config(text=f"Processing Live Frame: {index}")
            else:
                self.progress_var.set((index / max_frames) * 100)
                self.status_bar.config(text=f"Processing Frame: {index + 1}/{max_frames}")
                if index >= max_frames - 1:
                    self.is_processing = False
            index += 1

        if is_live:
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')
            self.max_frames = len(self.all_kpts_3d)

        self.finalize_processing()

    def draw_3d_skeleton(self, kpts, skeleton_connections):
        self.ax_3d.clear();
        self.ax_3d.set_facecolor('#cccccc');
        self.ax_3d.grid(color='white', linestyle='-', linewidth=0.5)
        self.ax_3d.tick_params(axis='x', colors='white');
        self.ax_3d.tick_params(axis='y', colors='white');
        self.ax_3d.tick_params(axis='z', colors='white')
        self.ax_3d.xaxis.label.set_color('white');
        self.ax_3d.yaxis.label.set_color('white');
        self.ax_3d.zaxis.label.set_color('white')
        self.ax_3d.view_init(elev=90, azim=180)
        for p1_idx, p2_idx in skeleton_connections:
            if not np.all(kpts[p1_idx] == -1) and not np.all(kpts[p2_idx] == -1):
                self.ax_3d.plot([kpts[p1_idx, 0], kpts[p2_idx, 0]], [kpts[p1_idx, 1], kpts[p2_idx, 1]],
                                [kpts[p1_idx, 2], kpts[p2_idx, 2]], linewidth=2, c='red')
        if not np.all(kpts[11:13] == -1) and not np.all(kpts[7:9] == -1):
            shoulder_midpoint = (kpts[11] + kpts[12]) / 2.0;
            head_center = (kpts[7] + kpts[8]) / 2.0
            if np.linalg.norm(head_center - shoulder_midpoint) < 5.0:
                self.ax_3d.plot([head_center[0], shoulder_midpoint[0]], [head_center[1], shoulder_midpoint[1]],
                                [head_center[2], shoulder_midpoint[2]], linewidth=2, c='red')
        valid_kpts = kpts[~np.all(kpts == -1, axis=1)]
        if valid_kpts.shape[0] > 0:
            max_range = np.array([valid_kpts[:, i].max() - valid_kpts[:, i].min() for i in range(3)]).max() / 2.0;
            mid = np.mean(valid_kpts, axis=0)
            self.ax_3d.set_xlim(mid[0] - max_range, mid[0] + max_range);
            self.ax_3d.set_ylim(mid[1] - max_range, mid[1] + max_range);
            self.ax_3d.set_zlim(mid[2] - max_range, mid[2] + max_range)
        else:
            self.ax_3d.set_xlim3d(-10, 10);
            self.ax_3d.set_ylim3d(-10, 10);
            self.ax_3d.set_zlim3d(-10, 10)

        self.ax_3d.set_xlabel('x');
        self.ax_3d.set_ylabel('y');
        self.ax_3d.set_zlabel('z')

    def get_skeleton_definition(self):
        torso = [[11, 12], [12, 20], [20, 19], [19, 11]];
        head = [[7, 3], [3, 2], [2, 1], [1, 0], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]]
        arm_l = [[11, 13], [13, 15]];
        arm_r = [[12, 14], [14, 16]]
        leg_l = [[19, 21], [21, 23]];
        leg_r = [[20, 22], [22, 24]]
        foot_l = [[23, 25], [25, 27], [23, 27]];
        foot_r = [[24, 26], [26, 28], [24, 28]]
        return torso + head + arm_l + arm_r + leg_l + leg_r + foot_l + foot_r

    def extract_keypoints(self, result, shape):
        if result.pose_landmarks:
            all_landmarks = result.pose_landmarks.landmark
            return [[int(round(all_landmarks[p_idx].x * shape[1])), int(round(all_landmarks[p_idx].y * shape[0]))] for
                    p_idx in pose_keypoints]
        return [[-1, -1]] * len(pose_keypoints)

    def display_2d_frame(self, frame, result, label, index, cam_id):
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4,
                                                                               circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4))
        frame_resized = cv2.resize(frame, (300, 200));
        frame_2d_path = f'./images/cam{cam_id}_frame{str(index).zfill(8)}.png'
        cv2.imwrite(frame_2d_path, frame_resized);
        self.frame_2d_paths[cam_id].append(frame_2d_path)
        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
        self.lock.acquire();
        label.config(image=img_tk);
        label.image = img_tk;
        self.lock.release()

    def triangulate_points(self, P_list, kpts_2d):
        return np.array(
            [DLT_multi(P_list, [cam_kpts[i] for cam_kpts in kpts_2d]) for i in range(len(pose_keypoints))]).reshape(
            (len(pose_keypoints), 3))

    def finalize_processing(self):
        for cap in [self.cap0, self.cap1, self.cap2]:
            if cap: cap.release()
        self.cap0 = self.cap1 = self.cap2 = None

        self.action_menu.entryconfig("Upload Video", state="normal")
        self.action_menu.entryconfig("Connect USB Camera", state="normal")
        self.action_menu.entryconfig("Stop Detection", state="disabled")

        if hasattr(self, 'all_kpts_3d') and len(self.all_kpts_3d) > 0:
            write_keypoints_to_disk('kpts_3d.dat', np.array(self.all_kpts_3d))
            self.status_bar.config(text="Processing Complete. Click REPLAY to interact.")
            self.action_menu.entryconfig("Replay", state="normal")
            self.file_menu.entryconfig("Save Session", state="normal")
            self.export_menu.entryconfig("Export 3D Skeleton Video", state="normal")
            self.export_menu.entryconfig("Export 3D Keypoints CSV", state="normal")
        else:
            self.status_bar.config(text="Detection Failed or Stopped.")

        self.is_processing = False;
        self.previous_kpts = None

    def update_frame_display(self, value):
        frame_idx = int(float(value))
        if not hasattr(self, 'all_kpts_3d') or frame_idx >= len(self.all_kpts_3d): return
        self.status_bar.config(text=f"Replay Frame: {frame_idx + 1}/{self.max_frames}")
        kpts_3d_frame = self.all_kpts_3d[frame_idx]
        prev_kpts = self.all_kpts_3d[frame_idx - 1] if frame_idx > 0 else kpts_3d_frame
        velocities = self.calculate_velocities(kpts_3d_frame, prev_kpts, self.fps)
        reba_score, risk_level, angles, scores = self.calculate_reba_score(kpts_3d_frame)
        self.reba_label.config(text=f"REBA Score: {reba_score} ({risk_level})");
        self.update_tables(angles, scores, velocities)
        self.draw_3d_skeleton(kpts_3d_frame, self.get_skeleton_definition());
        self.canvas_3d.draw()

        if hasattr(self, 'frame_2d_paths') and self.frame_2d_paths:
            for i, label in enumerate([self.label1, self.label2, self.label3]):
                if i in self.cam_ids and frame_idx < len(self.frame_2d_paths[i]):
                    try:
                        img_path = self.frame_2d_paths[i][frame_idx]
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                            self.lock.acquire();
                            label.config(image=img_tk);
                            label.image = img_tk;
                            self.lock.release()
                    except Exception as e:
                        print(f"Error updating 2D view for cam {i}: {e}")

    def update_tables(self, angles, scores, velocities):
        for part, angle_val in angles.items():
            if part in self.angle_table_rows: self.angle_table.set(self.angle_table_rows[part], 'angle',
                                                                   f"{angle_val:.1f}")
        for part, score_val in scores.items():
            if part in self.angle_table_rows: self.angle_table.set(self.angle_table_rows[part], 'score', score_val)
        for part, vel_val in velocities.items():
            if part in self.angle_table_rows: self.angle_table.set(self.angle_table_rows[part], 'velocity',
                                                                   f"{vel_val:.2f}")

    def save_csv(self):
        if not hasattr(self, 'all_kpts_3d') or not self.all_kpts_3d:
            tk.messagebox.showwarning('Fail to Export CSV', 'No 3D coordinates to be exported.');
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")],
                                                 initialfile="3D_Keypoints.csv")
        if save_path:
            flat_kpts = np.array(self.all_kpts_3d).reshape(len(self.all_kpts_3d), -1)
            pd.DataFrame(flat_kpts).to_csv(save_path, index=False)
            tk.messagebox.showinfo('Export CSV', 'Successfully exported 3D keypoints as CSV')

    def save(self):
        import glob
        path = './images/';
        plot_files = sorted(glob.glob(os.path.join(path, '????????.png')))
        if not plot_files:
            tk.messagebox.showwarning('Save Error', "Could not find any saved 3D plot images to create a video.");
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI Video", "*.avi")],
                                                 initialfile="3D_Skeleton_Video.avi")
        if not save_path: return
        img = cv2.imread(plot_files[0]);
        height, width, _ = img.shape
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
        for item in plot_files: video.write(cv2.imread(item))
        video.release();
        tk.messagebox.showinfo('Save', f'Successfully saved 3D skeleton video to {save_path}')

    def save_session(self):
        save_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Session Files", "*.json")],
            initialfile="session.json"
        )
        if not save_path:
            return

        session_data = {
            "kpts_3d_path": os.path.abspath('kpts_3d.dat'),
            "cam_ids": self.cam_ids,
            "max_frames": self.max_frames,
            "image_dir": os.path.abspath('./images/'),
            "num_keypoints": len(pose_keypoints),
            "fps": self.fps
        }

        try:
            with open(save_path, 'w') as f:
                json.dump(session_data, f, indent=4)
            self.status_bar.config(text=f"Session saved to {os.path.basename(save_path)}")
        except Exception as e:
            tk.messagebox.showerror("Save Error", f"Failed to save session: {e}")

    def load_session(self):
        load_path = filedialog.askopenfilename(
            filetypes=[("Session Files", "*.json")]
        )
        if not load_path:
            return

        try:
            with open(load_path, 'r') as f:
                session_data = json.load(f)

            required_keys = ["kpts_3d_path", "cam_ids", "max_frames", "image_dir", "num_keypoints"]
            if not all(key in session_data for key in required_keys):
                raise ValueError("Session file is missing required data.")

            kpts_3d_path = session_data["kpts_3d_path"]
            if not os.path.exists(kpts_3d_path):
                raise FileNotFoundError(f"3D keypoint file not found: {kpts_3d_path}")

            self.cam_ids = session_data["cam_ids"]
            self.max_frames = session_data["max_frames"]
            self.fps = session_data.get("fps", 30)  # Load FPS, default to 30
            num_keypoints = session_data["num_keypoints"]

            kpts_data = np.loadtxt(kpts_3d_path)
            self.all_kpts_3d = kpts_data.reshape((self.max_frames, num_keypoints, 3))

            image_dir = session_data["image_dir"]
            self.frame_2d_paths = [[] for _ in range(3)]
            if image_dir:
                for cam_id in self.cam_ids:
                    for i in range(self.max_frames):
                        path = os.path.join(image_dir, f'cam{cam_id}_frame{str(i).zfill(8)}.png')
                        self.frame_2d_paths[cam_id].append(path)

            self.status_bar.config(text=f"Session loaded from {os.path.basename(load_path)}. Click Replay.")
            self.action_menu.entryconfig("Replay", state="normal")
            self.file_menu.entryconfig("Save Session", state="normal")
            self.export_menu.entryconfig("Export 3D Skeleton Video", state="normal")
            self.export_menu.entryconfig("Export 3D Keypoints CSV", state="normal")

            self.replay()

        except Exception as e:
            tk.messagebox.showerror("Load Error", f"Failed to load session: {e}")

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
            current_frame = self.frame_slider.get()
            if current_frame < self.max_frames - 1:
                self.frame_slider.set(current_frame + 1)
                delay_ms = int(1000 / self.fps) if self.fps > 0 else 33
                self.master.after(delay_ms, self.update_replay_loop)
            else:
                self.pause_replay()

    def skip_forward(self):
        skip_frames = int(10 * self.fps) if self.fps > 0 else 300
        current_frame = self.frame_slider.get()
        new_frame = min(current_frame + skip_frames, self.max_frames - 1)
        self.frame_slider.set(new_frame)

    def skip_backward(self):
        skip_frames = int(10 * self.fps) if self.fps > 0 else 300
        current_frame = self.frame_slider.get()
        new_frame = max(current_frame - skip_frames, 0)
        self.frame_slider.set(new_frame)

    def calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2;
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        mag_v1, mag_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if mag_v1 == 0 or mag_v2 == 0: return 0
        return math.degrees(math.acos(np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0)))

    def calculate_angle_vertical(self, p1, p2):
        return self.calculate_angle(p1, p2, p2 + np.array([0, 0, 1]))

    def calculate_velocities(self, current_kpts, previous_kpts, fps=30):
        velocities = {}
        if previous_kpts is None or len(current_kpts) != len(previous_kpts): return velocities
        prev_shoulder_mid, prev_hip_mid, prev_head_mid = (previous_kpts[11] + previous_kpts[12]) / 2, (
                previous_kpts[19] + previous_kpts[20]) / 2, (previous_kpts[7] + previous_kpts[8]) / 2
        curr_shoulder_mid, curr_hip_mid, curr_head_mid = (current_kpts[11] + current_kpts[12]) / 2, (
                current_kpts[19] + current_kpts[20]) / 2, (current_kpts[7] + current_kpts[8]) / 2
        velocities['Neck'] = np.linalg.norm(curr_head_mid - prev_head_mid) * fps
        velocities['Trunk'] = np.linalg.norm(curr_shoulder_mid - prev_hip_mid) * fps
        velocities['L Upper Arm'] = np.linalg.norm(current_kpts[13] - previous_kpts[13]) * fps
        velocities['R Upper Arm'] = np.linalg.norm(current_kpts[14] - previous_kpts[14]) * fps
        velocities['L Lower Arm'] = np.linalg.norm(current_kpts[15] - previous_kpts[15]) * fps
        velocities['R Lower Arm'] = np.linalg.norm(current_kpts[16] - previous_kpts[16]) * fps
        velocities['L Wrist'] = np.linalg.norm(current_kpts[17] - previous_kpts[17]) * fps
        velocities['R Wrist'] = np.linalg.norm(current_kpts[18] - previous_kpts[18]) * fps
        velocities['L Knee'] = np.linalg.norm(current_kpts[23] - previous_kpts[23]) * fps
        velocities['R Knee'] = np.linalg.norm(current_kpts[24] - previous_kpts[24]) * fps
        return velocities

    def calculate_reba_score(self, kpts):
        angles, scores = {}, {}
        force_str = self.force_load_var.get();
        force_load_score = int(force_str.split(' ')[0]) if force_str else 0
        coupling_score, activity_score = 0, 0
        shoulder_mid, hip_mid, head_mid = (kpts[11] + kpts[12]) / 2, (kpts[19] + kpts[20]) / 2, (kpts[7] + kpts[8]) / 2

        angles["Neck"] = self.calculate_angle_vertical(head_mid, shoulder_mid);
        scores["Neck"] = 1 if angles["Neck"] <= 20 else 2
        angles["Trunk"] = self.calculate_angle_vertical(shoulder_mid, hip_mid)
        if angles["Trunk"] <= 5:
            scores["Trunk"] = 1
        elif angles["Trunk"] <= 20:
            scores["Trunk"] = 2
        elif angles["Trunk"] <= 60:
            scores["Trunk"] = 3
        else:
            scores["Trunk"] = 4

        angles["L Knee"] = 180 - self.calculate_angle(kpts[19], kpts[21], kpts[23]);
        scores["L Knee"] = 1
        if 30 <= angles["L Knee"] <= 60:
            scores["L Knee"] = 2
        elif angles["L Knee"] > 60:
            scores["L Knee"] = 3
        angles["R Knee"] = 180 - self.calculate_angle(kpts[20], kpts[22], kpts[24]);
        scores["R Knee"] = 1
        if 30 <= angles["R Knee"] <= 60:
            scores["R Knee"] = 2
        elif angles["R Knee"] > 60:
            scores["R Knee"] = 3

        leg_score = max(scores["L Knee"], scores["R Knee"])
        trunk_idx, neck_idx, leg_idx = min(scores["Trunk"] - 1, 3), min(scores["Neck"] - 1, 2), min(leg_score - 1, 3)
        scores["Score A"] = self.TABLE_A[trunk_idx][neck_idx][leg_idx] + force_load_score

        angles["L Upper Arm"] = self.calculate_angle_vertical(shoulder_mid, kpts[11]);
        angles["R Upper Arm"] = self.calculate_angle_vertical(shoulder_mid, kpts[12])
        if angles["L Upper Arm"] <= 20:
            scores["L Upper Arm"] = 1
        elif angles["L Upper Arm"] <= 45:
            scores["L Upper Arm"] = 2
        elif angles["L Upper Arm"] <= 90:
            scores["L Upper Arm"] = 3
        else:
            scores["L Upper Arm"] = 4
        if angles["R Upper Arm"] <= 20:
            scores["R Upper Arm"] = 1
        elif angles["R Upper Arm"] <= 45:
            scores["R Upper Arm"] = 2
        elif angles["R Upper Arm"] <= 90:
            scores["R Upper Arm"] = 3
        else:
            scores["R Upper Arm"] = 4
        upper_arm_score = max(scores["L Upper Arm"], scores["R Upper Arm"])

        angles["L Lower Arm"] = 180 - self.calculate_angle(kpts[11], kpts[13], kpts[15]);
        angles["R Lower Arm"] = 180 - self.calculate_angle(kpts[12], kpts[14], kpts[16])
        scores["L Lower Arm"] = 2 if angles["L Lower Arm"] < 60 or angles["L Lower Arm"] > 100 else 1
        scores["R Lower Arm"] = 2 if angles["R Lower Arm"] < 60 or angles["R Lower Arm"] > 100 else 1
        lower_arm_score = max(scores["L Lower Arm"], scores["R Lower Arm"])

        angles["L Wrist"] = 180 - self.calculate_angle(kpts[13], kpts[15], kpts[17]);
        angles["R Wrist"] = 180 - self.calculate_angle(kpts[14], kpts[16], kpts[18])
        scores["L Wrist"] = 1 if abs(angles["L Wrist"] - 180) <= 15 else 2
        scores["R Wrist"] = 1 if abs(angles["R Wrist"] - 180) <= 15 else 2
        wrist_score = max(scores["L Wrist"], scores["R Wrist"])

        upper_arm_idx, lower_arm_idx, wrist_idx = min(upper_arm_score - 1, 5), min(lower_arm_score - 1, 1), min(
            wrist_score - 1, 2)
        scores["Score B"] = self.TABLE_B[upper_arm_idx][lower_arm_idx][wrist_idx] + coupling_score

        score_a_idx, score_b_idx = min(scores["Score A"] - 1, 11), min(scores["Score B"] - 1, 11)
        table_c_score = self.TABLE_C[score_a_idx][score_b_idx]
        final_score = table_c_score + activity_score

        if final_score <= 1:
            risk = "Negligible"
        elif 2 <= final_score <= 3:
            risk = "Low"
        elif 4 <= final_score <= 7:
            risk = "Medium"
        elif 8 <= final_score <= 10:
            risk = "High"
        else:
            risk = "Very High"

        return final_score, risk, angles, scores

    def initialize_reba_tables(self):
        self.TABLE_A = np.array([[[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]], [[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]],
                                 [[3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8]],
                                 [[4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]])
        self.TABLE_B = np.array(
            [[[1, 2, 2], [1, 2, 3]], [[2, 3, 4], [3, 4, 5]], [[3, 4, 5], [4, 5, 5]], [[4, 5, 6], [5, 6, 7]],
             [[5, 6, 7], [6, 7, 8]], [[6, 7, 8], [7, 8, 9]]])
        self.TABLE_C = np.array([[1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7], [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
                                 [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8], [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
                                 [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9], [5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10],
                                 [6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10, 10], [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
                                 [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
                                 [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
                                 [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
                                 [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12]])

    def run(self):
        if not os.path.exists('./images'): os.makedirs('./images')
        self.master.mainloop()


if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    app.run()