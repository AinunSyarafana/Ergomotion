import cv2
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
from tkinter import filedialog
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

plt.style.use('seaborn-v0_8')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [1280, 720]

pose_keypoints = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32
]


class OptionDialog(tk.Toplevel):
    def __init__(self, master, options):
        super().__init__(master)
        self.title("Browse File")
        self.geometry("400x100")
        self.resizable(width=False, height=False)
        window_width = self.winfo_reqwidth()
        window_height = self.winfo_reqheight()
        x = (1200 - window_width) // 2
        y = (674 - window_height) // 2
        self.geometry(f"+{x}+{y}")
        self.selected_option = tk.StringVar()
        label = tk.Label(self, text="Select Video From:")
        label.pack(pady=10)
        for option in options:
            button = tk.Button(self, text=option, width=15, height=2,
                               command=lambda option=option: self.on_option_selected(option))
            button.pack(side=tk.LEFT, padx=5)

    def on_option_selected(self, option):
        self.selected_option.set(option)
        self.destroy()


class Application:
    def __init__(self, master):
        self.master = master
        self.master.title("Interactive 3D Motion Reconstruction with REBA")
        self.master.configure(bg="#d0bdf4")
        self.master.resizable(width=False, height=False)

        w, h = 1000, 850
        ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
        x, y = (ws / 2) - (w / 2), (hs / 2) - (h / 2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))

        style = ThemedStyle(root)
        style.set_theme("yaru")
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#d0bdf4")

        self.setup_menus()
        self.setup_widgets()
        self.initialize_reba_tables()

        self.lock = threading.Lock()
        self.signal = threading.Event()
        self.cap0 = self.cap1 = self.cap2 = None
        self.max_frames = 0
        self.is_processing = False
        self.cam_ids = []
        self.all_kpts_3d = []
        self.frame_2d_paths = [[] for _ in range(3)]

    def setup_menus(self):
        self.menubar = tk.Menu(self.master)
        self.master.config(menu=self.menubar)
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Reload Application", command=self.reload_app)
        file_menu.add_command(label="Exit", command=self.exit_app)
        export_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Export", menu=export_menu)
        export_menu.add_command(label="Export 3D Skeleton Video", command=self.save)
        export_menu.add_separator()
        export_menu.add_command(label="Export 3D Keypoints CSV", command=self.save_csv)

    def setup_widgets(self):
        button_frame = ttk.Frame(self.master, style="Custom.TFrame")
        button_frame.pack(pady=5)
        ttk.Button(button_frame, text="UPLOAD", command=self.choose_file, cursor="hand2").pack(side='left', padx=10)
        ttk.Button(button_frame, text="DETECT", command=self.detect, cursor="hand2").pack(side='left', padx=10)
        self.replay_button = ttk.Button(button_frame, text="REPLAY", command=self.replay, cursor="hand2",
                                        state='disabled')
        self.replay_button.pack(side='left', padx=10)
        self.save_button = ttk.Button(button_frame, text="SAVE VIDEO", command=self.save, cursor="hand2",
                                      state='disabled')
        self.save_button.pack(side='left', padx=10)
        self.csv_button = ttk.Button(button_frame, text="EXPORT TO CSV", command=self.save_csv, cursor="hand2",
                                     state='disabled')
        self.csv_button.pack(side='right', padx=10)

        reba_frame = ttk.Frame(self.master, style="Custom.TFrame")
        reba_frame.pack(pady=5)
        self.reba_label = tk.Label(reba_frame, text="REBA Score: -", font=("Helvetica", 16, "bold"),
                                   background="#d0bdf4")
        self.reba_label.pack()

        self.fig_3d = plt.figure(figsize=(5, 4))
        self.fig_3d.patch.set_facecolor('#d0bdf4')
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.fig_3d.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.master)
        self.canvas_3d.get_tk_widget().pack(pady=5)

        video_frame = ttk.Frame(self.master, style="Custom.TFrame")
        video_frame.pack(pady=5)
        self.label1 = tk.Label(video_frame, background="#d0bdf4");
        self.label1.pack(side=tk.LEFT, padx=5)
        self.label2 = tk.Label(video_frame, background="#d0bdf4");
        self.label2.pack(side=tk.LEFT, padx=5)
        self.label3 = tk.Label(video_frame, background="#d0bdf4");
        self.label3.pack(side=tk.LEFT, padx=5)

        self.slider_frame = ttk.Frame(self.master, style="Custom.TFrame")
        self.slider_frame.pack(pady=10, fill='x', padx=20)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.slider_frame, orient=tk.HORIZONTAL, length=800, mode='determinate',
                                            variable=self.progress_var)
        self.frame_slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                      command=self.update_frame_display)
        self.frame_label = tk.Label(self.slider_frame, text="Ready", background="#d0bdf4")
        self.progress_bar.pack(fill='x')
        self.frame_label.pack()

    def exit_app(self):
        self.master.quit()

    def reload_app(self):
        os.execl(sys.executable, sys.executable, *sys.argv)

    def choose_file(self):
        for btn in [self.csv_button, self.save_button, self.replay_button]: btn.configure(state='disabled')
        self.frame_2d_paths = [[] for _ in range(3)]
        dialog = OptionDialog(self.master, ["Cam0 (Front View)", "Cam1 (Side View)", "Cam2 (Third View)"])
        self.master.wait_window(dialog)
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
        tk.messagebox.showinfo(title=input_source, message=f'{file_path} has been uploaded.')

    def detect(self):
        if self.is_processing: tk.messagebox.showinfo('Processing', 'Detection is already running.'); return
        if len([c for c in [self.cap0, self.cap1, self.cap2] if c is not None]) < 2: tk.messagebox.showwarning(
            'No Input', 'Please upload at least 2 video files to detect.'); return

        self.is_processing = True
        self.progress_bar.pack(fill='x');
        self.frame_slider.pack_forget()
        for btn in [self.replay_button, self.save_button, self.csv_button]: btn.configure(state='disabled')
        self.frame_label.config(text="Processing...");
        self.progress_var.set(0)
        threading.Thread(target=self.play_video).start()

    def replay(self):
        if self.max_frames > 0:
            self.progress_bar.pack_forget();
            self.frame_slider.pack(fill='x')
            self.frame_slider.configure(state='normal', to=self.max_frames)
            self.frame_slider.set(0);
            self.update_frame_display(0)

    def play_video(self):
        cam_info = []
        if self.cap0: cam_info.append((0, self.cap0, self.label1))
        if self.cap1: cam_info.append((1, self.cap1, self.label2))
        if self.cap2: cam_info.append((2, self.cap2, self.label3))

        self.cam_ids = [info[0] for info in cam_info];
        caps = [info[1] for info in cam_info]
        P_list = [get_projection_matrix(cam_id) for cam_id, _, _ in cam_info]
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        max_frames = min(frame_counts) if frame_counts else 0
        self.max_frames = max_frames - 1

        poses = {id: mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) for id, _, _ in cam_info}
        kpts_2d = {cam_id: [] for cam_id, _, _ in cam_info};
        self.all_kpts_3d = []
        skeleton_connections = self.get_skeleton_definition()
        last_good_kpts = None

        for index in range(max_frames):
            rets, frames = zip(*[cap.read() for cap in caps])
            if not all(rets): break

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

            reba_score, risk_level = self.calculate_reba_score(frame_p3ds_centered)
            self.reba_label.config(text=f"REBA Score: {reba_score} ({risk_level})")

            self.draw_3d_skeleton(frame_p3ds_centered, skeleton_connections)
            self.fig_3d.savefig(f'./images/{str(index).zfill(8)}.png', bbox_inches='tight', pad_inches=0, dpi=100)
            self.canvas_3d.draw()
            self.progress_var.set((index / max_frames) * 100)
            self.frame_label.config(text=f"Processing Frame: {index + 1}/{max_frames}")

        self.finalize_processing(kpts_2d)

    def draw_3d_skeleton(self, kpts, skeleton_connections):
        self.ax_3d.clear()
        self.ax_3d.set_facecolor('#d0bdf4')
        self.ax_3d.grid(color='white', linestyle='-', linewidth=0.5)
        self.ax_3d.tick_params(axis='x', colors='white');
        self.ax_3d.tick_params(axis='y', colors='white');
        self.ax_3d.tick_params(axis='z', colors='white')
        self.ax_3d.xaxis.label.set_color('white');
        self.ax_3d.yaxis.label.set_color('white');
        self.ax_3d.zaxis.label.set_color('white')
        self.ax_3d.view_init(elev=90, azim=-180)

        for p1_idx, p2_idx in skeleton_connections:
            if not np.all(kpts[p1_idx] == -1) and not np.all(kpts[p2_idx] == -1):
                self.ax_3d.plot([kpts[p1_idx, 0], kpts[p2_idx, 0]], [kpts[p1_idx, 1], kpts[p2_idx, 1]],
                                [kpts[p1_idx, 2], kpts[p2_idx, 2]], linewidth=2, c='red')

        if not np.all(kpts[11:13] == -1) and not np.all(kpts[7:9] == -1):
            shoulder_midpoint = (kpts[11] + kpts[12]) / 2.0;
            head_center = (kpts[7] + kpts[8]) / 2.0
            neck_length = np.linalg.norm(head_center - shoulder_midpoint)
            if neck_length < 5.0:
                self.ax_3d.plot([head_center[0], shoulder_midpoint[0]], [head_center[1], shoulder_midpoint[1]],
                                [head_center[2], shoulder_midpoint[2]], linewidth=2, c='red')

        valid_kpts = kpts[~np.all(kpts == -1, axis=1)]
        if valid_kpts.shape[0] > 0:
            max_range = np.array([valid_kpts[:, i].max() - valid_kpts[:, i].min() for i in range(3)]).max() / 2.0
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
        torso = [[11, 12], [12, 18], [18, 17], [17, 11]]
        head = [[7, 3], [3, 2], [2, 1], [1, 0], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]]
        arm_l = [[11, 13], [13, 15]];
        arm_r = [[12, 14], [14, 16]]
        leg_l = [[17, 19], [19, 21]];
        leg_r = [[18, 20], [20, 22]]
        foot_l = [[21, 23], [23, 25], [21, 25]];
        foot_r = [[22, 24], [24, 26], [22, 26]]
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
        frame_resized = cv2.resize(frame, (300, 200))
        frame_2d_path = f'./images/cam{cam_id}_frame{str(index).zfill(8)}.png'
        cv2.imwrite(frame_2d_path, frame_resized)
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

    def finalize_processing(self, kpts_2d):
        for cap in [self.cap0, self.cap1, self.cap2]:
            if cap: cap.release()
        self.cap0 = self.cap1 = self.cap2 = None
        for cam_id in self.cam_ids: write_keypoints_to_disk(f'kpts_cam{cam_id}.dat', np.array(kpts_2d[cam_id]))
        write_keypoints_to_disk('kpts_3d.dat', np.array(self.all_kpts_3d))

        if self.max_frames > 0:
            self.progress_var.set(100);
            self.frame_label.config(text="Processing Complete. Click REPLAY to interact.")
            for btn in [self.csv_button, self.save_button, self.replay_button]: btn.configure(state='normal')
        else:
            self.frame_label.config(text="Detection Failed.")
        self.is_processing = False;
        self.signal.set()

    def update_frame_display(self, value):
        frame_idx = int(float(value))
        if frame_idx >= len(self.all_kpts_3d): return
        self.frame_label.config(text=f"Replay Frame: {frame_idx}/{self.max_frames}")

        kpts_3d_frame = self.all_kpts_3d[frame_idx]
        reba_score, risk_level = self.calculate_reba_score(kpts_3d_frame)
        self.reba_label.config(text=f"REBA Score: {reba_score} ({risk_level})")

        self.draw_3d_skeleton(kpts_3d_frame, self.get_skeleton_definition())
        self.canvas_3d.draw()

        for i, label in enumerate([self.label1, self.label2, self.label3]):
            if i in self.cam_ids and frame_idx < len(self.frame_2d_paths[i]):
                try:
                    img = cv2.imread(self.frame_2d_paths[i][frame_idx])
                    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    self.lock.acquire();
                    label.config(image=img_tk);
                    label.image = img_tk;
                    self.lock.release()
                except Exception as e:
                    print(f"Error updating 2D view for cam {i}: {e}")

    def save_csv(self):
        if not self.all_kpts_3d: tk.messagebox.showwarning('Fail to Export CSV',
                                                           'No 3D coordinates to be exported.'); return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")],
                                                 initialfile="3D_Keypoints.csv")
        if save_path:
            pd.DataFrame(np.array(self.all_kpts_3d).reshape(len(self.all_kpts_3d), -1)).to_csv(save_path, index=False)
            tk.messagebox.showinfo('Export CSV', 'Successfully exported 3D keypoints as CSV')

    def save(self):
        import glob
        path = './images/'
        plot_files = sorted(glob.glob(os.path.join(path, '????????.png')))
        if not plot_files: tk.messagebox.showwarning('Save Error',
                                                     "Could not find 3D plot images to create video."); return
        save_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI Video", "*.avi")],
                                                 initialfile="3D_Skeleton_Video.avi")
        if not save_path: return
        img = cv2.imread(plot_files[0]);
        height, width, _ = img.shape
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
        for item in plot_files: video.write(cv2.imread(item))
        video.release();
        tk.messagebox.showinfo('Save', f'Successfully saved 3D skeleton video to {save_path}')

    def generate_expert_report(self):
        pass

    def calculate_angle(self, p1, p2, p3):
        v1 = p1 - p2;
        v2 = p3 - p2
        dot_product = np.dot(v1, v2)
        mag_v1 = np.linalg.norm(v1);
        mag_v2 = np.linalg.norm(v2)
        if mag_v1 == 0 or mag_v2 == 0: return 0
        angle = math.acos(np.clip(dot_product / (mag_v1 * mag_v2), -1.0, 1.0))
        return math.degrees(angle)

    def calculate_angle_vertical(self, p1, p2):
        v = p1 - p2
        v_vertical = np.array([0, 0, 1])
        return self.calculate_angle(p1, p2, p2 + v_vertical)

    def calculate_reba_score(self, kpts):
        force_load_score, coupling_score, activity_score = 0, 0, 0
        shoulder_mid, hip_mid, head_mid = (kpts[11] + kpts[12]) / 2, (kpts[17] + kpts[18]) / 2, (kpts[7] + kpts[8]) / 2

        neck_angle = self.calculate_angle_vertical(head_mid, shoulder_mid)
        neck_score = 1 if neck_angle <= 20 else 2

        trunk_angle = self.calculate_angle_vertical(shoulder_mid, hip_mid)
        if trunk_angle <= 5:
            trunk_score = 1
        elif trunk_angle <= 20:
            trunk_score = 2
        elif trunk_angle <= 60:
            trunk_score = 3
        else:
            trunk_score = 4

        leg_angle = min(self.calculate_angle(kpts[17], kpts[19], kpts[21]),
                        self.calculate_angle(kpts[18], kpts[20], kpts[22]))
        leg_score = 1
        if 120 <= leg_angle <= 150:
            leg_score += 1
        elif leg_angle < 120:
            leg_score += 2

        trunk_idx, neck_idx, leg_idx = min(trunk_score - 1, 4), min(neck_score - 1, 2), min(leg_score - 1, 3)
        score_a = self.TABLE_A[trunk_idx][neck_idx][leg_idx] + force_load_score

        upper_arm_angle = self.calculate_angle_vertical(kpts[11], kpts[13])
        if upper_arm_angle <= 20:
            upper_arm_score = 1
        elif upper_arm_angle <= 45:
            upper_arm_score = 2
        elif upper_arm_angle <= 90:
            upper_arm_score = 3
        else:
            upper_arm_score = 4

        lower_arm_angle = self.calculate_angle(kpts[11], kpts[13], kpts[15])
        lower_arm_score = 1 if 60 <= lower_arm_angle <= 100 else 2

        wrist_angle = self.calculate_angle(kpts[13], kpts[15], kpts[15] + (kpts[15] - kpts[13]))
        wrist_score = 1 if abs(wrist_angle - 180) <= 15 else 2

        upper_arm_idx, lower_arm_idx, wrist_idx = min(upper_arm_score - 1, 5), min(lower_arm_score - 1, 1), min(
            wrist_score - 1, 2)
        score_b = self.TABLE_B[upper_arm_idx][lower_arm_idx][wrist_idx] + coupling_score

        score_a_idx, score_b_idx = min(score_a - 1, 11), min(score_b - 1, 11)
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

        return final_score, risk

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