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

plt.style.use('seaborn-v0_8') 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [1280,720]

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
        self.master.title("Interactive 3D Motion Reconstruction")
        self.master.configure(bg="#d0bdf4")
        self.master.resizable(width=False, height=False)       

        w, h = 1000, 800
        ws, hs = root.winfo_screenwidth(), root.winfo_screenheight()
        x, y = (ws/2) - (w/2), (hs/2) - (h/2)
        root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        style = ThemedStyle(root)
        style.set_theme("yaru")
        style = ttk.Style()
        style.configure("Custom.TFrame", background="#d0bdf4")
        
        self.setup_menus()
        self.setup_widgets()

        self.lock = threading.Lock()
        self.signal = threading.Event()
        self.cap0 = self.cap1 = self.cap2 = None
        self.max_frames = 0
        self.is_processing = False
        self.cam_ids = []
        self.all_kpts_3d = []
        self.frame_2d_paths = [[] for _ in range(3)] # Stores paths to 2D images for replay

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
        export_menu.add_command(label="Generate Expert Report", command=self.generate_expert_report)
    
    def setup_widgets(self):
        button_frame = ttk.Frame(self.master, style="Custom.TFrame")
        button_frame.pack(pady=10)
       
        ttk.Button(button_frame, text="UPLOAD", command=self.choose_file, cursor="hand2").pack(side='left', padx=10)
        ttk.Button(button_frame, text="DETECT", command=self.detect, cursor="hand2").pack(side='left', padx=10)
        self.replay_button = ttk.Button(button_frame, text="REPLAY", command=self.replay, cursor="hand2", state='disabled')
        self.replay_button.pack(side='left', padx=10)
        self.save_button = ttk.Button(button_frame, text="SAVE VIDEO", command=self.save, cursor="hand2", state='disabled')
        self.save_button.pack(side='left', padx=10)
        self.csv_button = ttk.Button(button_frame, text="EXPORT TO CSV", command=self.save_csv, cursor="hand2", state='disabled')
        self.csv_button.pack(side='right', padx=10)

        self.fig_3d = plt.figure(figsize=(5, 4))
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self.fig_3d.subplots_adjust(left=0, right=1, bottom=0, top=1)

        self.canvas_3d = FigureCanvasTkAgg(self.fig_3d, master=self.master)
        self.canvas_3d.get_tk_widget().pack(pady=5)

        video_frame = ttk.Frame(self.master, style="Custom.TFrame")
        video_frame.pack(pady=5)
        self.label1 = tk.Label(video_frame, background="#d0bdf4")
        self.label1.pack(side=tk.LEFT, padx=5)
        self.label2 = tk.Label(video_frame, background="#d0bdf4")
        self.label2.pack(side=tk.LEFT, padx=5)
        self.label3 = tk.Label(video_frame, background="#d0bdf4")
        self.label3.pack(side=tk.LEFT, padx=5)
        
        self.slider_frame = ttk.Frame(self.master, style="Custom.TFrame")
        self.slider_frame.pack(pady=10, fill='x', padx=20)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.slider_frame, orient=tk.HORIZONTAL, length=800, mode='determinate', variable=self.progress_var)
        self.frame_slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_frame_display)
        self.frame_label = tk.Label(self.slider_frame, text="Ready", background="#d0bdf4")
        
        self.progress_bar.pack(fill='x')
        self.frame_label.pack()

    def exit_app(self):
        self.master.quit()

    def reload_app(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)
        
    def choose_file(self):     
        self.csv_button.configure(state='disabled')
        self.save_button.configure(state='disabled')
        self.replay_button.configure(state='disabled')
        self.frame_2d_paths = [[] for _ in range(3)] # Reset paths
        
        options = ["Cam0 (Front View)", "Cam1 (Side View)", "Cam2 (Third View)"]
        dialog = OptionDialog(self.master, options)
        self.master.wait_window(dialog)
        input_source = dialog.selected_option.get()        
        
        if not input_source: return
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi;*.mp4;*.mov")])
        if not file_path: return

        if "Cam0" in input_source: self.cap0 = cv2.VideoCapture(file_path)
        elif "Cam1" in input_source: self.cap1 = cv2.VideoCapture(file_path)
        elif "Cam2" in input_source: self.cap2 = cv2.VideoCapture(file_path)
        tk.messagebox.showinfo(title=input_source, message=f'{file_path} has been uploaded.')
    
    def detect(self):
        if self.is_processing:
            tk.messagebox.showinfo('Processing', 'Detection is already running.')
            return
        if len([c for c in [self.cap0, self.cap1, self.cap2] if c is not None]) < 2:
            tk.messagebox.showwarning('No Input', 'Please upload at least 2 video files to detect.')
            return

        self.is_processing = True
        self.progress_bar.pack(fill='x')
        self.frame_slider.pack_forget()
        self.replay_button.configure(state='disabled')
        self.save_button.configure(state='disabled')
        self.csv_button.configure(state='disabled')
        self.frame_label.config(text="Processing...")
        self.progress_var.set(0)
        
        threading.Thread(target=self.play_video).start()   
            
    def replay(self):
        if self.max_frames > 0:
            self.progress_bar.pack_forget()
            self.frame_slider.pack(fill='x')
            self.frame_slider.configure(state='normal', to=self.max_frames)
            self.frame_slider.set(0)
            self.update_frame_display(0)

    def play_video(self):    
        cam_info = []
        if self.cap0: cam_info.append((0, self.cap0, self.label1))
        if self.cap1: cam_info.append((1, self.cap1, self.label2))
        if self.cap2: cam_info.append((2, self.cap2, self.label3))
        
        self.cam_ids = [info[0] for info in cam_info]
        caps = [info[1] for info in cam_info]
        
        P_list = [get_projection_matrix(cam_id) for cam_id, _, _ in cam_info]
        
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        max_frames = min(frame_counts) if frame_counts else 0
        self.max_frames = max_frames - 1 

        poses = {id: mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) for id, _, _ in cam_info}
        
        kpts_2d = {cam_id: [] for cam_id, _, _ in cam_info}
        self.all_kpts_3d = []
        
        body, colors = self.get_skeleton_definition()
        
        for index in range(max_frames):
            rets, frames = zip(*[cap.read() for cap in caps])
            if not all(rets): break

            current_frame_2d_kpts = []
            for i, (cam_id, _, label) in enumerate(cam_info):
                frame_rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
                result = poses[cam_id].process(frame_rgb)
                
                frame_keypoints = self.extract_keypoints(result, frames[i].shape)
                kpts_2d[cam_id].append(frame_keypoints) 
                current_frame_2d_kpts.append(frame_keypoints)
                
                self.display_2d_frame(frames[i], result, label, index, cam_id)

            frame_p3ds = self.triangulate_points(P_list, current_frame_2d_kpts)
            self.all_kpts_3d.append(frame_p3ds)
            
            self.draw_3d_skeleton(frame_p3ds, body, colors)
            self.fig_3d.savefig(f'./images/{str(index).zfill(8)}.png', bbox_inches='tight', pad_inches=0, dpi=100)
            self.canvas_3d.draw()

            self.progress_var.set((index / max_frames) * 100)
            self.frame_label.config(text=f"Processing Frame: {index+1}/{max_frames}")
            
        self.finalize_processing(kpts_2d)

    def draw_3d_skeleton(self, kpts, body, colors):
        self.ax_3d.clear()
        self.ax_3d.view_init(elev=5, azim=-90)

        for bodypart, part_color in zip(body, colors):
            for p1_idx, p2_idx in bodypart:
                if not np.all(kpts[p1_idx] == -1) and not np.all(kpts[p2_idx] == -1):
                    self.ax_3d.plot(
                        [kpts[p1_idx, 0], kpts[p2_idx, 0]],
                        [kpts[p1_idx, 1], kpts[p2_idx, 1]],
                        [kpts[p1_idx, 2], kpts[p2_idx, 2]],
                        linewidth=2, c=part_color
                    )
        
        # --- CHANGE 1: Neck connects from head center to shoulder center ---
        left_shoulder, right_shoulder = kpts[11], kpts[12]
        left_ear, right_ear = kpts[7], kpts[8]

        # Check if all necessary points were detected
        if not np.all(left_shoulder == -1) and not np.all(right_shoulder == -1) and \
           not np.all(left_ear == -1) and not np.all(right_ear == -1):
            
            shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
            head_center = (left_ear + right_ear) / 2.0 # Midpoint between ears
            
            self.ax_3d.plot(
                [head_center[0], shoulder_midpoint[0]],
                [head_center[1], shoulder_midpoint[1]],
                [head_center[2], shoulder_midpoint[2]],
                linewidth=2, c='cyan'
            )
        
        valid_kpts = kpts[~np.all(kpts == -1, axis=1)]
        if valid_kpts.shape[0] > 0:
            x_min, x_max = valid_kpts[:,0].min(), valid_kpts[:,0].max()
            y_min, y_max = valid_kpts[:,1].min(), valid_kpts[:,1].max()
            z_min, z_max = valid_kpts[:,2].min(), valid_kpts[:,2].max()
            mid_x, mid_y, mid_z = (x_max + x_min)/2, (y_max + y_min)/2, (z_max + z_min)/2
            max_range = np.array([x_max-x_min, y_max-y_min, z_max-z_min]).max() / 2.0
            self.ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)
        else:
            self.ax_3d.set_xlim3d(-10, 10); self.ax_3d.set_ylim3d(-10, 10); self.ax_3d.set_zlim3d(-10, 10)

        self.ax_3d.set_xticks([]); self.ax_3d.set_yticks([]); self.ax_3d.set_zticks([])
        self.ax_3d.set_xlabel('x'); self.ax_3d.set_ylabel('y'); self.ax_3d.set_zlabel('z')

    def get_skeleton_definition(self):
        torso = [[11, 12], [12, 18], [18, 17], [17, 11]]
        head = [[7, 3], [3, 2], [2, 1], [1, 0], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]]
        arm_l = [[11, 13], [13, 15]]; arm_r = [[12, 14], [14, 16]]
        leg_l = [[17, 19], [19, 21]]; leg_r = [[18, 20], [20, 22]]
        foot_l = [[21, 23], [23, 25], [21, 25]]; foot_r = [[22, 24], [24, 26], [22, 26]]
        body = [torso, head, arm_l, arm_r, leg_l, leg_r, foot_l, foot_r]
        colors = ['red', 'yellow', 'blue', 'green', 'black', 'orange', 'purple', 'brown']
        return body, colors

    def extract_keypoints(self, result, shape):
        if result.pose_landmarks:
            all_landmarks = result.pose_landmarks.landmark
            return [[int(round(all_landmarks[p_idx].x * shape[1])), int(round(all_landmarks[p_idx].y * shape[0]))] for p_idx in pose_keypoints]
        return [[-1, -1]] * len(pose_keypoints)

    def display_2d_frame(self, frame, result, label, index, cam_id):
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4))
        frame_resized = cv2.resize(frame, (300, 200))

        # Save the frame for replay and store its path
        frame_2d_path = f'./images/cam{cam_id}_frame{str(index).zfill(8)}.png'
        cv2.imwrite(frame_2d_path, frame_resized)
        self.frame_2d_paths[cam_id].append(frame_2d_path)

        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)))
        
        self.lock.acquire()
        label.config(image=img_tk); label.image = img_tk
        self.lock.release()

    def triangulate_points(self, P_list, kpts_2d):
        kpts_3d = [DLT_multi(P_list, [cam_kpts[kpt_idx] for cam_kpts in kpts_2d]) for kpt_idx in range(len(pose_keypoints))]
        return np.array(kpts_3d).reshape((len(pose_keypoints), 3))

    def finalize_processing(self, kpts_2d):
        for cap in [self.cap0, self.cap1, self.cap2]:
            if cap: cap.release()
        self.cap0 = self.cap1 = self.cap2 = None
        
        for cam_id in self.cam_ids:
            write_keypoints_to_disk(f'kpts_cam{cam_id}.dat', np.array(kpts_2d[cam_id]))
        write_keypoints_to_disk('kpts_3d.dat', np.array(self.all_kpts_3d))
        
        if self.max_frames > 0:
            self.progress_var.set(100)
            self.frame_label.config(text="Processing Complete. Click REPLAY to interact.")
            self.csv_button.configure(state='normal')
            self.save_button.configure(state='normal')
            self.replay_button.configure(state='normal') 
        else:
            self.frame_label.config(text="Detection Failed.")

        self.is_processing = False
        self.signal.set()

    def update_frame_display(self, value):
        frame_idx = int(float(value))
        if frame_idx >= len(self.all_kpts_3d): return

        self.frame_label.config(text=f"Replay Frame: {frame_idx}/{self.max_frames}")
        
        # Redraw 3D skeleton from data for interactivity
        body, colors = self.get_skeleton_definition()
        kpts_3d_frame = self.all_kpts_3d[frame_idx]
        self.draw_3d_skeleton(kpts_3d_frame, body, colors)
        self.canvas_3d.draw()

        # --- FIX 2: Synchronize 2D views during replay ---
        cam_labels = [self.label1, self.label2, self.label3]
        for i, label in enumerate(cam_labels):
            if i in self.cam_ids and frame_idx < len(self.frame_2d_paths[i]):
                try:
                    frame_path = self.frame_2d_paths[i][frame_idx]
                    img = cv2.imread(frame_path)
                    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                    
                    self.lock.acquire()
                    label.config(image=img_tk)
                    label.image = img_tk
                    self.lock.release()
                except Exception as e:
                    print(f"Error updating 2D view for cam {i}: {e}")

    def save_csv(self):               
        if not self.all_kpts_3d:
            tk.messagebox.showwarning('Fail to Export CSV', 'No 3D coordinates to be exported.')
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], initialfile="3D_Keypoints.csv")
        if save_path:
            reshaped_data = np.array(self.all_kpts_3d).reshape(len(self.all_kpts_3d), -1)
            pd.DataFrame(reshaped_data).to_csv(save_path, index=False)
            tk.messagebox.showinfo('Export CSV', 'Successfully exported 3D keypoints as CSV')

    def save(self):
        import glob
        path = './images/'
        image_files = sorted(glob.glob(os.path.join(path, 'cam0_*.png'))) # Save based on one camera's frames
        if not image_files:
            tk.messagebox.showwarning('Save Error', "Please run detection first to generate frames before saving.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI Video", "*.avi")], initialfile="3D_Skeleton_Video.avi")
        if not save_path: return 

        # Create video from the separate 3D plot images saved to disk
        plot_files = sorted(glob.glob(os.path.join(path, '????????.png')))
        if not plot_files:
            tk.messagebox.showwarning('Save Error', "Could not find 3D plot images to create video.")
            return
            
        img = cv2.imread(plot_files[0])
        height, width, _ = img.shape
        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))
        
        for item in plot_files:
            video.write(cv2.imread(item))
        video.release()
        tk.messagebox.showinfo('Save', f'Successfully saved 3D skeleton video to {save_path}')

    def generate_expert_report(self): pass
    
    def run(self):
        if not os.path.exists('./images'): os.makedirs('./images')
        self.master.mainloop()

if __name__ == '__main__':
    root = tk.Tk()
    app = Application(root)
    app.run()