import cv2
import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedStyle
from PIL import Image, ImageTk
from tkinter import filedialog
import threading
import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
import os 
# DLT is replaced by DLT_multi in utils, so we import DLT_multi
from utils import DLT_multi, get_projection_matrix, write_keypoints_to_disk 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# Re-import DLT_multi from utils for clarity
from utils import DLT_multi
import pandas as pd
# Using a specific seaborn version style. Ensure 'seaborn-v0_8' is available, or change to 'ggplot'.
plt.style.use('seaborn-v0_8') 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [1280,720]

# Keypoints list includes full head landmarks
pose_keypoints = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Head landmarks (11 points)
    11, 12, 13, 14, 15, 16,            # Body and arms (6 points)
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32  # Legs and feet (10 points)
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
        self.master.title("Motion Detection with 3D Reconstruction App")
        self.master.configure(bg="#d0bdf4")
        self.master.resizable(width=False, height=False)       

        w = 1000 
        h = 750 

        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)

        root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        style = ThemedStyle(root)
        style.set_theme("yaru")

        style = ttk.Style()
        style.configure("Custom.TFrame", background="#d0bdf4")
        
        # --- Menu Bar Setup ---
        self.menubar = tk.Menu(master)
        master.config(menu=self.menubar)
        
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Reload Application", command=self.reload_app) 
        self.file_menu.add_command(label="Exit", command=self.exit_app) 
        
        # --- Export Menu ---
        self.export_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Export", menu=self.export_menu)
        
        self.export_menu.add_command(label="Export 3D Skeleton Video", command=self.save) 
        self.export_menu.add_command(label="Export Cam0 Video", command=lambda: self.export_cam_video(0)) 
        self.export_menu.add_command(label="Export Cam1 Video", command=lambda: self.export_cam_video(1)) 
        self.export_menu.add_command(label="Export Cam2 Video", command=lambda: self.export_cam_video(2))
        
        # --- NEW Export Options ---
        self.export_menu.add_separator()
        self.export_menu.add_command(label="Export 3D Keypoints CSV", command=self.save_csv) 
        self.export_menu.add_command(label="Generate Expert Report", command=self.generate_expert_report)
        # ---------------------------

        # 1. Button Frame (Top-most)
        button_frame = ttk.Frame(master, style="Custom.TFrame")
        button_frame.pack(pady=10)
       
        choose_file_button = ttk.Button(button_frame, text="UPLOAD", command=self.choose_file, cursor="hand2")
        choose_file_button.pack(side='left', padx=10)

        detect_button = ttk.Button(button_frame, text="DETECT", command=self.detect, cursor="hand2")
        detect_button.pack(side='left', padx=10)

        self.replay_button = ttk.Button(button_frame, text="REPLAY", command=self.replay, cursor="hand2")
        self.replay_button.pack(side='left', padx=10)
        self.replay_button.configure(state='disabled')


        self.save_button = ttk.Button(button_frame, text="SAVE VIDEO", command=self.save, cursor="hand2")
        self.save_button.pack(side='left', padx=10)
        self.save_button.configure(state='disabled')


        self.csv_button = ttk.Button(button_frame, text="EXPORT TO CSV", command=self.save_csv, cursor="hand2")
        self.csv_button.pack(side='right', padx=10)
        self.csv_button.configure(state='disabled')

        button_frame.pack_configure(anchor='center')
        
        # 2. 3D Plot Label (Next: Top center display)
        self.label0 = tk.Label(master, background="#d0bdf4")
        self.label0.pack(pady=5)

        # 3. Frame for 2D Video Views (Next: Below 3D Plot)
        video_frame = ttk.Frame(master, style="Custom.TFrame")
        video_frame.pack(pady=5)

        # Cam0 (Left)
        self.label1 = tk.Label(video_frame, background="#d0bdf4")
        self.label1.pack(side=tk.LEFT, padx=5)

        # Cam1 (Middle)
        self.label2 = tk.Label(video_frame, background="#d0bdf4")
        self.label2.pack(side=tk.LEFT, padx=5)

        # Cam2 (Right)
        self.label3 = tk.Label(video_frame, background="#d0bdf4")
        self.label3.pack(side=tk.LEFT, padx=5)
        
        # 4. Progress/Slider Frame (Bottom)
        self.slider_frame = ttk.Frame(master, style="Custom.TFrame")
        self.slider_frame.pack(pady=10)
        
        # Progress Bar (for DETECT status)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.slider_frame, orient=tk.HORIZONTAL, length=800, mode='determinate', variable=self.progress_var)
        self.progress_bar.pack(side=tk.TOP, padx=10)
        
        # Replay Slider (overwrites progress bar space, only visible after DETECT)
        self.frame_slider = ttk.Scale(self.slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.update_frame_display, length=800)
        self.frame_slider.pack(side=tk.TOP, padx=10)
        self.frame_slider.lower() # Hide initially
        self.frame_slider.configure(state='disabled')
        
        self.frame_label = tk.Label(self.slider_frame, text="Ready", background="#d0bdf4")
        self.frame_label.pack(side=tk.TOP)

        # Hidden file path labels (Bottom-most)
        self.labelfilepath = tk.Label(master, background="#d0bdf4", foreground="#d0bdf4")
        self.labelfilepath.pack()

        self.labelfilepath2 = tk.Label(master, background="#d0bdf4", foreground="#d0bdf4")
        self.labelfilepath2.pack()
        
        self.labelfilepath3 = tk.Label(master, background="#d0bdf4", foreground="#d0bdf4")
        self.labelfilepath3.pack()

        # synchronize cap0, cap1, and cap2
        self.lock = threading.Lock()
        self.signal = threading.Event()

        self.cap0 = None
        self.cap1 = None
        self.cap2 = None
        
        self.max_frames = 0
        self.frame_paths = [] # 3D frame paths
        self.frame_2d_paths = [[] for _ in range(3)] # 2D frame paths per cam
        self.is_processing = False
        self.cam_ids = []
    
    # --- Menu Bar Methods ---
    def exit_app(self):
        """Closes the application."""
        self.master.quit()

    def reload_app(self):
        """Restarts the application by executing the script again."""
        python = sys.executable
        os.execl(python, python, *sys.argv)
        
    def generate_expert_report(self):
        """Placeholder for generating a detailed expert analysis report."""
        if not os.path.exists('kpts_3d.dat') or os.path.getsize('kpts_3d.dat') == 0:
            tk.messagebox.showwarning('Report Generation Failed', 'No 3D keypoint data found. Please run DETECT first.')
            return

        # Placeholder functionality: just ask to save a text file.
        save_path = filedialog.asksaveasfilename(
            defaultextension=".txt", 
            filetypes=[("Text File", "*.txt")],
            initialfile="Expert_Report.txt"
        )
        
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    f.write("--- Expert Motion Analysis Report ---\n\n")
                    f.write(f"Total Frames Analyzed: {self.max_frames + 1}\n\n")
                    f.write("...Detailed joint kinematics and metrics would be here...\n\n")
                    f.write("Report generated successfully.")
                tk.messagebox.showinfo('Report Generation', f'Expert Report template saved to {save_path}')
            except Exception as e:
                tk.messagebox.showerror('Error', f'Could not save report: {e}')
        
    def export_cam_video(self, cam_id):
        """
        Exports the processed video from a specific camera ID (Cam0, Cam1, Cam2).
        """
        if cam_id not in self.cam_ids:
            tk.messagebox.showwarning('Export Failed', f'Cam{cam_id} was not used in the last detection run.')
            return

        cam_path_label = [self.labelfilepath, self.labelfilepath2, self.labelfilepath3][cam_id]
        source_path = cam_path_label.cget("text")

        if not os.path.exists(source_path):
             tk.messagebox.showwarning('Export Failed', f'Source video file not found for Cam{cam_id}.')
             return

        try:
            source_cap = cv2.VideoCapture(source_path)
            fps = source_cap.get(cv2.CAP_PROP_FPS)
            orig_width = int(source_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(source_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            source_cap.release()
        except Exception:
            tk.messagebox.showwarning('Export Failed', 'Could not read original video properties.')
            return

        output_size = (orig_width, orig_height) 
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".avi", 
            filetypes=[("AVI Video", "*.avi")],
            initialfile=f"Cam{cam_id}_Processed_Video.avi"
        )

        if not save_path:
            return

        video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps or 10, output_size, True)

        try:
            temp_cap = cv2.VideoCapture(source_path)
            kpts_data = np.loadtxt(f'kpts_cam{cam_id}.dat')
            kpts_per_frame = kpts_data.reshape(-1, len(pose_keypoints), 2)
            
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            
            for kpt_frame in kpts_per_frame:
                ret, frame = temp_cap.read()
                if not ret: break

                # Redraw landmarks onto the original frame using saved keypoints
                landmarks = []
                for kpt_pixel in kpt_frame:
                    if kpt_pixel[0] != -1: 
                        landmarks.append(mp.framework.formats.pose_landmark_pb2.PoseLandmark(
                            x=kpt_pixel[0] / orig_width,
                            y=kpt_pixel[1] / orig_height,
                            z=0, 
                            visibility=1
                        ))
                    else:
                        landmarks.append(mp.framework.formats.pose_landmark_pb2.PoseLandmark(
                            x=0, y=0, z=0, visibility=0
                        ))

                pose_landmarks = mp.framework.formats.pose_landmark_pb2.PoseLandmarkList(landmark=landmarks)

                mp_drawing.draw_landmarks(frame, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4))

                video.write(frame)
            
            temp_cap.release()
            video.release()
            tk.messagebox.showinfo('Export Successful', f'Processed video for Cam{cam_id} saved to {save_path}')

        except Exception as e:
            tk.messagebox.showerror('Export Error', f'An error occurred during Cam{cam_id} video export: {e}')
            if 'video' in locals() and video.isOpened():
                video.release()
            if os.path.exists(save_path):
                 os.remove(save_path) 
    # -----------------------------
    
    def choose_file(self):     
        self.csv_button.configure(state='disabled')
        self.save_button.configure(state='disabled')
        self.replay_button.configure(state='disabled')
        self.frame_slider.configure(state='disabled')
        self.frame_label.config(text="Ready")
        self.progress_bar.lower()
        self.frame_slider.lower()
        self.progress_var.set(0)
        self.cleanup_images() 
        self.frame_paths = []
        self.frame_2d_paths = [[] for _ in range(3)]

        options = ["Cam0 (Front View)", "Cam1 (Side View)", "Cam2 (Third View)"]
        dialog = OptionDialog(self.master, options)
        self.master.wait_window(dialog)
        input_source = dialog.selected_option.get()        
        
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.avi;*.mp4;*.mov")])
        if file_path == "":
            tk.messagebox.showwarning(title='No Video', message='Please upload a video file.')
            return

        if input_source == "Cam0 (Front View)":            
            self.cap0 = cv2.VideoCapture(file_path)
            self.signal.clear()
            tk.messagebox.showinfo(title='Cam0', message=file_path +' has been uploaded.')
            self.labelfilepath.config(text=file_path)
        
        elif input_source == "Cam1 (Side View)":
            self.cap1 = cv2.VideoCapture(file_path)
            self.signal.clear()                
            tk.messagebox.showinfo(title='Cam1', message= file_path +' has been uploaded.')
            self.labelfilepath2.config(text=file_path)
            
        elif input_source == "Cam2 (Third View)":
            self.cap2 = cv2.VideoCapture(file_path)
            self.signal.clear()                
            tk.messagebox.showinfo(title='Cam2', message= file_path +' has been uploaded.')
            self.labelfilepath3.config(text=file_path)
    
    def detect(self):
         if self.is_processing:
             tk.messagebox.showinfo('Processing', 'Detection is already running.')
             return
             
         cams = [c for c in [self.cap0, self.cap1, self.cap2] if c is not None]
         
         if len(cams) >= 2: 
            self.is_processing = True
            self.progress_bar.lift()
            self.frame_slider.lower()
            self.replay_button.configure(state='disabled')
            self.save_button.configure(state='disabled')
            self.csv_button.configure(state='disabled')
            self.frame_label.config(text="Processing...")
            
            self.cleanup_images()
            self.frame_paths = []
            self.frame_2d_paths = [[] for _ in range(3)]
            self.max_frames = 0
            self.progress_var.set(0)
            
            thread = threading.Thread(target=self.play_video) 
            thread.start()   
         else:
            tk.messagebox.showwarning('No Input', 'Please upload at least 2 video files to detect.')
            
    def replay(self):
        if self.max_frames > 0:
            self.frame_slider.lift() 
            self.progress_bar.lower() 
            self.frame_slider.configure(state='normal') 
            self.frame_slider.set(0)
            self.update_frame_display(0)
        else:
            tk.messagebox.showwarning('No Data', 'No detection data available for replay. Please run DETECT first.')


    def play_video(self):    
        
        # 1. Setup camera info and projection matrices
        cam_info = []
        if self.cap0 is not None: cam_info.append((0, cv2.VideoCapture(self.labelfilepath.cget("text")), self.label1))
        if self.cap1 is not None: cam_info.append((1, cv2.VideoCapture(self.labelfilepath2.cget("text")), self.label2))
        if self.cap2 is not None: cam_info.append((2, cv2.VideoCapture(self.labelfilepath3.cget("text")), self.label3))
        
        self.cam_ids = [info[0] for info in cam_info]
        caps = [info[1] for info in cam_info]
        
        if len(cam_info) < 2:
             self.is_processing = False
             return 

        P_list = []
        try:
            for cam_id, cap, _ in cam_info:
                P_list.append(get_projection_matrix(cam_id))
                cap.set(3, frame_shape[1]) 
                cap.set(4, frame_shape[0]) 
        except FileNotFoundError as e:
            self.is_processing = False
            tk.messagebox.showerror('Calibration Error', f'Missing calibration file for Cam{cam_id}: {e}.')
            return
        
        # Get frame count for progress bar
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
        max_frames = min(frame_counts) if frame_counts else 0
        self.max_frames = max_frames - 1 

        poses = {id: mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) for id, _, _ in cam_info}
        
        # Data storage
        kpts_2d = {cam_id: [] for cam_id, _, _ in cam_info}
        kpts_3d = []
        
        # Skeleton connection definitions
        torso = [[11, 12], [12, 18], [18, 17], [17, 11]]
        head = [[7, 3], [3, 2], [2, 1], [1, 0], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10]]
        arm_l = [[11, 13], [13, 15]]
        arm_r = [[12, 14], [14, 16]]
        leg_l = [[17, 19], [19, 21]]
        leg_r = [[18, 20], [20, 22]]
        foot_l = [[21, 23], [23, 25], [21, 25]]
        foot_r = [[22, 24], [24, 26], [22, 26]]
        
        # --- CHANGE: Removed 'neck' from the lists below ---
        body = [torso, head, arm_l, arm_r, leg_l, leg_r, foot_l, foot_r]
        colors = ['red', 'yellow', 'blue', 'green', 'black', 'orange', 'purple', 'brown']
        
        # Figure for 3D plot
        fig = plt.figure(figsize=(3, 3)) 
        ax = fig.add_subplot(111, projection='3d')
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.view_init(elev=5, azim=-90) 
        
        index = 0
        
        # 2. Main processing loop
        while True:
            # Read all available frames
            frames = []
            rets = []
            for cap in caps:
                ret, frame = cap.read()
                rets.append(ret)
                frames.append(frame)

            if not all(rets) or (max_frames > 0 and index >= max_frames): 
                break

            processed_frames = []
            current_frame_2d_kpts = []

            for i, (cam_id, cap, label) in enumerate(cam_info):
                frame = frames[i]
                
                # --- Pose detection and drawing ---
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                result = poses[cam_id].process(frame_rgb)
                frame_rgb.flags.writeable = True
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                frame_keypoints = []
                if result.pose_landmarks:
                    # Correctly extract landmarks based on the pose_keypoints list order
                    all_landmarks = result.pose_landmarks.landmark
                    for p_idx in pose_keypoints:
                        landmark = all_landmarks[p_idx]
                        pxl_x = landmark.x * frame.shape[1]
                        pxl_y = landmark.y * frame.shape[0]
                        frame_keypoints.append([int(round(pxl_x)), int(round(pxl_y))])
                else:
                    frame_keypoints = [[-1, -1]] * len(pose_keypoints)
                    
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4))
                
                processed_frames.append(frame)
                kpts_2d[cam_id].append(frame_keypoints) 
                current_frame_2d_kpts.append(frame_keypoints)
                
                # Save 2D frame
                frame_2d_path = f'./images/cam{cam_id}_frame{str(index).zfill(8)}.png'
                frame_2d_resized = cv2.resize(frame, (300, 200))
                cv2.imwrite(frame_2d_path, frame_2d_resized)
                self.frame_2d_paths[cam_id].append(frame_2d_path)
                
                # --- Live 2D Display ---
                frame_2d_display = cv2.cvtColor(frame_2d_resized, cv2.COLOR_BGR2RGB)
                img_tk_2d = self.convert_image_to_tk(frame_2d_display)
                
                self.lock.acquire()
                label.config(image=img_tk_2d)
                label.image = img_tk_2d
                self.lock.release()
                
            # --- Triangulation ---
            frame_p3ds = [DLT_multi(P_list, [cam_kpts[kpt_idx] for cam_kpts in current_frame_2d_kpts]) 
                          for kpt_idx in range(len(pose_keypoints))]
            frame_p3ds = np.array(frame_p3ds).reshape((len(pose_keypoints), 3))

            # --- DYNAMIC CENTERING ---
            valid_kpts = frame_p3ds[~np.all(frame_p3ds == -1, axis=1)]
            
            if len(valid_kpts) > 0:
                center_offset = np.mean(valid_kpts, axis=0)
                frame_p3ds_centered = frame_p3ds - center_offset
            else:
                frame_p3ds_centered = frame_p3ds 

            kpts_3d.append(frame_p3ds_centered)

            # --- 3D Plotting & Saving to disk ---
            for bodypart, part_color in zip(body, colors):
                for _c in bodypart:
                    p1_valid = not np.all(frame_p3ds_centered[_c[0]] == -1)
                    p2_valid = not np.all(frame_p3ds_centered[_c[1]] == -1)
                    if p1_valid and p2_valid:
                        ax.plot(xs=[frame_p3ds_centered[_c[0], 0], frame_p3ds_centered[_c[1], 0]], 
                                ys=[frame_p3ds_centered[_c[0], 1], frame_p3ds_centered[_c[1], 1]],
                                zs=[frame_p3ds_centered[_c[0], 2], frame_p3ds_centered[_c[1], 2]], linewidth=2, c=part_color)
            
            ax.set_title("") 
            
            # Non-dynamic, constant view
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]);
            ax.set_xlim3d(-10, 10); ax.set_xlabel('x'); 
            ax.set_ylim3d(-10, 10); ax.set_ylabel('y'); 
            ax.set_zlim3d(-10, 10); ax.set_zlabel('z'); 

            # Save 3D plot
            frame_3d_path = './images/{}.png'.format(str(index).zfill(8))
            plt.savefig(frame_3d_path, bbox_inches='tight', pad_inches=0, dpi=100)
            self.frame_paths.append(frame_3d_path)
            ax.cla()
            
            # --- Live 3D Display ---
            frame_3d = cv2.imread(frame_3d_path)
            frame_3d_display = cv2.cvtColor(cv2.resize(frame_3d, (300, 240)), cv2.COLOR_BGR2RGB) 
            img_tk = self.convert_image_to_tk(frame_3d_display)

            self.lock.acquire()
            self.label0.config(image=img_tk)
            self.label0.image = img_tk
            self.lock.release()
            
            # Update progress bar
            if max_frames > 0:
                self.progress_var.set((index / max_frames) * 100)
            self.frame_label.config(text=f"Processing Frame: {index+1}/{max_frames}")
            
            index += 1
            
            k = cv2.waitKey(1)
            if k & 0xFF == 27: break 

        # 3. Finalization and UI Setup
        cv2.destroyAllWindows()
        for cap in caps: cap.release()
        
        plt.close(fig) 
        
        for cam_id in self.cam_ids:
            write_keypoints_to_disk(f'kpts_cam{cam_id}.dat', np.array(kpts_2d[cam_id]))
        write_keypoints_to_disk('kpts_3d.dat', np.array(kpts_3d))
        
        self.cap0, self.cap1, self.cap2 = None, None, None 

        if len(self.frame_paths) > 0:
            self.max_frames = len(self.frame_paths) - 1
            self.frame_slider.configure(from_=0, to=self.max_frames, state='disabled')
            
            self.progress_var.set(100)
            self.frame_label.config(text=f"Processing Complete. Click REPLAY to navigate.")
            
            self.csv_button.configure(state='normal')
            self.save_button.configure(state='normal')
            self.replay_button.configure(state='normal') 
            
            self.update_frame_display(self.max_frames)
        else:
            self.frame_label.config(text="Detection Failed. No frames generated.")

        self.is_processing = False
        self.signal.set()

    def update_frame_display(self, value):
        frame_idx = int(float(value))
        
        if frame_idx < 0 or frame_idx > self.max_frames:
            return

        self.frame_label.config(text=f"Replay Frame: {frame_idx}/{self.max_frames}")
        
        # 1. Update 3D Frame (label0) - Using smaller display size
        frame_3d_path = self.frame_paths[frame_idx]
        frame_3d = cv2.imread(frame_3d_path)
        
        frame_3d_display = cv2.cvtColor(cv2.resize(frame_3d, (300, 240)), cv2.COLOR_BGR2RGB)
        img_tk = self.convert_image_to_tk(frame_3d_display)
        
        self.lock.acquire()
        self.label0.config(image=img_tk)
        self.label0.image = img_tk
        self.lock.release()

        # 2. Update 2D Frames (label1, label2, label3)
        cam_labels = [self.label1, self.label2, self.label3]
        
        for i, label in enumerate(cam_labels):
            if i in self.cam_ids: 
                if i < len(self.frame_2d_paths) and frame_idx < len(self.frame_2d_paths[i]):
                     frame_2d_path = self.frame_2d_paths[i][frame_idx]
                else:
                     continue
                
                try:
                    frame_2d = cv2.imread(frame_2d_path)
                    frame_2d_display = cv2.cvtColor(frame_2d, cv2.COLOR_BGR2RGB)
                    img_tk_2d = self.convert_image_to_tk(frame_2d_display)
                    
                    self.lock.acquire()
                    label.config(image=img_tk_2d)
                    label.image = img_tk_2d
                    self.lock.release()
                except Exception:
                    pass


    def save_csv(self):               
        kpts_3d_csv = np.loadtxt('kpts_3d.dat')
        if kpts_3d_csv.size == 0:
            self.csv_button.config(state='disabled')
            tk.messagebox.showwarning('Fail to Export CSV', 'No 3D coordinates to be exported.')

        else:
            self.csv_button.config(state='normal')
            df = pd.DataFrame(kpts_3d_csv)
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")], initialfile="3D_Keypoints.csv")

            if save_path:
                df.to_csv(save_path, index=False)
                tk.messagebox.showinfo('Export CSV', 'Successfully exported 3D keypoints as CSV')
                self.csv_button.configure(state='disabled')
                

    def save(self):
        # This function is now specifically for Export 3D Skeleton Video
        import glob,os
        with open('kpts_3d.dat', 'r') as file:
            file_contents = file.read()
        if file_contents:
            path = './images/'
            
            image_files = glob.glob(os.path.join(path, '*.png'))          
            filelist = sorted([f for f in image_files if os.path.basename(f)[0].isdigit()]) 
            
            fps = 10  
            size = (800, 547) 

            if filelist: 
                count = len(glob.glob("output/Output*.avi")) + 1           
                video_name = filedialog.asksaveasfilename(
                    defaultextension=".avi", 
                    filetypes=[("AVI Video", "*.avi")],
                    initialfile=f"3D_Skeleton_Video.avi"
                )
                
                if not video_name: return 

                video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), fps, size, True)

                for item in filelist:
                    img = cv2.imread(item)
                    img = cv2.resize(img, (size))
                    video.write(img)                    
                video.release()
                
                cv2.destroyAllWindows()
                    
                tk.messagebox.showinfo('Save', f'Successfully saved 3D skeleton video to {video_name}')                
            else:
                tk.messagebox.showwarning('NULL', "Please run detection first to generate 3D frames before saving.")            
   
    def convert_image_to_tk(self, img):
        height, width, channels = img.shape
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        return img_tk
    
    def cleanup_images(self):
        # Cleans up temporary image files
        import glob, os
        path = './images/'
        image_files = glob.glob(os.path.join(path, '*.png'))
        for file in image_files:
            os.remove(file)

    def run(self):
        self.master.mainloop()

root = tk.Tk()
app = Application(root)
app.run()