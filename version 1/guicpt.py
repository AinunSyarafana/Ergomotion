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
# DLT is replaced by DLT_multi in utils, so we import DLT_multi
from utils import DLT_multi, get_projection_matrix, write_keypoints_to_disk 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# Re-import DLT_multi from utils for clarity
from utils import DLT_multi
import pandas as pd
plt.style.use('seaborn-v0_8')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [1280,720]

pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28,0,3,6]

class OptionDialog(tk.Toplevel):
    def __init__(self, master, options):
        super().__init__(master)        
        self.title("Browse File")
        self.geometry("400x100") # Expanded size for 3 buttons
        self.resizable(width=False, height=False)
        window_width = self.winfo_reqwidth()
        window_height = self.winfo_reqheight()
       
        x = (1200 - window_width) // 2
        y = (674 - window_height) // 2
        self.geometry(f"+{x}+{y}")

        self.selected_option = tk.StringVar()

        label = tk.Label(self, text="Select Video From:")
        label.pack(pady=10)

        # Updated to include "Cam2 (Third View)"
        for option in options:
            button = tk.Button(self, text=option, width=15, height=2,
                               command=lambda option=option: self.on_option_selected(option))
            button.pack(side=tk.LEFT, padx=5) # Reduced padx

    def on_option_selected(self, option):
        self.selected_option.set(option)
        self.destroy()

class Application:
    def __init__(self, master):       

        self.master = master
        self.master.title("Motion Detection with 3D Reconstruction App")
        self.master.configure(bg="#d0bdf4")
        self.master.resizable(width=False, height=False)       

        w = 1000 # Increased width to fit 3 videos
        h = 700 # Increased height

        ws = root.winfo_screenwidth()
        hs = root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)

        root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        
        style = ThemedStyle(root)
        style.set_theme("yaru")

        style = ttk.Style()
        style.configure("Custom.TFrame", background="#d0bdf4")

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

        # 3D Plot Label (Center Top)
        self.label0 = tk.Label(master, background="#d0bdf4")
        self.label0.pack(pady=5)

        # Frame for 2D Video Views (Bottom)
        video_frame = ttk.Frame(master, style="Custom.TFrame")
        video_frame.pack(pady=5)

        # Cam0 (Left)
        self.label1 = tk.Label(video_frame, background="#d0bdf4")
        self.label1.pack(side=tk.LEFT, padx=5)

        # Cam1 (Middle)
        self.label2 = tk.Label(video_frame, background="#d0bdf4")
        self.label2.pack(side=tk.LEFT, padx=5)

        # Added Label for Cam2 (Right)
        self.label3 = tk.Label(video_frame, background="#d0bdf4")
        self.label3.pack(side=tk.RIGHT, padx=5)

        # Hidden file path labels
        self.labelfilepath = tk.Label(master, background="#d0bdf4", foreground="#d0bdf4")
        self.labelfilepath.pack()

        self.labelfilepath2 = tk.Label(master, background="#d0bdf4", foreground="#d0bdf4")
        self.labelfilepath2.pack()
        
        # Added Label for Cam2 file path
        self.labelfilepath3 = tk.Label(master, background="#d0bdf4", foreground="#d0bdf4")
        self.labelfilepath3.pack()

        # synchronize cap0, cap1, and cap2
        self.lock = threading.Lock()
        self.signal = threading.Event()

        # Added self.cap2 initialization
        self.cap0 = None
        self.cap1 = None
        self.cap2 = None
    
    def choose_file(self):     
        self.csv_button.configure(state='disabled')
        self.save_button.configure(state='disabled')
        self.replay_button.configure(state='disabled')

        # Updated options to include "Cam2 (Third View)"
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
            
        # Added logic for Cam2
        elif input_source == "Cam2 (Third View)":
            self.cap2 = cv2.VideoCapture(file_path)
            self.signal.clear()                
            tk.messagebox.showinfo(title='Cam2', message= file_path +' has been uploaded.')
            self.labelfilepath3.config(text=file_path)
    
    def detect(self):
         # Get all available cameras
         cams = [c for c in [self.cap0, self.cap1, self.cap2] if c is not None]
         
         if len(cams) >= 2: # Requires at least 2 cameras for triangulation
            thread = threading.Thread(target=self.play_video) 
            thread.start()   
         else:
            tk.messagebox.showwarning('No Input', 'Please upload at least 2 video files to detect.')

    def replay(self):
        # Re-initialize all caps from file paths
        path0 = self.labelfilepath.cget("text")
        path1 = self.labelfilepath2.cget("text")
        path2 = self.labelfilepath3.cget("text") # Added path2

        if path0: self.cap0 = cv2.VideoCapture(path0)
        if path1: self.cap1 = cv2.VideoCapture(path1)
        if path2: self.cap2 = cv2.VideoCapture(path2) # Added cap2 init

        cams = [c for c in [self.cap0, self.cap1, self.cap2] if c is not None]
        
        if len(cams) >= 2:
            thread = threading.Thread(target=self.play_video)
            thread.start()   
        else:
            tk.messagebox.showwarning('No Input', 'There are not enough videos to replay')

    def play_video(self):    
        self.csv_button.configure(state='disabled')
        self.save_button.configure(state='normal')
        self.replay_button.configure(state='normal')

        # List of available cameras and their IDs/Labels
        cam_info = []
        if self.cap0 is not None: cam_info.append((0, self.cap0, self.label1))
        if self.cap1 is not None: cam_info.append((1, self.cap1, self.label2))
        if self.cap2 is not None: cam_info.append((2, self.cap2, self.label3)) # Added Cam2 info

        if len(cam_info) < 2:
             tk.messagebox.showwarning('Error', 'Need at least 2 views for 3D reconstruction.')
             return

        # Load projection matrices P_i and set video properties
        P_list = []
        try:
            for cam_id, cap, _ in cam_info:
                # get_projection_matrix() will attempt to load cX.dat and rot_trans_cX.dat
                P_list.append(get_projection_matrix(cam_id))
                cap.set(3, frame_shape[1]) #set width
                cap.set(4, frame_shape[0]) #height
        except FileNotFoundError as e:
            tk.messagebox.showerror('Calibration Error', f'Missing calibration file for Cam{cam_id}: {e}. Ensure c{cam_id}.dat and rot_trans_c{cam_id}.dat exist.')
            return

        # Create body keypoints detector objects for each camera
        poses = {id: mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) for id, _, _ in cam_info}

        kpts_2d = [[] for _ in range(len(cam_info))] # list of lists to store 2D kpts for each camera
        kpts_3d = []
        
        # Body connection definitions (unchanged)
        torso = [[3, 4], [4, 10], [10, 9], [9, 3]]
        armr = [[4, 6], [6, 8]]
        arml = [[3, 5], [5, 7]]
        legr = [[9, 11], [11, 13]]
        legl = [[10, 12], [12, 14]]
        head = [[0, 1], [0, 2]]
        body = [torso, arml, armr, legr, legl, head]
        colors = ['red', 'blue', 'green', 'black', 'orange', 'yellow']

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=-80, azim=0)
        index = 0
        
        # Define keypoint indices used for DLT
        pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28, 0, 3, 6]

        while True:
            # Read all available frames
            frames = []
            rets = []
            
            for _, cap, _ in cam_info:
                ret, frame = cap.read()
                rets.append(ret)
                frames.append(frame)

            if not all(rets): 
                # If any camera stream ends, break
                break

            # Process frames for detection
            processed_frames = []
            current_frame_2d_kpts = []

            for i, (cam_id, cap, label) in enumerate(cam_info):
                frame = frames[i]
                
                frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                result = poses[cam_id].process(frame_rgb)

                frame_rgb.flags.writeable = True
                frame = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)

                # Keypoints detection
                frame_keypoints = []
                
                if result.pose_landmarks:
                    for j, landmark in enumerate(result.pose_landmarks.landmark):
                        if j not in pose_keypoints: continue
                        pxl_x = landmark.x * frame.shape[1]
                        pxl_y = landmark.y * frame.shape[0]
                        pxl_x = int(round(pxl_x))
                        pxl_y = int(round(pxl_y))
                        frame_keypoints.append([pxl_x, pxl_y])
                else:
                    # If no keypoints are found, fill with [-1,-1]
                    frame_keypoints = [[-1, -1]] * len(pose_keypoints)
                    
                
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4))
                
                processed_frames.append(frame)
                kpts_2d[i].append(frame_keypoints)
                current_frame_2d_kpts.append(frame_keypoints)
                
            # Triangulation of 3D position
            frame_p3ds = []
            
            # Iterate over each keypoint (0 to len(pose_keypoints)-1)
            for kpt_idx in range(len(pose_keypoints)):
                
                # Extract the 2D point for the current keypoint from all available cameras
                uv_list = [cam_kpts[kpt_idx] for cam_kpts in current_frame_2d_kpts]
                
                # Triangulate using DLT_multi, combining all active views
                _p3d = DLT_multi(P_list, uv_list)
                
                frame_p3ds.append(_p3d)

            
            frame_p3ds = np.array(frame_p3ds).reshape((len(pose_keypoints), 3))
            kpts_3d.append(frame_p3ds)

            # Plot 3D result (unchanged)
            for bodypart, part_color in zip(body, colors):
                for _c in bodypart:
                    ax.plot(xs=[frame_p3ds[_c[0], 0], frame_p3ds[_c[1], 0]], ys=[frame_p3ds[_c[0], 1], frame_p3ds[_c[1], 1]],
                            zs=[frame_p3ds[_c[0], 2], frame_p3ds[_c[1], 2]], linewidth=2, c=part_color)

            for i in range(len(pose_keypoints)):
                ax.text(frame_p3ds[i, 0], frame_p3ds[i, 1], frame_p3ds[i, 2], str(i))
                ax.scatter(xs=frame_p3ds[i:i + 1, 0], ys=frame_p3ds[i:i + 1, 1], zs=frame_p3ds[i:i + 1, 2])
                
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.set_xlim3d(-10, 20)
            ax.set_xlabel('x')
            ax.set_ylim3d(-10, 20)
            ax.set_ylabel('y')
            ax.set_zlim3d(-10, 0)
            ax.set_zlabel('z')

            # Save 3D plot and display
            plt.savefig('./images/{}.png'.format(str(index).zfill(8)))
            ax.cla()

            frame_3d = cv2.imread('./images/{}.png'.format(str(index).zfill(8)))
            top = 70
            bottom = 70
            left = 100
            right = 80

            frame_3d = frame_3d[top:-bottom, left:-right]
            frame_3d = cv2.resize(frame_3d,(400,320)) # Reduced size for 3D plot
            
            index = index + 1
            frame_3d = cv2.cvtColor(frame_3d, cv2.COLOR_BGR2RGB)
            img_tk = self.convert_image_to_tk(frame_3d)
            
            self.lock.acquire()
            self.label0.config(image=img_tk)
            self.label0.image = img_tk
            self.lock.release()

            # Display 2D processed frames
            for i, (_, _, label) in enumerate(cam_info):
                frame_2d = processed_frames[i]
                frame_2d = cv.resize(frame_2d,(300,200)) # Smaller size for 3 views in a row

                frame_2d = cv2.cvtColor(frame_2d, cv2.COLOR_BGR2RGB)
                img_tk = self.convert_image_to_tk(frame_2d)
                self.lock.acquire()
                label.config(image=img_tk)
                label.image = img_tk
                self.lock.release()

            k = cv.waitKey(1)
            if k & 0xFF == 27: break 

        cv.destroyAllWindows()
        for _, cap, _ in cam_info:
            cap.release()

        # Write keypoints to disk for all active cameras
        for i, (cam_id, _, _) in enumerate(cam_info):
            kpts_2d_array = np.array(kpts_2d[i])
            write_keypoints_to_disk(f'kpts_cam{cam_id}.dat', kpts_2d_array)
        
        kpts_3d_array = np.array(kpts_3d)
        write_keypoints_to_disk('kpts_3d.dat', kpts_3d_array)
        
        kpts_3d_csv = np.loadtxt('kpts_3d.dat')
        if kpts_3d_csv.size != 0:
            self.csv_button.configure(state='normal')
        
        self.cap0 = None
        self.cap1 = None
        self.cap2 = None # Reset cap2

        from mpl_toolkits.mplot3d import Axes3D

        self.save()   
        self.save_button.configure(state='disabled')

        import glob,os
        path = './images/'
        image_files = glob.glob(os.path.join(path, '*.png')) 
        for file in image_files:
            os.remove(file)
        self.signal.set()

    def save_csv(self):               
        kpts_3d_csv = np.loadtxt('kpts_3d.dat')
        if kpts_3d_csv.size == 0:
            self.csv_button.config(state='disabled')
            tk.messagebox.showwarning('Fail to Export CSV', 'No 3D coordinates to be exported.')

        else:
            self.csv_button.config(state='normal')
            df = pd.DataFrame(kpts_3d_csv)
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])

            if save_path:
                df.to_csv(save_path, index=False)
                tk.messagebox.showinfo('Export CSV', 'Successfully exported 3D keypoints as CSV')
                self.csv_button.configure(state='disabled')
                

    def save(self):
        import glob,os
        with open('kpts_3d.dat', 'r') as file:
            file_contents = file.read()
        if file_contents:
            path = './images/'
            image_files = glob.glob(os.path.join(path, '*.png'))          

            filelist = os.listdir(path)

            fps = 10  
            size = (800, 547)  #size to convert video (Original size, adjust if needed)      

            if image_files: 
                count = len(glob.glob("output/Output*.avi")) + 1           
                video = cv2.VideoWriter("output/Output "+str(count)+".avi", cv2.VideoWriter_fourcc(*'XVID'), fps, size, True)

                for item in filelist:
                    if item.endswith('.png'):
                        item = path + item
                        img = cv2.imread(item)
                        img = cv2.resize(img, (size))
                        video.write(img)                    
                video.release()
                
                # Simple cleanup logic
                if filelist: 
                    for file in image_files:
                        os.remove(file)
                cv2.destroyAllWindows()
                    
                tk.messagebox.showinfo('Save', 'Successfully saved as Output '+ str(count)+".avi")                
            else:
                tk.messagebox.showwarning('NULL', "Please upload valid video for motion detection before saving")            
   
    def convert_image_to_tk(self, img):
        height, width, channels = img.shape
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        return img_tk

    def run(self):
        self.master.mainloop()

root = tk.Tk()
app = Application(root)
app.run()