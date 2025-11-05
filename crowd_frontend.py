import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

# ------------------ Crowd Detection Functions ------------------

def preprocess_image(img_path, target_size=640):
    img0 = cv2.imread(img_path)
    if img0 is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    h0, w0 = img0.shape[:2]
    scale = target_size / max(h0, w0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img0, (new_w, new_h))
    dw, dh = target_size - new_w, target_size - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    return img0, img_norm

def detect_and_count(img_path, model_path="yolov8s.pt", conf_threshold=0.3):
    model = YOLO(model_path)
    results = model(img_path, conf=conf_threshold, classes=[0])  # class 0 = person
    det = results[0]
    count = len(det.boxes)
    return det, count

# ----------- Improved Occupancy Estimation Logic --------------
def occupancy_level(count, image_area=640*640, avg_area_per_person=15000):
    if count == 0:
        return "Low"
    occupancy_percent = (count * avg_area_per_person / image_area) * 100

    if occupancy_percent < 25:
        return "Low"
    elif occupancy_percent < 60:
        return "Medium"
    else:
        return "High"

# ---------------------------------------------------------------

def draw_results(img0, det):
    im0 = img0.copy()
    for box in det.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return im0

# ------------------ GUI Functions ------------------

def browse_image():
    global img_path
    img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if img_path:
        img = Image.open(img_path).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

def process_image():
    if not img_path:
        messagebox.showwarning("Warning", "Please upload an image first!")
        return
    try:
        img0, _ = preprocess_image(img_path)
        det, count = detect_and_count(img_path)
        occ = occupancy_level(count)
        out_img = draw_results(img0, det)

        # Update GUI fields
        count_var.set(str(count))
        occupancy_var.set(occ)

        # Show annotated image
        img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((300, 300))
        img_tk = ImageTk.PhotoImage(img_pil)
        img_label.config(image=img_tk)
        img_label.image = img_tk

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ------------------ GUI Layout ------------------

root = tk.Tk()
root.title("Crowd Detection & Occupancy Estimation")
root.geometry("400x600")
root.resizable(False, False)
root.config(bg="#F3F4F6")

img_path = None

# ----- Top Frame for Image -----
frame_img = tk.Frame(root, bg="#E5E7EB", bd=2, relief="groove")
frame_img.pack(pady=15, padx=15)

placeholder = Image.new("RGB", (300, 300), color="#D1D5DB")
img_tk = ImageTk.PhotoImage(placeholder)
img_label = tk.Label(frame_img, image=img_tk, bg="#E5E7EB")
img_label.image = img_tk
img_label.pack()

# ----- Buttons Frame -----
frame_btn = tk.Frame(root, bg="#F3F4F6")
frame_btn.pack(pady=10)

upload_btn = tk.Button(frame_btn, text="Upload Image", width=15, command=browse_image,
                       bg="#2563EB", fg="white", font=("Segoe UI", 10, "bold"), relief="flat")
upload_btn.grid(row=0, column=0, padx=5, pady=5)

process_btn = tk.Button(frame_btn, text="Detect Crowd", width=15, command=process_image,
                        bg="#16A34A", fg="white", font=("Segoe UI", 10, "bold"), relief="flat")
process_btn.grid(row=0, column=1, padx=5, pady=5)

# ----- Results Frame -----
frame_res = tk.LabelFrame(root, text="Results", bg="#F3F4F6", fg="#1F2937",
                          font=("Segoe UI", 11, "bold"), padx=10, pady=10)
frame_res.pack(padx=15, pady=10, fill="x")

tk.Label(frame_res, text="People Count:", bg="#F3F4F6", font=("Segoe UI", 10, "bold")).pack(pady=(0,5))
count_var = tk.StringVar(value="0")
count_entry = ttk.Entry(frame_res, textvariable=count_var, state="readonly", justify="center",
                        font=("Consolas", 12))
count_entry.pack(pady=(0,10), fill="x")

tk.Label(frame_res, text="Occupancy Level:", bg="#F3F4F6", font=("Segoe UI", 10, "bold")).pack(pady=(0,5))
occupancy_var = tk.StringVar(value="N/A")
occupancy_entry = ttk.Entry(frame_res, textvariable=occupancy_var, state="readonly", justify="center",
                            font=("Consolas", 12))
occupancy_entry.pack(pady=(0,10), fill="x")

# ----- Footer -----
footer = tk.Label(root, text="Crowd Detection using YOLOv8", bg="#F3F4F6",
                  fg="#6B7280", font=("Segoe UI", 9))
footer.pack(side="bottom", pady=10)

if __name__ == "__main__":
    root.mainloop()
