import cv2
import time
import os
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from paddleocr import PaddleOCR

# ==========================================
# 1. KHỞI TẠO HỆ THỐNG LƯU TRỮ (DATABASE LOCAL)
# ==========================================
SAVE_DIR = "Data_Export"
IMG_DIR = os.path.join(SAVE_DIR, "images")
CSV_FILE = os.path.join(SAVE_DIR, "history.csv")

# Tự động tạo thư mục nếu chưa có
os.makedirs(IMG_DIR, exist_ok=True)

# Tạo file CSV và ghi tiêu đề cột nếu file chưa tồn tại
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Thoi Gian", "File Anh", "Noi Dung Doc Duoc", "Do Tu Tin Trung Binh"])

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH AI & CAMERA
# ==========================================
print("[INFO] Đang nạp hệ thống nhận diện...")
# Nạp "bộ não" 96.9% của bạn
ocr = PaddleOCR(rec_model_dir='./output/rec_inference/', lang='en')

# Camera DroidCam (Hoặc Webcam vật lý)
cap = cv2.VideoCapture(1)

# ==========================================
# 3. THIẾT KẾ GIAO DIỆN DASHBOARD (TKINTER)
# ==========================================
root = tk.Tk()
root.title("Camera AI by Phạm Quốc Thắng")
root.geometry("1000x550")
root.configure(bg="#f0f0f0")

# Khung bên trái (Chứa Camera)
frame_left = tk.Frame(root, bg="#000", width=640, height=480)
frame_left.pack(side=tk.LEFT, padx=10, pady=10)
lbl_video = tk.Label(frame_left)
lbl_video.pack()

# Khung bên phải (Chứa Lịch sử Quét)
frame_right = tk.Frame(root, bg="#f0f0f0", width=340)
frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

lbl_title = tk.Label(frame_right, text="LỊCH SỬ NHẬN DIỆN", font=("Arial", 14, "bold"), bg="#f0f0f0")
lbl_title.pack(pady=5)

# --- ĐỒNG HỒ THỜI GIAN THỰC ---
lbl_datetime = tk.Label(frame_right, text="Đang đồng bộ thời gian...", font=("Arial", 12, "italic"), bg="#f0f0f0", fg="#0052cc")
lbl_datetime.pack(pady=2)

# Bảng hiển thị danh sách (Treeview)
columns = ("Thời gian", "Nội dung", "Độ tự tin")
tree = ttk.Treeview(frame_right, columns=columns, show="headings", height=20)
tree.heading("Thời gian", text="Thời gian")
tree.column("Thời gian", width=80, anchor=tk.CENTER)
tree.heading("Nội dung", text="Nội dung")
tree.column("Nội dung", width=120, anchor=tk.W)
tree.heading("Độ tự tin", text="Độ tự tin")
tree.column("Độ tự tin", width=80, anchor=tk.CENTER)
tree.pack(fill=tk.BOTH, expand=True)

# Biến đếm thời gian quét tự động
last_scan_time = time.time()
SCAN_INTERVAL = 2.0  # Quét mỗi 2 giây

# ==========================================
# 4. VÒNG LẶP XỬ LÝ CHÍNH
# ==========================================
def update_dashboard():
    global last_scan_time
    
    ret, frame = cap.read()
    if ret:
        current_time = time.time()
        
        # CẬP NHẬT NGÀY GIỜ LÊN MÀN HÌNH
        now_str = datetime.now().strftime("Hôm nay: %d/%m/%Y  |  %H:%M:%S")
        lbl_datetime.config(text=now_str)
        
        # 1. Vẽ Vùng Quan Tâm (ROI) - Chỉ đọc chữ lọt vào trong khung này
        x1, y1, x2, y2 = 50, 150, 590, 350
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "VUNG QUET MA / BIEN SO", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # CHU KỲ QUÉT AI TỰ ĐỘNG
        if current_time - last_scan_time >= SCAN_INTERVAL:
            
            # 2. CẮT ẢNH TRƯỚC KHI ĐỌC (Lọc bỏ nhiễu bối cảnh)
            crop_img = frame[y1:y2, x1:x2]
            result = ocr.ocr(crop_img)
            
            if result and result[0]:
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                time_display = datetime.now().strftime("%H:%M:%S")
                
                texts = []
                confidences = []
                
                for line in result[0]:
                    text = line[1][0].strip()
                    conf = line[1][1]
                    
                    # 3. BỘ LỌC RÁC: Loại bỏ watermark và các ký tự ảo giác
                    if "droidcam" not in text.lower() and len(text) >= 3:
                        texts.append(text)
                        confidences.append(conf)
                
                # Chỉ xử lý và lưu nếu mảng texts chứa dữ liệu hợp lệ
                if texts:
                    full_text = " | ".join(texts)
                    avg_conf = sum(confidences) / len(confidences)
                    conf_str = f"{avg_conf*100:.1f}%"
                    
                    # Lưu hình ảnh 
                    img_filename = f"scan_{timestamp_str}.jpg"
                    img_path = os.path.join(IMG_DIR, img_filename)
                    cv2.imwrite(img_path, frame)
                    
                    # Ghi thông tin vào file cấu trúc CSV
                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([time_display, img_filename, full_text, conf_str])
                    
                    # Đẩy lên Giao diện Bảng điều khiển (Thêm vào dòng đầu tiên)
                    tree.insert("", 0, values=(time_display, full_text, conf_str))
                    
            last_scan_time = current_time

        # Hiển thị luồng Video lên Giao diện (Mượt mà 30 FPS)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        
        lbl_video.imgtk = img_tk
        lbl_video.configure(image=img_tk)

    # Lặp lại luồng quét UI sau 15ms
    root.after(15, update_dashboard)

# Khởi động vòng lặp Camera
update_dashboard()

# Khởi động Giao diện và Quản lý RAM
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()