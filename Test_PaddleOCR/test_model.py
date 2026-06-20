from paddleocr import PaddleOCR

# Khởi tạo mặc định: Sẽ tải mô hình Det tự động cắt khung, 
# và dùng mô hình Rec do bạn train để đọc.
ocr = PaddleOCR(rec_model_dir='./output/rec_inference/', lang='en')

print("\n[INFO] Đang quét ảnh...\n")

# Chạy quét toàn bộ ảnh
result = ocr.ocr('56.png')

print("="*45)
if result and result[0]:
    for idx, line in enumerate(result[0]):
        text = line[1][0]
        conf = line[1][1]
        print(f"Khung chữ {idx + 1}: {text: <15} | Tự tin: {conf * 100:.2f}%")
else:
    print("Không tìm thấy chữ!")
print("="*45)