import paddle
from paddle.inference import Config, create_predictor
import cv2
import numpy as np

# --- THÊM THƯ VIỆN GIẢI MÃ CỦA PADDLEOCR ---
from ppocr.postprocess.rec_postprocess import CTCLabelDecode 

# 1. Cấu hình nạp file Inference
model_dir = r'C:\Users\Admin\Documents\AI\Final_Project\Source code\PaddleOCR\output\rec_inference\inference'
config = Config(model_dir + ".pdmodel", model_dir + ".pdiparams")
config.disable_gpu()
predictor = create_predictor(config)

# 2. Đọc và tiền xử lý ảnh
img = cv2.imread('56.png')
img = cv2.resize(img, (320, 48)) 
input_data = img.transpose((2, 0, 1)) 
input_data = input_data[np.newaxis, :] / 255.0 

# 3. Chạy dự đoán (Đưa ảnh vào mạn lưới Neural)
input_names = predictor.get_input_names()
input_handle = predictor.get_input_handle(input_names[0])
input_handle.copy_from_cpu(input_data.astype('float32'))
predictor.run()

# 4. Lấy ma trận xác suất thô
output_names = predictor.get_output_names()
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu()

# 5. GIẢI MÃ TENSOR THÀNH VĂN BẢN
# Khởi tạo bộ giải mã với từ điển tiếng Anh mặc định của OCR
decoder = CTCLabelDecode(character_dict_path='ppocr/utils/en_dict.txt', use_space_char=True)

# Chuyển Tensor qua bộ giải mã
decode_out = decoder(output_data)
text = decode_out[0][0]
confidence = decode_out[0][1]

print("\n" + "="*45)
print(f"🎉 CHỮ ĐỌC ĐƯỢC : {text}")
print(f"🎯 ĐỘ TỰ TIN   : {confidence * 100:.2f}%")
print("="*45 + "\n")