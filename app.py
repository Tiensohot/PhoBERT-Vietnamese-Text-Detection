# app.py
from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer
from model import TransformerCLS4Concat
import docx
import os

# --- 1. Khởi tạo ứng dụng Flask ---
app = Flask(__name__)

# --- 2. Tải Model và Tokenizer PhoBERT ---
MODEL_NAME = "vinai/phobert-base"
BEST_MODEL_PATH = "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
loaded_model = TransformerCLS4Concat(model_name=MODEL_NAME, num_labels=2)
state = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
loaded_model.load_state_dict(state["model_state_dict"])
loaded_model.to(DEVICE)
loaded_model.eval()

print(">>> Model PhoBERT và Tokenizer đã được tải xong!")

# --- Hàm đọc nội dung file ---
def read_docx_file(file_stream):
    doc = docx.Document(file_stream)
    return "\n".join([para.text for para in doc.paragraphs])

def read_txt_file(file_stream):
    return file_stream.read().decode("utf-8")

# --- 3. Định nghĩa các "route" đường dẫn url ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        text_to_predict = ""
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filename = file.filename
            if filename.endswith('.docx'):
                text_to_predict = read_docx_file(file.stream)
            elif filename.endswith('.txt'):
                text_to_predict = read_txt_file(file.stream)
            else:
                return jsonify({'error': 'Định dạng file không được hỗ trợ. Vui lòng sử dụng .txt hoặc .docx'}), 400
        else:
            text_to_predict = request.form.get('text')

        if not text_to_predict or not text_to_predict.strip():
            return jsonify({'error': 'Không có nội dung để phân tích.'}), 400

        # Tokenize và dự đoán bằng PhoBERT
        enc = tokenizer(text_to_predict, return_tensors="pt", padding=True, truncation=True, max_length=256)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            outputs = loaded_model(**enc)
            logits = outputs["logits"]
            
            if logits is None or logits.size(0) == 0:
                return jsonify({'error': 'Văn bản quá ngắn hoặc không hợp lệ để phân tích.'}), 400
            
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            if probs.ndim > 1:
                probs = probs[0]
            
            prediction_idx = int(probs.argmax())

        labels = ["Human", "AI"]
        prediction_label = labels[prediction_idx]
        confidence = float(probs[prediction_idx])

        return jsonify({
            'prediction': prediction_label,
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'human': round(float(probs[0]) * 100, 2),
                'ai': round(float(probs[1]) * 100, 2)
            }
        })

    except Exception as e:
        print(f"Đã xảy ra lỗi không mong muốn: {e}")
        return jsonify({'error': 'Đã có lỗi xảy ra phía máy chủ.'}), 500

# --- 4. Chạy ứng dụng ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)