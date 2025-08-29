 <h1>🚀Nhận diện văn bản tiếng Việt sinh bởi mô hình ngôn ngữ lớn (LLM) sử dụng PhoBERT và học sâu</h1>

 <h2>📌Giới thiệu</h2>

Đây là đồ án chuyên ngành thuộc Ngành Khoa học Máy tính – Trường Đại học Công nghiệp Hà Nội.
Đề tài tập trung xây dựng hệ thống có khả năng phân biệt văn bản do con người viết và văn bản sinh ra bởi AI (Large Language Models – LLM).

Chúng tôi sử dụng PhoBERT, một mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, kết hợp với Deep Learning (PyTorch) để xây dựng bộ phân loại văn bản.
Sản phẩm cuối cùng gồm:

📒 Mô hình huấn luyện (final.ipynb, best_model.pt)

🌐 Ứng dụng web demo với Flask + HTML/CSS/JS

 <h2>🛠️Công nghệ sử dụng</h2>

Áp dụng kiến thức từ nhiều môn học trong chương trình:

Xử lý ngôn ngữ tự nhiên (NLP) → PhoBERT tokenizer, embedding, classification

Trí tuệ nhân tạo (AI) → các mô hình học sâu

Học sâu (Deep Learning) → fine-tune PhoBERT bằng PyTorch

Kỹ thuật phần mềm & Lập trình Web → Flask, HTML, CSS, JS

Khoa học dữ liệu → xử lý và gán nhãn tập dữ liệu (1350 mẫu Human/AI)

 <h2>📂Cấu trúc thư mục</h2>
PhoBERT-Text-Detection/
│── app.py              # Flask server (chạy demo)
│── model.py            # Định nghĩa PhoBERT + classifier
│── best_model.pt       # Trọng số mô hình đã huấn luyện
│── final.ipynb         # Notebook huấn luyện & đánh giá
│── templates/
│   └── index.html      # Giao diện demo web
│── requirements.txt    # Thư viện cần cài đặt

 <h2>▶️Cách chạy demo</h2>
git clone ""https://github.com/Tiensohot/PhoBERT-Vietnamese-Text-Detection/""
download file "best_model.pt" tại link: https://drive.google.com/file/d/1ucctLZc7JNaRlUsEdXMJS-o8k_CmAwyU/view?usp=sharing
cd PhoBERT-Vietnamese-Text-Detection

pip install -r requirements.txt

python app.py

Mở trình duyệt tại: http://127.0.0.1:5000

→ Dán văn bản hoặc upload file .txt/.docx để phân tích.

 <h2>📊Kết quả nổi bật</h2>

✅ Accuracy: 99.6%

✅ F1-score: 0.9963

🔥 Vượt trội hơn TF-IDF + SVM và Rule-based

<h2>👨‍👩‍👦‍👦Đóng góp của nhóm</h2>

Đồ án là thành quả teamwork nghiêm túc của Nhóm 09:

Nguyễn Huy Cương – Tiền xử lý dữ liệu, huấn luyện mô hình

Đào Văn Hiệp – Phân tích thuật toán, viết báo cáo

Bùi Đức Tiến – Xây dựng giao diện demo, tích hợp hệ thống

Chúng tôi đã cùng nhau làm việc từ tìm hiểu lý thuyết → triển khai mô hình → đánh giá → xây dựng demo → viết báo cáo.
Sự phối hợp ăn ý là chìa khóa giúp đồ án hoàn thành đúng tiến độ và đạt chất lượng cao.

 <h2>Lời cảm ơn</h2>

Chúng em xin cảm ơn TS. Nguyễn Mạnh Cường đã tận tình hướng dẫn và hỗ trợ nhóm trong suốt quá trình thực hiện đồ án.
