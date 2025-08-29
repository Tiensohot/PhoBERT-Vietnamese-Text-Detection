Nhận diện văn bản tiếng Việt sinh bởi mô hình ngôn ngữ lớn (LLM) sử dụng PhoBERT và học sâu
📌 Giới thiệu

Đây là đồ án chuyên ngành thuộc Ngành Khoa học Máy tính – Trường Đại học Công nghiệp Hà Nội.
Đề tài tập trung xây dựng hệ thống có khả năng phân biệt văn bản do con người viết và văn bản sinh ra bởi AI (Large Language Models – LLM).

Chúng tôi sử dụng PhoBERT, một mô hình ngôn ngữ tiền huấn luyện cho tiếng Việt, kết hợp với Deep Learning (PyTorch) để xây dựng bộ phân loại văn bản.
Sản phẩm cuối cùng gồm:

Mô hình huấn luyện (final.ipynb, best_model.pt).

Ứng dụng web demo với Flask + HTML/CSS/JS để giảng viên và người dùng có thể trực tiếp thử nghiệm.

🛠️ Công nghệ sử dụng

Đề tài áp dụng nhiều kiến thức liên quan trực tiếp đến các môn học đã được giảng dạy trong chương trình:

Xử lý ngôn ngữ tự nhiên (NLP) → áp dụng PhoBERT tokenizer, embedding, classification.

Trí tuệ nhân tạo (AI) → sử dụng mô hình học sâu để phân loại văn bản.

Học sâu (Deep Learning) → fine-tune PhoBERT với PyTorch.

Kỹ thuật phần mềm & Lập trình Web → xây dựng ứng dụng demo bằng Flask, HTML, CSS, JS.

Cơ sở dữ liệu & Khoa học dữ liệu → xử lý tập dữ liệu văn bản (1350 mẫu, gán nhãn Human/AI).

📂 Cấu trúc thư mục
PhoBERT-Text-Detection/
│── app.py              # Flask server (chạy demo)
│── model.py            # Định nghĩa mô hình PhoBERT + classifier
│── best_model.pt       # Trọng số mô hình đã huấn luyện
│── final.ipynb         # Notebook huấn luyện, tiền xử lý, đánh giá
│── templates/
│   └── index.html      # Giao diện demo
│── DACN_nhom09.docx    # Báo cáo đồ án
│── requirements.txt    # Thư viện cần cài đặt

▶️ Cách chạy demo

Clone repo:

git clone https://github.com/<tenuser>/PhoBERT-Text-Detection.git
cd PhoBERT-Text-Detection


Cài đặt thư viện:

pip install -r requirements.txt


Chạy ứng dụng Flask:

python app.py


Mở trình duyệt tại: http://127.0.0.1:5000

→ Dán văn bản hoặc upload file .txt/.docx để phân tích.

📊 Kết quả nổi bật

Độ chính xác mô hình (Accuracy): 99.6% trên tập kiểm thử.

F1-score: 0.9963 (cân bằng tốt giữa Human và AI).

So sánh với baseline (TF-IDF + SVM, Rule-based) → PhoBERT vượt trội.

👨‍👩‍👦‍👦 Đóng góp của nhóm

Đồ án là thành quả teamwork nghiêm túc của Nhóm 09, mỗi thành viên đều đảm nhận một phần quan trọng:

Nguyễn Huy Cương – Tiền xử lý dữ liệu, huấn luyện mô hình.

Đào Văn Hiệp – Phân tích thuật toán, viết báo cáo.

Bùi Đức Tiến – Xây dựng giao diện demo, tích hợp hệ thống.

Chúng tôi đã làm việc nhóm liên tục trong nhiều tuần, từ khâu tìm hiểu lý thuyết – triển khai mô hình – đánh giá – xây dựng ứng dụng demo – viết báo cáo.
Sự phối hợp ăn ý giữa các thành viên là yếu tố quyết định giúp đồ án hoàn thiện đúng tiến độ và đạt chất lượng cao.

🙏 Lời cảm ơn

Nhóm xin gửi lời cảm ơn sâu sắc đến TS. Nguyễn Mạnh Cường đã tận tình hướng dẫn, định hướng và góp ý giúp nhóm vượt qua khó khăn để hoàn thiện đề tài này.
