# Day 06 · Data Ingestion Engineering

**Lab cá nhân** · **4 giờ** · **Corpus: OWASP Top 10 for LLM Applications 2025 (`36/45` trang nội dung)**

Biến tài liệu thành kho tri thức có thể truy xuất cho AI agent, rồi đo xem cách chuẩn bị dữ liệu nào hiệu quả nhất trên PostgreSQL + pgvector và OpenAI Embeddings.

| 📄 Dữ liệu | 🧩 Chiến lược | 🎯 Benchmark |
|---|---|---|
| **36 trang nội dung** (từ 45 trang PDF, CC BY-SA 4.0) | **3 cấu hình (`A`, `B`, `C`)** có đối chứng | **20 câu hỏi** (`15 dev / 5 test`, `6` câu filter) |

---

## Tổng quan kiến trúc & dữ liệu bài làm

- **Corpus & Manifest**: `OWASP-Top-10-for-LLMs-v2025.pdf` tại [starter/data/OWASP-Top-10-for-LLMs-v2025.pdf](starter/data/OWASP-Top-10-for-LLMs-v2025.pdf) và quy tắc lọc trang tại [data/manifest.json](data/manifest.json) (`36` trang giữ lại, `78,862` ký tự chuẩn hóa).
- **Benchmark**: [benchmark/questions.json](benchmark/questions.json) (`15 dev / 5 test` tách theo `risk_id`, holdout `LLM06`, `LLM08`, `LLM10`) và [benchmark/coherence_positions.json](benchmark/coherence_positions.json) (`10` vị trí neo khóa trước).
- **Pipeline (`starter/src/`)**:
  - [ingestion.py](starter/src/ingestion.py): đọc PDF bằng `pypdf`, chẩn đoán trang rỗng/lỗi font, lọc trang nội dung, chuẩn hóa Unicode `NFKC` và gắn metadata `risk_id`, `risk_title`, `printed_page`, `section_types`.
  - [chunking.py](starter/src/chunking.py): `fixed_size(size=500, overlap=50)` (`190` chunks) và `by_paragraph(max_size=1050)` giữ ngữ cảnh `section_type` liên trang (`122` chunks).
  - [embeddings.py](starter/src/embeddings.py): `MyEmbedder` dùng `openai/text-embedding-3-small` (`1536` chiều), theo dõi token/chi phí thực tế (`$0.02 / 1M tokens`), fail-fast nếu thiếu `OPENAI_API_KEY`.
  - [store.py](starter/src/store.py): `PgVectorStore` kết nối PostgreSQL 16.16 + pgvector 0.8.7 (`<=>` cosine distance, exact sequential scan), upsert idempotent theo `(strategy, chunk_id)`, dọn stale chunk cùng transaction, hỗ trợ lọc SQL tham số hóa `section_type`.
  - [evaluation.py](starter/src/evaluation.py): `evaluate_query` và `summarize` đo `Hit@1`, `Evidence Hit@3`, `filter_compliance`, tách riêng `dev`/`test`.
  - [run.py](starter/src/run.py): chạy end-to-end 3 chiến lược `A`, `B`, `C`, kiểm chứng nạp lại không nhân đôi và xuất trace/báo cáo vào `starter/results/` và [artifacts/experiment_log.csv](artifacts/experiment_log.csv).

---

## Hướng dẫn cài đặt và chạy tái lập

### 1. Chuẩn bị môi trường (chạy từ thư mục `starter/`)
```powershell
cd starter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Điền OPENAI_API_KEY vào starter/.env (không commit .env)
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

Phiên bản đã kiểm chứng trong lần chạy chính thức:
- `Python 3.13.12`, `openai 3.50.0`, `psycopg 3.3.8`, `pypdf 6.21.4`
- `PostgreSQL 16.16` + `pgvector 0.8.7` (container `day6_student_starter-postgres-1`, cổng `127.0.0.1:55433`)

### 2. Chạy kiểm thử từng module
```powershell
# Từ thư mục starter/
.\.venv\Scripts\python.exe scripts/verify_ingestion.py
.\.venv\Scripts\python.exe scripts/verify_benchmark.py
.\.venv\Scripts\python.exe scripts/verify_chunking.py
.\.venv\Scripts\python.exe scripts/verify_embeddings.py
.\.venv\Scripts\python.exe scripts/verify_store.py
.\.venv\Scripts\python.exe scripts/verify_evaluation.py
```

### 3. Chạy toàn bộ pipeline và xem kết quả
```powershell
cd starter
.\.venv\Scripts\python.exe -m src.run
```

Tệp kết quả sinh ra sau khi chạy:
- Kết quả tổng hợp & nhật ký: [starter/results/metrics_summary.json](starter/results/metrics_summary.json), [artifacts/experiment_log.csv](artifacts/experiment_log.csv)
- Trace truy xuất top-3 đầy đủ: `starter/results/traces_A.json`, `starter/results/traces_B.json`, `starter/results/traces_C.json`, `starter/results/traces_all.json`
- Nhãn đánh giá thủ công: [starter/results/coherence_samples.json](starter/results/coherence_samples.json), [starter/results/test_audit_samples.json](starter/results/test_audit_samples.json)
- Báo cáo cá nhân: [report/individual/2A202603370_PhamVietHung.md](report/individual/2A202603370_PhamVietHung.md)

---

## Tài liệu đề bài & tiêu chí tham chiếu

| Bạn cần tìm | Tài liệu |
|---|---|
| Setup baseline PostgreSQL + pgvector | [starter/README.md](starter/README.md) |
| Các bước làm và cách tính chỉ số | [GUIDE.md](GUIDE.md) |
| Tiến độ trong 4 giờ | [CHECKPOINTS.md](CHECKPOINTS.md) |
| Năm tiêu chí benchmark và bộ câu hỏi | [BENCHMARK.md](BENCHMARK.md) |
| Tiêu chí chấm điểm | [RUBRIC.md](RUBRIC.md) |
| Quy tắc sử dụng dữ liệu, AI và API | [RULES.md](RULES.md) |
| Cách nộp bài | [SUBMISSION.md](SUBMISSION.md) |
| Hồ sơ học viên | [STUDENT.md](STUDENT.md) |
