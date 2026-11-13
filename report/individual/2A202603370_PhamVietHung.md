# Báo cáo cá nhân — Day 6

- **Học viên**: Phạm Việt Hưng (`2A202603370` — Lớp `K5-L3A`)
- **Thời điểm chạy chính thức**: `2026-11-13T12:09:32Z` → `2026-11-13T12:09:41Z` (UTC)

---

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang**:
  - Chủ đề: Bảo mật ứng dụng Mô hình Ngôn ngữ Lớn (*OWASP Top 10 for LLM Applications 2025*, header v2.0, bản phát hành `2024-11-18`).
  - Nguồn & Giấy phép: Tài liệu PDF công khai tại [OWASP GenAI](https://owasp.org/www-project-top-10-for-large-language-model-applications/assets/PDF/OWASP-Top-10-for-LLMs-v2025.pdf), lưu bản cục bộ ở `starter/data/OWASP-Top-10-for-LLMs-v2025.pdf`, cấp phép **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**, không chứa dữ liệu cá nhân hay mật.
  - Quy tắc tính trang (ghi chi tiết tại [data/manifest.json](../../data/manifest.json)): Đếm trang vật lý PDF (từ `1` đến `45`) có nội dung kỹ thuật thực chất sau chuẩn hóa. Giữ lại **`36/45` trang** (`78,862` ký tự chuẩn hóa); loại **`9` trang** không đạt nội dung gồm: trang bìa (`p.1`), bản quyền & lịch sử sửa đổi (`p.2`), mục lục (`p.3–4`), lời cảm ơn/chữ ký trưởng dự án (`p.6`), trang tràn mục tham khảo chỉ có 2 dòng bullet (`p.14`), phụ lục chỉ có hình ảnh không trích được chữ (`p.43`), và hai trang nhà tài trợ bị lỗi bảng mã font (`p.44–45`). Trang `p.5` (*Letter / What's New in the 2025 Top 10*) được giữ vì chứa ngữ cảnh thay đổi phiên bản 2025.

- **Chuẩn hóa, lỗi và giới hạn**:
  - `starter/src/ingestion.py` chuẩn hóa Unicode **NFKC**, thay khoảng trắng không ngắt `\u00a0`, khử soft-hyphen (`\u00ad`) và ký tự vỡ font `\ufffd`, bỏ header/footer lặp lại (`OWASP Top 10...`, `genai.owasp.org`, số trang rời), và khôi phục ranh giới đoạn `\n\n` giữa tiêu đề mục và bullet.
  - Lỗi trích xuất được log tại [starter/results/ingestion_log.json](../../starter/results/ingestion_log.json): `pypdf 6.21.4` ghi nhận cảnh báo `incorrect startxref pointer(1)` và `Error -3 while decompressing data` trên một số stream ảnh/font; `p.43` chỉ có `raw_chars < 30` (`empty_or_image_only`) và `p.44–45` có `suspicious_ratio > 0.10` (`garbled_encoding`), cả 3 trang đều đã được loại tự động. Không chạy OCR nên nội dung chữ trong sơ đồ ảnh (`p.43`) không nằm trong phạm vi tìm kiếm.

- **Embedding model/provider/version/số chiều, vector store/version, metric, lệnh chạy và chi phí**:
  - Môi trường: `Python 3.13.12`, `openai 3.50.0`, `psycopg 3.3.8`, `pypdf 6.21.4`.
  - Embedding model: **OpenAI `text-embedding-3-small`** (`1536` chiều), giá `$0.02 / 1,000,000 tokens`, dùng chung một hàm/model cho cả document chunks (batch `64`) và queries; fail-fast nếu thiếu `OPENAI_API_KEY` (tuyệt đối không fallback sang mock).
  - Vector store & Metric: **PostgreSQL `16.16` + `pgvector 0.8.7`** chạy qua Docker (`day6_student_starter-postgres-1`, `127.0.0.1:55433/day6_lab`), bảng `document_chunks` với khóa chính `(strategy, chunk_id)`, tìm kiếm vector **exact sequential scan** theo **cosine distance `<=>`** (`score = 1.0 - (embedding <=> query_vec)`).
  - Kiểm chứng nạp lại idempotent: Khi upsert lại toàn bộ `190` chunk của cấu hình `A`, số dòng trong DB giữ nguyên **`190 == 190`**, không phát sinh bản ghi trùng; cơ chế xóa stale chunk tự động dọn chunk cũ cùng `(strategy, document_id)` trong cùng transaction.
  - Lệnh chạy tái lập từ thư mục `starter/`:
    - Kiểm tra hạ tầng: `.\.venv\Scripts\python.exe scripts/check_setup.py`
    - Kiểm thử từng module: `.\.venv\Scripts\python.exe scripts/verify_{ingestion,benchmark,chunking,embeddings,store,evaluation}.py`
    - Chạy pipeline đầy đủ: `.\.venv\Scripts\python.exe -m src.run`
  - Tổng lượng sử dụng & chi phí OpenAI API cho cả lần chạy: **`32,044` tokens** (`$0.000641 USD`), gồm `A` (`16,644` tokens / `$0.000333`), `B` (`14,971` tokens / `$0.000299`), `C` tái sử dụng vector của `B` (`0` tokens / `$0`), và `20` câu query (`429` tokens / `$0.000009`).

- **Metadata bắt buộc, trường lọc bổ sung, nguồn giá trị và lý do chọn**:
  - Metadata truy vết bắt buộc trên mọi chunk: `strategy` (`str`), `document_id` (`str`), `source` (`str`), `page` (`int`, trang vật lý PDF), kèm `risk_id` (`OVERVIEW`, `LLM01`–`LLM10`), `risk_title` và `printed_page`.
  - Trường lọc bổ sung: **`section_type`** (`overview`, `description`, `examples`, `mitigation`, `scenarios`, `references`, `frameworks`), trích xuất trực tiếp từ các heading cấp 1 chuẩn hóa trong PDF và giữ ngữ cảnh liên trang qua `chunk_corpus(..., strategy="by_paragraph")`.
  - Lý do chọn & hữu dụng: Trong tài liệu OWASP, cùng một rủi ro thường lặp từ khóa ở nhiều phần khác nhau (ví dụ thuật ngữ *"mailbox"* vừa có trong `mitigation` p.27 vừa có trong `scenarios` p.29; *"multi-tenant"* vừa có trong `examples` p.33 vừa có trong `mitigation` p.34–35). Lọc theo `section_type` giúp truy xuất đúng loại thông tin người dùng cần.
  - Hạn chế: (1) Một số kịch bản trong mục `scenarios` có tiểu mục `Mitigation` con nằm lồng bên trong (như Scenario #2 trang `p.35`), nhưng `section_type` chỉ gán theo heading cấp 1 nên bộ lọc `section_type="mitigation"` sẽ loại bỏ các đoạn `Mitigation` nằm trong `scenarios`; (2) `section_type` chỉ được gắn khi chunking theo đoạn (`B`, `C`), còn `fixed_size` (`A`) để `null` do ranh giới cửa sổ cắt xuyên qua nhiều heading.

---

## Cấu hình và benchmark

- **Ba cấu hình đối chứng (`A`, `B`, `C`)**:
  - **A (Baseline fixed-size, không lọc)**: `fixed_size(size=500, overlap=50)`, lưu metadata truy vết cơ bản (`section_type=null`), không bật SQL filter. Tạo `190` chunks (trung vị `500` ký tự).
  - **B (Structure-aware theo đoạn, không lọc)**: `by_paragraph(max_size=1050, overlap=0)`, gắn `section_type` vào metadata nhưng **không bật lọc khi search**. Tạo `122` chunks (min `28`, trung vị `713`, max `1,048` ký tự). **So với A chỉ đổi chiến lược chia chunk.**
  - **C (Structure-aware + metadata filter)**: Dùng **chính xác cùng 122 chunks và vector như B**, chỉ **bật lọc SQL tham số hóa `section_type`** cho những câu hỏi có trường `filter` trong benchmark (với câu có `filter: null`, C chạy không lọc như B). **So với B chỉ đổi duy nhất cơ chế metadata filter khi truy xuất.**

- **Bộ benchmark (`≥20` câu) & kiểm chứng evidence**:
  - `20` câu tại [benchmark/questions.json](../../benchmark/questions.json), chia trước theo chủ đề (`risk_id`) để đánh giá holdout topic: **`15` câu `dev`** (`OVERVIEW`, `LLM01–05`, `LLM07`, `LLM09`) và **`5` câu `test`** (`LLM06`, `LLM08`, `LLM10` chưa xuất hiện trong `dev`). Nguy cơ rò rỉ còn lại: `dev` và `test` vẫn chia sẻ cùng mẫu cấu trúc tài liệu OWASP (`Description`, `Examples`, `Mitigation`, `Scenarios`).
  - Có **`6` câu kiểm tra metadata filter** (`q03, q07, q09, q11` trong `dev`; `q17, q18` trong `test`) với `filter` là ngữ cảnh giả định trước khi truy xuất (tuyệt đối không đưa `gold_risk_id`, `gold_page` hay đáp án vào query/filter).
  - Trước khi chạy thử bất kỳ chiến lược nào, `verify_benchmark.py` đã xác minh **`20/20`** `evidence_quote` (cùng trang thay thế/quote hợp lệ khi có nhiều nguồn) và **`10/10`** `coherence_positions` khớp nguyên văn trên văn bản đã chuẩn hóa.

- **Tiêu chí chọn chiến lược đã chốt trước khi xem test**:
  1. Ưu tiên cao nhất cho **`Evidence Hit@3`** (và `Filter Compliance` trên các câu có filter) trên tập `dev`;
  2. Kế đến là **`Grounding quality` / `Chunk coherence`** (không cắt gãy ngữ nghĩa);
  3. Sau cùng là **chi phí token và số lượng chunk**.

---

## Kết quả

### Bảng 1 · Chỉ số truy xuất tự động, quy mô chunk, thời gian và chi phí

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian (ingest + search) | Chi phí OpenAI (ingest) |
|---|---|---:|---:|---:|---:|---:|
| **A** | Dev | `5/15 (33.33%)` | `7/15 (46.67%)` | 190 | `3.009s + 0.441s` | `$0.000333` (`16,644 tok`) |
| **A** | Test | `2/5 (40.00%)` | `2/5 (40.00%)` | 190 | — | — |
| **B** | Dev | `12/15 (80.00%)` | `14/15 (93.33%)` | 122 | `1.602s + 0.523s` | `$0.000299` (`14,971 tok`) |
| **B** | Test | `3/5 (60.00%)` | `3/5 (60.00%)` | 122 | — | — |
| **C** | Dev | `12/15 (80.00%)` | `14/15 (93.33%)` | 122 | `0.193s (reuse B) + 0.424s` | `$0.000000` (dùng chung vector B) |
| **C** | Test | `3/5 (60.00%)` | `3/5 (60.00%)` | 122 | — | — |

*(Ghi chú thêm từ `metrics_summary.json`: `Page Hit@3` trên Dev đạt `14/15` ở cả A, B, C; trên Test đạt `4/5` ở cả A, B, C. Chi phí embed 20 câu query dùng chung là `$0.000009` cho `429` tokens.)*

### Bảng 2 · Kết quả đánh giá thủ công 5 khía cạnh (`Coherence`, `Metadata`, `Filter`, `Grounding`)

| Strategy | Coherence trung bình /2 (`n=10`) | Metadata đủ và đúng (`n/15` hit test) | Filter tuân thủ (`n/N` Dev / Test) | Grounding `0/1/2` (số câu Test) và trung bình /2 |
|---|---:|---|---|---|
| **A** | **`0.70 / 2`** (`0`: 4, `1`: 5, `2`: 1) | `0/15` đủ trường lọc (`15/15` đúng `source/page`, nhưng thiếu `section_type=null`) | `0/4` (Dev) / `0/2` (Test) | `0`: 2, `1`: 1, `2`: 2 · **TB: `1.00 / 2`** |
| **B** | **`1.90 / 2`** (`0`: 0, `1`: 1, `2`: 9) | `15/15` (đủ & đúng `source, page, section_type`) | `1/4` (Dev) / `1/2` (Test) | `0`: 1, `1`: 1, `2`: 3 · **TB: `1.40 / 2`** |
| **C** | **`1.90 / 2`** (`0`: 0, `1`: 1, `2`: 9) | `15/15` (đủ & đúng `source, page, section_type`) | **`4/4` (Dev) / `2/2` (Test)** | `0`: 1, `1`: 1, `2`: 3 · **TB: `1.40 / 2`** |

### Đường dẫn trace, nhãn thủ công và cách chọn mẫu
- Trace đầy đủ top-3: [traces_A.json](../../starter/results/traces_A.json), [traces_B.json](../../starter/results/traces_B.json), [traces_C.json](../../starter/results/traces_C.json), [traces_all.json](../../starter/results/traces_all.json)
- Nhãn chấm thủ công Coherence (`10` vị trí neo pre-locked, trải đều `LLM01–LLM10` từ `pos01` đến `pos10` trước khi chạy thí nghiệm): [coherence_samples.json](../../starter/results/coherence_samples.json)
- Nhãn chấm thủ công Grounding & Metadata (`5` câu test `q16–q20` × top-3 × 3 chiến lược = `45` bản ghi): [test_audit_samples.json](../../starter/results/test_audit_samples.json)
- **Phương pháp & thiên lệch chấm thủ công**: Nhãn Coherence và Grounding/Metadata được trợ lý AI đọc text thực tế đề xuất sơ bộ và tác giả bài làm (Phạm Việt Hưng) rà soát xác nhận từng mẫu. Vì tác giả vừa thiết kế benchmark vừa nghiệm thu, có nguy cơ thiên lệch ưu ái cấu hình theo đoạn (`B`/`C`); để giảm thiểu, mọi điểm số đều gắn lý do cụ thể đối chiếu trực tiếp text chunk và giữ nguyên các lỗi mất ý của `B`/`C` (như `pos06` chỉ `1/2` và `q16`, `q17` lần lượt `0/2` và `1/2`).

---

## Phân tích và kết luận

### 1. So sánh A–B (tác động của Chunking) và B–C (tác động của Metadata Filter)
- **A → B (đổi `fixed_size(500, 50)` sang `by_paragraph(1050)`)**:
  - Tuy **tỷ lệ tìm đúng trang nguồn** (`Page Hit@3`) gần như giống hệt nhau (`14/15` Dev, `4/5` Test), `Evidence Hit@3` của `A` chỉ đạt `7/15 (46.67%)` trên Dev và `2/5 (40.00%)` trên Test, trong khi `B` tăng vọt lên **`14/15 (93.33%)`** trên Dev và **`3/5 (60.00%)`** trên Test (`Hit@1` Dev tăng từ `33.33%` lên `80.00%`).
  - Nguyên nhân trực tiếp nằm ở **Chunk Coherence**: điểm trung bình của `A` chỉ đạt **`0.70/2`** (4/10 vị trí neo như `pos04, pos05, pos06, pos08` bị cắt ngang giữa câu làm mất chủ ngữ/vị ngữ), khiến bằng chứng bị đứt khúc hoặc mất ngữ cảnh câu hỏi nên vector similarity giảm mạnh. Ngược lại `B` đạt **`1.90/2`** và còn giảm `35.8%` số chunk (`190 → 122`), giúp giảm ~10% token/chi phí embed (`16,644 → 14,971` tokens).
- **B → C (giữ nguyên chunk B, chỉ bật SQL filter `section_type`)**:
  - `Evidence Hit@3` giữ nguyên `14/15` (Dev) và `3/5` (Test), nhưng **tuân thủ scope section tăng từ `1/4 → 4/4` (Dev) và `1/2 → 2/2` (Test)**.
  - Ở `q18` (Test, hỏi bước kiểm soát truy cập trong `section_type="mitigation"`), `B` kéo về Rank 1 từ `scenarios` (`p35:c001`, chứa tiểu mục mitigation trong kịch bản) và Rank 3 từ `examples` (`p33:c002`). Khi bật filter ở `C`, Rank 1 chuyển chính xác sang mục phòng ngừa chính thức (`p34:c002`, `mitigation`) và toàn bộ top-3 đều thuộc `mitigation` (`LLM08`, `LLM04`, `LLM01`).

### 2. Hai ví dụ lỗi thực tế tiêu biểu
1. **Ví dụ lỗi cắt biên chunk & tràn ngưỡng `max_size` — Câu `q17` (Test, `LLM06`, `starter/data/OWASP-Top-10-for-LLMs-v2025.pdf` trang `29`, mục `scenarios`)**:
   - Câu hỏi yêu cầu 3 biện pháp ngăn chặn indirect prompt injection trên trợ lý email cá nhân.
   - Ở chiến lược **A**, `fixed:p29:c001` (`500` ký tự) bị cắt ngay giữa đoạn mô tả kịch bản (`"...maliciously-crafted incoming email tricks t"`), chưa chạm tới bất kỳ biện pháp khắc phục nào → `Grounding = 0/2`.
   - Ở chiến lược **B & C**, `para:p29:c001` (`900` ký tự) lấy trọn kịch bản và 2 gạch đầu dòng đầu (*mail-reading only*, *OAuth read-only scope*), nhưng bullet thứ 3 (*manual review before sending*) đẩy tổng độ dài vượt `max_size=1050` nên bị tách sang `para:p29:c002` (không vào top-3). Hệ quả: **B & C trượt `Evidence Hit@3`** (Grounding thực tế đạt `1/2`, thiếu 1/3 ý).
2. **Ví dụ lỗi nhiễu ngữ nghĩa tương đồng (semantic confounder) — Câu `q16` (Test, `LLM06`, trang `28`) & Câu `q04` (Dev, `LLM02`, trang `11`)**:
   - Ở **`q16`** (*"Which two controls do not prevent Excessive Agency directly, yet help limit the damage..."*), cả A, B và C đều bỏ lỡ trang `p.28` và ưu tiên trả về đoạn `p.27` (*"The following actions can prevent Excessive Agency..."*) cùng phần mô tả `p.26`, vì embedding dày đặc khớp cụm *"prevent Excessive Agency"* mạnh hơn cụm phủ định *"will not prevent... limit damage"* → `Grounding = 0/2` trên cả 3 chiến lược.
   - Tương tự ở **`q04`** (`Dev`, hỏi về hạn chế của system prompt trong `LLM02` p.11), cả 3 chiến lược đều kéo nhầm sang trang `p.30–31` của `LLM07: System Prompt Leakage` do độ tương đồng từ vựng cao quanh khái niệm *"system prompt"*.

### 3. Chiến lược chọn, đánh đổi và điều kiện áp dụng
- **Chiến lược được chọn: `C` (`by_paragraph(max_size=1050)` + `section_type` SQL filter có điều kiện)**.
- Đạt đúng tiêu chí chốt trước: dẫn đầu `Evidence Hit@3` trên Dev (`14/15`), đạt `100%` `Filter Compliance` (`4/4` Dev, `2/2` Test), `Coherence 1.90/2`, `Grounding Test 1.40/2`, đồng thời số chunk ít hơn (`122` vs `190`) và chi phí embed thấp hơn `A`.
- **Đánh đổi & điều kiện áp dụng**: `C` phụ thuộc vào giả định metadata `section_type` do hệ thống/người dùng xác định chính xác trước khi tìm kiếm; nếu gán nhầm filter (hoặc thông tin cần tìm nằm trong tiểu mục lồng như `Mitigation` bên trong `scenarios` ở `p.35`), bộ lọc cứng (`WHERE section_type = ...`) sẽ loại bỏ ứng viên hợp lệ.

### 4. Hạn chế của dữ liệu, nhãn, test nhỏ và đo tài nguyên
- **Tập test rất nhỏ (`N = 5`, holdout 3 chủ đề `LLM06, LLM08, LLM10`)**: Mỗi câu thay đổi tương ứng `20%` tỷ lệ; không đủ để kết luận thống kê rộng hơn phạm vi corpus OWASP LLM 2025.
- **Độ phủ nguồn & lỗi PDF**: Trang `p.43` dạng ảnh sơ đồ không được OCR nên vắng mặt khỏi kho vector. Ngoài ra, chunking theo từng trang độc lập làm 2 đoạn văn vắt qua ranh giới trang PDF có thể mất một phần ngữ cảnh nếu không ghép liên trang ở mức câu văn.
- **Thiên lệch chấm điểm**: Tác giả tự rà soát nhãn dựa trên đề xuất của AI trên mẫu nhỏ; độ trễ API OpenAI và Docker local (`~0.4–3.0s`) mang tính tham khảo tại thời điểm chạy chứ không phản ánh tải production.

### 5. Nếu làm lại hoặc mở rộng, cải tiến gì?
1. **Kết hợp Hybrid Search (Dense cosine + BM25 / PostgreSQL `tsvector`) & Reranking**: Giúp phân biệt các mệnh đề phủ định/điều kiện như `q16` (*"do not prevent... limit damage"*) và tách ngữ cảnh `LLM02` vs `LLM07` (`q04`).
2. **Gắn metadata đa nhãn hoặc phân cấp (`section_type` + `has_mitigation` flag) & nới biên chunk thông minh**: Cho phép `max_size` linh hoạt thêm `~150–200` ký tự khi đang dở một danh sách bullet ngắn (như `q17` ở `p.29`) hoặc áp dụng overlapping giữa các đoạn liên tiếp trong cùng section.
3. **Mở rộng bộ benchmark độc lập**: Tăng số câu `test` lên `≥15–20` câu thuộc nhiều tài liệu bảo mật AI khác nhau (ví dụ NIST AI RMF, MITRE ATLAS) để kiểm chứng khả năng tổng quát hóa ngoài mẫu OWASP.
