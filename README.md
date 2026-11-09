# Food Review - Làng Đại học (ĐHQG TP.HCM)

> **Đồ án môn học:** IE104 - Thiết kế Giao diện Người dùng  
> **Chủ đề:** Website Review & Khám phá địa điểm ăn uống quanh Làng Đại học (ĐHQG-HCM)

---

## 🌟 Giới thiệu Dự án

**Food Review - Làng Đại học** là nền tảng thông tin và cộng đồng đánh giá địa điểm ăn uống dành riêng cho sinh viên và cư dân tại khu vực Làng Đại học Quốc gia TP.HCM.

Hệ thống tập trung giải quyết 3 câu hỏi thiết thực:
1. *Ăn ở đâu?* (Tìm theo khu vực KTX A, KTX B, Chợ Đêm, Dốc Quốc Tế...)
2. *Giá khoảng bao nhiêu?* (Lọc theo ngân sách sinh viên)
3. *Người trải nghiệm trước nhận xét thế nào?* (Đánh giá thực tế kèm hình ảnh món ăn)

---

## 🏗️ Kiến trúc Công nghệ (Backend)

- **Platform:** .NET 9 SDK (C# 13)
- **Framework:** ASP.NET Core Web API
- **Data Access Layer:** **Dapper Micro-ORM** (High-Performance Raw SQL & Multi-Mapping)
- **Database Engine:** SQLite (khởi tạo từ `docs/schema.sql`), hỗ trợ chuyển đổi sang PostgreSQL / SQL Server linh hoạt.
- **Bảo mật & Phân quyền:** JWT Bearer Authentication, băm mật khẩu BCrypt, RBAC (Guest, User, Admin).
- **Tài liệu API:** Swagger / OpenAPI UI tương tác trực tiếp (`/swagger`).

---

## 📁 Cấu trúc Thư mục

```
IE104-FoodReview-VNU/
├── docs/                                  # Tài liệu kiến trúc & đặc tả
│   ├── prd.md                             # Yêu cầu sản phẩm (PRD)
│   ├── ARCHITECTURE-SPINE.md              # Thiết kế kiến trúc & ERD
│   ├── DATA_COLLECTION_GUIDE.md           # Hướng dẫn thu thập dữ liệu & hình ảnh
│   └── schema.sql                         # Single Source of Truth DDL
├── backend/                               # Mã nguồn Backend
│   └── FoodReview.Api/                    # ASP.NET Core 9 Web API
│       ├── Controllers/                   # RESTful API Endpoints
│       ├── Models/                        # Domain Entities
│       ├── DTOs/                          # Request / Response Models
│       ├── Services/                      # Business Logic (Dapper)
│       ├── Data/                          # Connection Factory & Seed Data
│       └── Program.cs                     # Startup Pipeline
├── frontend/                              # Frontend thử nghiệm API (HTML/CSS/JS thuần)
│   ├── index.html
│   ├── css/styles.css
│   ├── js/                                # config, api, auth, places, reviews, admin
│   └── README.md                          # Hướng dẫn chạy & danh sách API đã nối
├── GEMINI.md                              # Behavioral guidelines cho AI Pair Programming
└── README.md
```

---

## 🚀 Hướng dẫn Cài đặt & Chạy Backend

### 1. Yêu cầu môi trường
- [.NET 9 SDK](https://dotnet.microsoft.com/download/dotnet/9.0) trở lên.

### 2. Khởi chạy API
```bash
# Di chuyển vào thư mục Web API
cd backend/FoodReview.Api

# Chạy ứng dụng (database sẽ tự động khởi tạo và seed dữ liệu mẫu)
dotnet run
```

### 3. Trải nghiệm Swagger API
Sau khi server khởi chạy, mở trình duyệt truy cập:
👉 **`http://localhost:5114/swagger`**

Tài khoản thử nghiệm sẵn có:
- **Admin:** `admin@langdaihoc.vn` / `Admin@123`
- **Sinh viên:** `sinhvien@uit.edu.vn` / `Student@123`

---

## 🖥️ Frontend thử nghiệm API

Thư mục `frontend/` chứa giao diện HTML + CSS + JavaScript thuần (không cần build) để thử toàn bộ API: đăng nhập, khám phá/lọc quán, chi tiết, yêu thích, đánh giá kèm upload ảnh và bảng quản trị.

```bash
# Terminal 1 — backend
cd backend/FoodReview.Api
dotnet run

# Terminal 2 — static server cho frontend
cd frontend
python -m http.server 5500
```

Mở 👉 **`http://localhost:5500`**. Nếu backend chạy ở cổng khác, cập nhật `API_BASE_URL` trong `frontend/js/config.js`.

Chi tiết cấu trúc và danh sách endpoint đã nối: [`frontend/README.md`](frontend/README.md).
