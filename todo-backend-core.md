# TODO BACKEND CORE (.NET 9 WEB API)
Theo chuẩn BMAD Graph-Driven Execution

- [x] [EPIC-0] Khởi tạo Project ASP.NET Core 9 Web API, Cấu hình EF Core SQLite, JWT, Swagger, CORS
- [x] [SVC-1] Models & DbContext — Định nghĩa Entities (User, Area, Category, Place, MenuItem, Review, ReviewImage, Report, Favorite)
- [x] [SVC-2] AuthService & AuthController — Đăng ký, Đăng nhập, JWT Token, GetMe, UpdateProfile, Phân quyền RBAC
- [x] [SVC-3] CategoryAreaService & Controllers — Danh mục món ăn & Khu vực Làng Đại học
- [x] [SVC-4] PlaceService & PlacesController — Tìm kiếm, Lọc (khu vực, giá, món), Chi tiết quán, Đánh giá trung bình
- [x] [SVC-5] ReviewService & ReviewsController — Đăng review, sửa/xóa review của mình, tính điểm rating
- [x] [SVC-6] InteractionService (Reports & Favorites) — Báo cáo review vi phạm, Lưu quán yêu thích
- [x] [SVC-7] AdminService & AdminController — Kiểm duyệt review, xử lý báo cáo, quản lý người dùng & quán
- [x] [SVC-8] FileStorageService & UploadController — Upload ảnh và cấu hình static files
- [x] [SVC-9] DbInitializer & Mock Data — Bộ dữ liệu mồi ban đầu + cơ chế import dữ liệu thật khi team thu thập xong
