# PRODUCT REQUIREMENTS DOCUMENT (PRD)
## WEBSITE REVIEW & KHÁM PHÁ ĐỊA ĐIỂM ĂN UỐNG QUANH LÀNG ĐẠI HỌC (ĐHQG TP.HCM)

---

### 1. Tóm tắt sản phẩm
- **Tên dự án:** Food Review - Làng Đại học (ĐHQG TP.HCM).
- **Mục tiêu:** Nền tảng thông tin và cộng đồng review giúp sinh viên và người sinh hoạt tại khu vực Làng Đại học tìm kiếm, khám phá và đánh giá các địa điểm ăn uống.
- **Giá trị cốt lõi:** Trả lời 3 câu hỏi:
  1. *Ăn ở đâu?*
  2. *Giá khoảng bao nhiêu?*
  3. *Người đã trải nghiệm nhận xét như thế nào?*
- **Phạm vi:** Nền tảng thông tin & đánh giá cộng đồng (Không bao gồm đặt món, thanh toán hay giao đồ ăn).

---

### 2. Đối tượng người dùng & Phân quyền (RBAC)
1. **Khách (Guest / Chưa đăng nhập):**
   - Tìm kiếm, lọc và xem danh sách địa điểm ăn uống.
   - Xem trang chi tiết địa điểm, menu tham khảo, điểm đánh giá trung bình và các bài review công khai.
   - Báo cáo (Report) review vi phạm.
2. **Người dùng (Authenticated User / Sinh viên):**
   - Toàn bộ quyền của Guest.
   - Đăng bài đánh giá (Review) kèm điểm số (1-5 sao), bình luận và tải ảnh trải nghiệm thực tế.
   - Chỉnh sửa hoặc xóa bài review của chính mình.
   - Thích (Like) review của người khác, lưu (Bookmark/Favorite) địa điểm yêu thích.
   - Cập nhật thông tin cá nhân (Profile, Avatar).
3. **Quản trị viên (Admin):**
   - Quản lý danh mục (Category) và khu vực (Area) Làng Đại học.
   - Thêm, sửa, ẩn/khôi phục thông tin địa điểm ăn uống (Place).
   - Kiểm duyệt review (duyệt, ẩn bài vi phạm tiêu chuẩn cộng đồng).
   - Xử lý các báo cáo vi phạm (Report queue).
   - Quản lý tài khoản người dùng (Khóa/Mở khóa), xem nhật ký thao tác (Audit log).

---

### 3. Mô hình Dữ liệu & Thực thể Cốt lõi (Entities)

| Thực thể | Thuộc tính chính | Mô tả |
|---|---|---|
| **User** | `UserId`, `Email`, `PasswordHash`, `DisplayName`, `AvatarUrl`, `Role` (User, Admin), `Status`, `CreatedAt` | Tài khoản người dùng |
| **Area** | `AreaId`, `Name`, `Description`, `ParentAreaId` | Cụm khu vực (KTX A, KTX B, Chợ đêm, ĐH Quốc Tế...) |
| **Category** | `CategoryId`, `Name`, `Slug`, `Icon`, `Status` | Loại món / Loại quán (Cơm, Bún, Trà sữa, Ăn vặt...) |
| **Place** | `PlaceId`, `Name`, `Slug`, `Description`, `Address`, `AreaId`, `CategoryId`, `PriceMin`, `PriceMax`, `OpeningHours`, `Latitude`, `Longitude`, `Status` (Draft, Published, Hidden), `CreatedAt`, `UpdatedAt` | Địa điểm ăn uống |
| **MenuItem** *(Optional)* | `MenuItemId`, `PlaceId`, `Name`, `Price`, `Note` | Món ăn tiêu biểu của quán |
| **Review** | `ReviewId`, `PlaceId`, `UserId`, `Rating` (1-5), `Content`, `Status` (Published, Hidden, Pending), `CreatedAt`, `UpdatedAt` | Bài đánh giá |
| **ReviewImage** | `ImageId`, `ReviewId`, `ImageUrl`, `AltText`, `CreatedAt` | Hình ảnh trải nghiệm đi kèm review |
| **Report** | `ReportId`, `ReporterId`, `ReviewId`, `Reason`, `Description`, `Status` (Pending, Resolved, Dismissed), `HandledBy`, `CreatedAt`, `HandledAt` | Báo cáo nội dung vi phạm |
| **Favorite** | `UserId`, `PlaceId`, `CreatedAt` | Địa điểm yêu thích của sinh viên |

---

### 4. Quy tắc nghiệp vụ cốt lõi (Business Rules)
1. **Đánh giá & Điểm số:**
   - Mỗi review thuộc về đúng 1 User và 1 Place.
   - Điểm số hợp lệ: Số nguyên từ 1 đến 5 sao.
   - Điểm trung bình (`AverageRating`) và tổng số đánh giá (`ReviewCount`) của quán chỉ tính trên các review có trạng thái `Published`.
   - Nếu quán chưa có đánh giá nào, hiển thị "Chưa có đánh giá" (không hiển thị 0/5).
2. **Quyền sở hữu:**
   - User chỉ được sửa hoặc xóa review của chính mình.
   - Admin không sửa nội dung của tác giả mà chỉ có quyền ẩn/kiểm duyệt.
3. **Trạng thái hiển thị:**
   - Chỉ địa điểm có trạng thái `Published` mới xuất hiện trên kết quả tìm kiếm và danh sách công khai.
   - Báo cáo vi phạm có trạng thái xử lý rõ ràng: `Pending` $\rightarrow$ `Resolved` / `Dismissed`.
4. **Phân trang & Lọc:**
   - Tìm kiếm theo từ khóa (tên quán, món ăn), lọc kết hợp (Khu vực, Khoảng giá, Thể loại).
   - Phân trang chuẩn phân đoạn `pageIndex` và `pageSize` để tối ưu tải trang.

