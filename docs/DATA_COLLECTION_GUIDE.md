# TÀI LIỆU HƯỚNG DẪN & BIỂU MẪU THU THẬP DỮ LIỆU
## DỰ ÁN: WEBSITE REVIEW ĐỊA ĐIỂM ĂN UỐNG LÀNG ĐẠI HỌC (ĐHQG TP.HCM)

> **Mục đích:** Tài liệu này được thiết kế để bạn chuyển trực tiếp cho các thành viên trong nhóm khảo sát và thu thập dữ liệu (thông tin quán, menu, hình ảnh và review mẫu). Dữ liệu sau khi thu thập theo mẫu này sẽ được nạp trực tiếp vào hệ thống Backend mà không cần chỉnh sửa lại.

---

### PHẦN 1: BẢNG DANH MỤC & KHU VỰC CẦN KHẢO SÁT

Để đảm bảo dữ liệu đồng nhất, nhóm khảo sát chỉ cần chọn đúng các mã danh mục và khu vực chuẩn hóa dưới đây:

#### 1. Khu vực chuẩn quanh Làng Đại học (`Area`)
| Mã khu vực | Tên khu vực | Mô tả & Mốc nhận diện |
|---|---|---|
| `AREA_KTX_A` | KTX Khu A & Cổng KTX A | Cổng chính KTX A, đường A1, khu vực các quán ăn bao quanh KTX A |
| `AREA_KTX_B` | KTX Khu B & Cổng KTX B | Trước cổng KTX B, đường Tô Vĩnh Diện, các trục đường quán ăn KTX B |
| `AREA_CHO_DEM`| Khu Chợ Đêm Làng ĐH | Khu chợ đêm cũ/mới, khu vực nhà văn hóa sinh viên |
| `AREA_DOC_QT` | Dốc ĐH Quốc Tế & ĐH Tự Nhiên | Trục đường dốc đối diện ĐH Quốc Tế, ĐH KHTN |
| `AREA_VANH_DAI`| Trục Đường Vành Đai | Dọc đường vành đai ĐHQG, đoạn gần ĐH Bách Khoa CS2, ĐH CNTT (UIT) |
| `AREA_KHAC` | Lân cận Làng ĐH | Khu vực gần Suối Tiên, Tân Lập, đường 621 |

#### 2. Phân loại danh mục quán (`Category`)
| Mã danh mục | Tên danh mục | Ví dụ điển hình |
|---|---|---|
| `CAT_COM` | Cơm trưa / Cơm sinh viên | Cơm tấm, cơm gà xối mỡ, cơm bình dân, cơm thố |
| `CAT_BUN_PHO` | Bún / Phở / Mì / Hủ tiếu | Bún bò Huế, bún riêu, hủ tiếu gõ, mì cay, mì Quảng |
| `CAT_AN_VAT` | Ăn vặt / Xiên que | Bánh tráng nướng, xiên bẩn, cá viên chiên, ốc, nem nướng |
| `CAT_TRA_SUA` | Trà sữa / Cà phê / Nước uống| Trà sữa sinh viên, cafe cóc, cafe máy lạnh chạy deadline |
| `CAT_LAU_NUONG`| Lẩu & Đồ nướng | Lẩu bò, lẩu gà lá é, đồ nướng sinh viên, buffet giá rẻ |
| `CAT_CHAY` | Quán chay | Cơm chay, bún chay |

---

### PHẦN 2: THÔNG TIN CHI TIẾT CỦA MỖI QUÁN ĂN (`Place`)

Mỗi quán ăn cần thu thập đầy đủ các trường thông tin sau:

1. **Tên quán:** Tên chính xác trên biển hiệu (Ví dụ: *Cơm Tấm Cây Bàng, Bún Bò Huế Chú Tèo*).
2. **Khu vực:** Chọn 1 trong các mã khu vực ở Phần 1.
3. **Danh mục chính:** Chọn 1 trong các mã danh mục ở Phần 1.
4. **Địa chỉ & Mốc chỉ đường:** 
   - Địa chỉ cụ thể hoặc mốc dễ tìm cho sinh viên (Ví dụ: *Số 12 đường A1, đối diện cổng KTX A*).
5. **Khoảng giá (VNĐ):**
   - Giá thấp nhất (`PriceMin`): ví dụ `25000`
   - Giá cao nhất (`PriceMax`): ví dụ `45000`
6. **Giờ hoạt động:** Giờ mở - đóng cửa (Ví dụ: `06:30 - 21:00`).
7. **Đặc điểm nổi bật / Tiện ích (Ghi chú):**
   - Ví dụ: *Cơm thêm miễn phí, có trà đá miễn phí, có máy lạnh, chỗ để xe rộng, đông vào giờ trưa*.
8. **Link / Tọa độ Google Maps (Nếu có):** Link vị trí quán trên Google Maps để làm chức năng chỉ đường.
9. **Thực đơn tiêu biểu (Menu):** Ít nhất 3 - 5 món bán chạy kèm giá:
   - *Món 1: Cơm sườn bì chả - 35.000đ*
   - *Món 2: Cơm đùi gà xối mỡ - 35.000đ*
   - *Món 3: Cơm chiên dương châu - 30.000đ*

---

### PHẦN 3: QUY CHUẨN HÌNH ẢNH (BẮT BUỘC)

Hình ảnh là yếu tố quan trọng nhất giúp giao diện website đẹp và trực quan. Người chụp ảnh cần tuân thủ các quy tắc sau:

#### 1. Các loại ảnh cần chụp cho mỗi quán:
- **Ảnh 1 - Ảnh Mặt tiền / Biển hiệu (Ảnh bìa - Cover Image):**
  - Chụp ngang toàn cảnh mặt tiền, thấy rõ biển hiệu tên quán.
  - Đây sẽ là ảnh đại diện hiển thị trên trang chủ và danh sách tìm kiếm.
- **Ảnh 2 - Ảnh Món ăn đặc trưng:**
  - Chụp cận cảnh đĩa cơm, tô bún, ly nước (đủ sáng, bắt mắt).
- **Ảnh 3 - Ảnh Bảng giá / Thực đơn (Menu):**
  - Chụp rõ bảng giá treo tường hoặc menu để làm dữ liệu tham khảo.
- **Ảnh 4 - Không gian quán (Tùy chọn):**
  - Chụp bàn ghế, không gian ngồi ăn của sinh viên.

#### 2. Yêu cầu kỹ thuật ảnh:
- **Tỷ lệ ảnh:** Ưu tiên **ảnh chụp ngang (Landscape) tỷ lệ 16:9 hoặc 4:3**. (Hạn chế ảnh chụp dọc vì khi đưa lên giao diện web sẽ bị cắt xén xấu).
- **Định dạng file:** `.jpg`, `.jpeg`, `.png`, hoặc `.webp`.
- **Dung lượng:** Dưới **5MB/ảnh** (ảnh rõ nét, không bị rung mờ).
- **Quy tắc đặt tên file ảnh:** Để không bị nhầm lẫn giữa các quán, đặt tên file theo cú pháp:
  ```
  [ten-quan]_[loai-anh].[jpg/png]
  
  Ví dụ:
  - com-tam-cay-bang_cover.jpg   (Ảnh bìa quán)
  - com-tam-cay-bang_menu.jpg    (Ảnh menu)
  - com-tam-cay-bang_mon1.jpg    (Ảnh món ăn)
  ```

---

### PHẦN 4: DỮ LIỆU ĐÁNH GIÁ MẪU (REVIEW MỒI CHO DEMO)

Để website có dữ liệu sinh động ngay khi chạy demo, mỗi quán cần chuẩn bị sẵn **2 đến 3 review thực tế**:

1. **Người đánh giá:** Tên hiển thị (Ví dụ: *Nguyễn Văn A - KTX Khu B, Trần Thị B - UIT K18*).
2. **Số sao:** Từ 1 đến 5 (Khuyến khích có cả 4 sao, 5 sao và 3 sao để số điểm trung bình có số thập phân tự nhiên, ví dụ: 4.3⭐, 4.7⭐).
3. **Nội dung nhận xét:** Viết đánh giá chân thật về:
   - Vị đồ ăn (đậm đà, vừa miệng hay nhạt)
   - Giá cả so với sinh viên (rẻ, hợp lý hay đắt)
   - Thái độ phục vụ và tốc độ ra món
   - Độ sạch sẽ và chỗ gửi xe
4. **Ảnh kèm theo review:** 1-2 ảnh chụp thực tế lúc ăn.

---

### PHẦN 5: BẢNG TEMPLATE EXCEL / GOOGLE SHEETS MẪU

Bạn có thể copy bảng này vào Google Sheets và chia sẻ link cho nhóm nhập dữ liệu:

| Tên Quán | Khu Vực | Danh Mục | Địa Chỉ / Mốc | Giá Min | Giá Max | Giờ Mở Cửa | Link Google Map | Tên File Ảnh Bìa | Món Tiêu Biểu & Giá | Điểm Sao Review 1 | Nội Dung Review 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Cơm Tấm Cây Bàng | AREA_KTX_B | CAT_COM | Đối diện KTX Khu B, đường Tô Vĩnh Diện | 25000 | 40000 | 06:00 - 21:00 | https://maps.app.goo.gl/... | com-tam-cay-bang_cover.jpg | Cơm sườn (28k), Sườn bì chả (35k) | 5 | Quán ăn quen thuộc, sườn ướp đậm vị, cơm thêm thoải mái |
| Bún Bò Huế O Oanh | AREA_KTX_A | CAT_BUN_PHO | Trục đường A1, gần KTX Khu A | 30000 | 45000 | 06:30 - 20:30 | https://maps.app.goo.gl/... | bun-bo-o-oanh_cover.jpg | Bún bò tái (35k), Bún bò đặc biệt (45k) | 4 | Nước dùng ngọt thanh, sa tế thơm, quán hơi đông vào sáng |
| Trà Sữa May | AREA_CHO_DEM | CAT_TRA_SUA | Ngay cổng Chợ Đêm Làng ĐH | 18000 | 35000 | 09:00 - 23:00 | https://maps.app.goo.gl/... | tra-sua-may_cover.jpg | Trà đào cam sả (25k), Trà sữa olong (28k) | 5 | Giá sinh viên, trân châu mềm dẻo, mở khuya tiện tụ tập |

---

### PHẦN 6: CÁCH BÀN GIAO CHO LẬP TRÌNH VIÊN

Khi nhóm thu thập xong, chỉ cần nén thành 1 thư mục dạng:
```
DuLieu_FoodReview/
├── ThongTinQuan.xlsx (hoặc file google sheet)
└── HinhAnh/
    ├── com-tam-cay-bang_cover.jpg
    ├── com-tam-cay-bang_menu.jpg
    ├── bun-bo-o-oanh_cover.jpg
    └── ...
```
Backend sẽ có sẵn script đọc trực tiếp thư mục này để đưa vào Database chỉ trong vài giây!

