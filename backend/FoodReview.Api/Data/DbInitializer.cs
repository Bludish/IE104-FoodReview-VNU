using Dapper;

namespace FoodReview.Api.Data;

public static class DbInitializer
{
    public static async Task InitializeAsync(IDbConnectionFactory factory, IWebHostEnvironment env)
    {
        using var conn = factory.CreateConnection();

        var schemaPath = Path.Combine(env.ContentRootPath, "schema.sql");
        if (!File.Exists(schemaPath))
        {
            schemaPath = Path.Combine(env.ContentRootPath, "..", "..", "docs", "schema.sql");
        }

        if (File.Exists(schemaPath))
        {
            var schemaSql = await File.ReadAllTextAsync(schemaPath);
            await conn.ExecuteAsync(schemaSql);
        }
        else
        {
            throw new FileNotFoundException($"Không tìm thấy file schema.sql tại '{schemaPath}'.");
        }

        var userCount = await conn.ExecuteScalarAsync<int>("SELECT COUNT(1) FROM Users");
        if (userCount == 0)
        {
            var adminHash = BCrypt.Net.BCrypt.HashPassword("Admin@123");
            var studentHash = BCrypt.Net.BCrypt.HashPassword("Student@123");

            var insertUserSql = @"
                INSERT INTO Users (Email, PasswordHash, DisplayName, AvatarUrl, Role, Status)
                VALUES (@Email, @PasswordHash, @DisplayName, @AvatarUrl, @Role, @Status)";

            await conn.ExecuteAsync(insertUserSql, new[]
            {
                new { Email = "admin@langdaihoc.vn", PasswordHash = adminHash, DisplayName = "Quản Trị Viên (Admin)", AvatarUrl = "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=150", Role = 1, Status = 0 },
                new { Email = "sinhvien@uit.edu.vn", PasswordHash = studentHash, DisplayName = "Minh Hoàng (UIT K19)", AvatarUrl = "https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=150", Role = 0, Status = 0 },
                new { Email = "thuytien@hcmus.edu.vn", PasswordHash = studentHash, DisplayName = "Thủy Tiên (KHTN)", AvatarUrl = "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=150", Role = 0, Status = 0 }
            });
        }

        var areaCount = await conn.ExecuteScalarAsync<int>("SELECT COUNT(1) FROM Areas");
        if (areaCount == 0)
        {
            var insertAreaSql = "INSERT INTO Areas (Name, Description) VALUES (@Name, @Description)";
            await conn.ExecuteAsync(insertAreaSql, new[]
            {
                new { Name = "KTX Khu A & Cổng KTX A", Description = "Khu vực các quán ăn bao quanh KTX A" },
                new { Name = "KTX Khu B & Cổng KTX B", Description = "Đường Tô Vĩnh Diện và các quán ăn phục vụ sinh viên KTX B" },
                new { Name = "Khu Chợ Đêm Làng Đại Học", Description = "Khu chợ đêm, phố ăn vặt gần Nhà văn hóa Sinh viên" },
                new { Name = "Dốc ĐH Quốc Tế & ĐH Tự Nhiên", Description = "Trục đường dốc đối diện cổng trường ĐH Quốc Tế" },
                new { Name = "Trục Đường Vành Đai ĐHQG", Description = "Đoạn gần ĐH Bách Khoa CS2 và ĐH CNTT (UIT)" },
                new { Name = "Khu Tân Lập & Lân Cận", Description = "Khu ẩm thực Tân Lập giáp ranh Làng Đại học" }
            });
        }

        var catCount = await conn.ExecuteScalarAsync<int>("SELECT COUNT(1) FROM Categories");
        if (catCount == 0)
        {
            var insertCatSql = "INSERT INTO Categories (Name, Slug, Icon) VALUES (@Name, @Slug, @Icon)";
            await conn.ExecuteAsync(insertCatSql, new[]
            {
                new { Name = "Cơm Trưa / Cơm Tấm", Slug = "com-trua-com-tam", Icon = "utensils" },
                new { Name = "Bún / Phở / Mì", Slug = "bun-pho-mi", Icon = "soup" },
                new { Name = "Ăn Vặt & Xiên Nướng", Slug = "an-vat-xien-nuong", Icon = "flame" },
                new { Name = "Trà Sữa & Cà Phê", Slug = "tra-sua-ca-phe", Icon = "coffee" },
                new { Name = "Lẩu & Nướng Sinh Viên", Slug = "lau-nuong-sinh-vien", Icon = "pot" },
                new { Name = "Món Chay", Slug = "mon-chay", Icon = "leaf" }
            });
        }

        var placeCount = await conn.ExecuteScalarAsync<int>("SELECT COUNT(1) FROM Places");
        if (placeCount == 0)
        {
            var ktxBId = await conn.ExecuteScalarAsync<int>("SELECT Id FROM Areas WHERE Name ILIKE '%Khu B%' LIMIT 1");
            var ktxAId = await conn.ExecuteScalarAsync<int>("SELECT Id FROM Areas WHERE Name ILIKE '%Khu A%' LIMIT 1");

            var catComId = await conn.ExecuteScalarAsync<int>("SELECT Id FROM Categories WHERE Slug = 'com-trua-com-tam' LIMIT 1");
            var catBunId = await conn.ExecuteScalarAsync<int>("SELECT Id FROM Categories WHERE Slug = 'bun-pho-mi' LIMIT 1");

            var studentId = await conn.ExecuteScalarAsync<int>("SELECT Id FROM Users WHERE Role = 0 LIMIT 1");

            var insertPlaceSql = @"
                INSERT INTO Places (Name, Slug, Description, Address, AreaId, CategoryId, PriceMin, PriceMax, OpeningHours, Latitude, Longitude, CoverImageUrl, Status)
                VALUES (@Name, @Slug, @Description, @Address, @AreaId, @CategoryId, @PriceMin, @PriceMax, @OpeningHours, @Latitude, @Longitude, @CoverImageUrl, @Status)
                RETURNING Id;";

            // Place 1
            var p1Id = await conn.ExecuteScalarAsync<int>(insertPlaceSql, new
            {
                Name = "Cơm Tấm Cây Bàng (Cơm Thêm Miễn Phí)",
                Slug = "com-tam-cay-bang",
                Description = "Quán cơm tấm bình dân quen thuộc của sinh viên KTX B. Sườn ướp đậm đà nướng than thơm phức, canh và cơm thêm tự phục vụ miễn phí.",
                Address = "Số 18 Đường Tô Vĩnh Diện, đối diện cổng KTX Khu B",
                AreaId = ktxBId,
                CategoryId = catComId,
                PriceMin = 25000m,
                PriceMax = 40000m,
                OpeningHours = "06:30 - 21:00",
                Latitude = 10.8801,
                Longitude = 106.7792,
                CoverImageUrl = "https://images.unsplash.com/photo-1541832676-9b763b0239ab?w=800",
                Status = 1
            });

            await conn.ExecuteAsync("INSERT INTO MenuItems (PlaceId, Name, Price, Note) VALUES (@PlaceId, @Name, @Price, @Note)", new[]
            {
                new { PlaceId = p1Id, Name = "Cơm sườn nướng", Price = 28000m, Note = (string?)"Món bán chạy nhất" },
                new { PlaceId = p1Id, Name = "Cơm sườn bì chả trứng", Price = 38000m, Note = (string?)"Suất đặc biệt" },
                new { PlaceId = p1Id, Name = "Cơm đùi gà chiên nước mắm", Price = 32000m, Note = (string?)null }
            });

            var r1Id = await conn.ExecuteScalarAsync<int>(@"
                INSERT INTO Reviews (PlaceId, UserId, Rating, Content, Status)
                VALUES (@PlaceId, @UserId, 5, @Content, 1)
                RETURNING Id;", new
            {
                PlaceId = p1Id,
                UserId = studentId,
                Content = "Quán ruột của mình suốt 3 năm ở KTX B! Miếng sườn to ướp rất vừa miệng, cơm thêm thoải mái không phụ thu."
            });

            await conn.ExecuteAsync("INSERT INTO ReviewImages (ReviewId, ImageUrl, AltText) VALUES (@ReviewId, @ImageUrl, @AltText)", new
            {
                ReviewId = r1Id,
                ImageUrl = "https://images.unsplash.com/photo-1541832676-9b763b0239ab?w=600",
                AltText = "Đĩa cơm sườn"
            });

            // Place 2
            var p2Id = await conn.ExecuteScalarAsync<int>(insertPlaceSql, new
            {
                Name = "Bún Bò Huế O Oanh - KTX Khu A",
                Slug = "bun-bo-hue-o-oanh",
                Description = "Quán bún bò gốc Huế chuẩn vị, nước dùng thơm nồng sả và mắm ruốc, tóp mỡ giòn rụm và rau sống tươi rói.",
                Address = "Đường A1, Cạnh cổng phụ KTX Khu A ĐHQG",
                AreaId = ktxAId,
                CategoryId = catBunId,
                PriceMin = 30000m,
                PriceMax = 45000m,
                OpeningHours = "06:00 - 21:30",
                Latitude = 10.8752,
                Longitude = 106.8011,
                CoverImageUrl = "https://images.unsplash.com/photo-1582878826629-29b7ad1cdc43?w=800",
                Status = 1
            });

            await conn.ExecuteAsync("INSERT INTO MenuItems (PlaceId, Name, Price, Note) VALUES (@PlaceId, @Name, @Price, @Note)", new[]
            {
                new { PlaceId = p2Id, Name = "Bún bò tái nạm", Price = 35000m, Note = (string?)null },
                new { PlaceId = p2Id, Name = "Bún bò gân giò đặc biệt", Price = 45000m, Note = (string?)null }
            });

            await conn.ExecuteAsync(@"
                INSERT INTO Reviews (PlaceId, UserId, Rating, Content, Status)
                VALUES (@PlaceId, @UserId, 4, @Content, 1)", new
            {
                PlaceId = p2Id,
                UserId = studentId,
                Content = "Nước dùng rất thơm, nêm nếm đậm đà. Ăn xong no căng bụng giá chỉ 35k quá hợp túi tiền sinh viên."
            });
        }
    }
}
