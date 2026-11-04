using Dapper;
using FoodReview.Api.Data;
using FoodReview.Api.DTOs;
using FoodReview.Api.Models;

namespace FoodReview.Api.Services;

public class ReviewService : IReviewService
{
    private readonly IDbConnectionFactory _dbFactory;

    public ReviewService(IDbConnectionFactory dbFactory)
    {
        _dbFactory = dbFactory;
    }

    public async Task<ReviewDto> CreateReviewAsync(int placeId, int userId, CreateReviewRequest request)
    {
        using var conn = _dbFactory.CreateConnection();

        var place = await conn.QueryFirstOrDefaultAsync<dynamic>(
            "SELECT Id, Name, Status FROM Places WHERE Id = @Id", new { Id = placeId });

        if (place == null)
        {
            throw new KeyNotFoundException("Địa điểm không tồn tại.");
        }
        if ((int)place.Status == 2)
        {
            throw new KeyNotFoundException("Địa điểm đã bị ẩn.");
        }

        var user = await conn.QueryFirstOrDefaultAsync<User>(
            "SELECT * FROM Users WHERE Id = @Id", new { Id = userId });

        if (user == null || user.Status == UserStatus.Banned)
        {
            throw new UnauthorizedAccessException("Tài khoản không hợp lệ hoặc đã bị khóa.");
        }

        var insertSql = @"
            INSERT INTO Reviews (PlaceId, UserId, Rating, Content, Status)
            VALUES (@PlaceId, @UserId, @Rating, @Content, 1);
            SELECT last_insert_rowid();";

        var reviewId = await conn.ExecuteScalarAsync<int>(insertSql, new
        {
            PlaceId = placeId,
            UserId = userId,
            request.Rating,
            Content = request.Content.Trim()
        });

        var images = new List<ReviewImageDto>();
        if (request.ImageUrls != null && request.ImageUrls.Count > 0)
        {
            var insertImgSql = @"
                INSERT INTO ReviewImages (ReviewId, ImageUrl)
                VALUES (@ReviewId, @ImageUrl);
                SELECT last_insert_rowid();";

            foreach (var url in request.ImageUrls.Where(u => !string.IsNullOrWhiteSpace(u)))
            {
                var imgId = await conn.ExecuteScalarAsync<int>(insertImgSql, new
                {
                    ReviewId = reviewId,
                    ImageUrl = url.Trim()
                });
                images.Add(new ReviewImageDto(imgId, url.Trim(), null));
            }
        }

        return new ReviewDto(
            reviewId,
            placeId,
            (string)place.Name,
            new ReviewAuthorDto(user!.Id, user.DisplayName, user.AvatarUrl),
            request.Rating,
            request.Content.Trim(),
            ReviewStatus.Published,
            DateTime.UtcNow,
            DateTime.UtcNow,
            images
        );
    }

    public async Task<ReviewDto> UpdateReviewAsync(int reviewId, int userId, UpdateReviewRequest request)
    {
        using var conn = _dbFactory.CreateConnection();

        var review = await conn.QueryFirstOrDefaultAsync<dynamic>(
            "SELECT r.*, p.Name AS PlaceName FROM Reviews r JOIN Places p ON r.PlaceId = p.Id WHERE r.Id = @Id",
            new { Id = reviewId });

        if (review == null)
        {
            throw new KeyNotFoundException("Không tìm thấy đánh giá.");
        }

        // GUARD: Only author can edit
        if ((int)review.UserId != userId)
        {
            throw new UnauthorizedAccessException("Bạn chỉ được phép chỉnh sửa đánh giá của chính mình.");
        }

        var updateSql = @"
            UPDATE Reviews
            SET Rating = @Rating,
                Content = @Content,
                UpdatedAt = datetime('now')
            WHERE Id = @Id";

        await conn.ExecuteAsync(updateSql, new
        {
            Id = reviewId,
            request.Rating,
            Content = request.Content.Trim()
        });

        if (request.ImageUrls != null)
        {
            await conn.ExecuteAsync("DELETE FROM ReviewImages WHERE ReviewId = @ReviewId", new { ReviewId = reviewId });
            var insertImgSql = "INSERT INTO ReviewImages (ReviewId, ImageUrl) VALUES (@ReviewId, @ImageUrl)";
            foreach (var url in request.ImageUrls.Where(u => !string.IsNullOrWhiteSpace(u)))
            {
                await conn.ExecuteAsync(insertImgSql, new { ReviewId = reviewId, ImageUrl = url.Trim() });
            }
        }

        var user = await conn.QueryFirstOrDefaultAsync<User>("SELECT * FROM Users WHERE Id = @Id", new { Id = userId });
        var images = (await conn.QueryAsync<ReviewImageDto>(
            "SELECT Id, ImageUrl, AltText FROM ReviewImages WHERE ReviewId = @ReviewId", new { ReviewId = reviewId })).AsList();

        return new ReviewDto(
            reviewId,
            (int)review.PlaceId,
            (string)review.PlaceName,
            new ReviewAuthorDto(user!.Id, user.DisplayName, user.AvatarUrl),
            request.Rating,
            request.Content.Trim(),
            (ReviewStatus)(int)review.Status,
            DateTime.Parse((string)review.CreatedAt),
            DateTime.UtcNow,
            images
        );
    }

    public async Task<bool> DeleteReviewAsync(int reviewId, int userId)
    {
        using var conn = _dbFactory.CreateConnection();

        var authorId = await conn.ExecuteScalarAsync<int?>("SELECT UserId FROM Reviews WHERE Id = @Id", new { Id = reviewId });
        if (!authorId.HasValue) return false;

        // GUARD: Only author can delete
        if (authorId.Value != userId)
        {
            throw new UnauthorizedAccessException("Bạn chỉ được phép xóa đánh giá của chính mình.");
        }

        await conn.ExecuteAsync("DELETE FROM Reviews WHERE Id = @Id", new { Id = reviewId });
        return true;
    }

    public async Task<PagedResult<ReviewDto>> GetReviewsByPlaceAsync(int placeId, int pageIndex = 1, int pageSize = 10)
    {
        using var conn = _dbFactory.CreateConnection();

        var totalCount = await conn.ExecuteScalarAsync<int>(
            "SELECT COUNT(1) FROM Reviews WHERE PlaceId = @PlaceId AND Status = 1", new { PlaceId = placeId });

        var sql = @"
            SELECT 
                r.Id, r.PlaceId, p.Name AS PlaceName,
                u.Id AS AuthorId, u.DisplayName AS AuthorName, u.AvatarUrl AS AuthorAvatar,
                r.Rating, r.Content, r.Status, r.CreatedAt, r.UpdatedAt
            FROM Reviews r
            JOIN Users u ON r.UserId = u.Id
            JOIN Places p ON r.PlaceId = p.Id
            WHERE r.PlaceId = @PlaceId AND r.Status = 1
            ORDER BY r.CreatedAt DESC
            LIMIT @PageSize OFFSET @Offset";

        var rows = await conn.QueryAsync<dynamic>(sql, new
        {
            PlaceId = placeId,
            PageSize = pageSize,
            Offset = (pageIndex - 1) * pageSize
        });

        var items = new List<ReviewDto>();
        foreach (var row in rows)
        {
            int revId = (int)row.Id;
            var images = (await conn.QueryAsync<ReviewImageDto>(
                "SELECT Id, ImageUrl, AltText FROM ReviewImages WHERE ReviewId = @ReviewId",
                new { ReviewId = revId })).AsList();

            items.Add(new ReviewDto(
                revId,
                (int)row.PlaceId,
                (string)row.PlaceName,
                new ReviewAuthorDto((int)row.AuthorId, (string)row.AuthorName, (string?)row.AuthorAvatar),
                (int)row.Rating,
                (string)row.Content,
                (ReviewStatus)(int)row.Status,
                DateTime.Parse((string)row.CreatedAt),
                DateTime.Parse((string)row.UpdatedAt),
                images
            ));
        }

        var totalPages = (int)Math.Ceiling(totalCount / (double)pageSize);
        return new PagedResult<ReviewDto>(items, totalCount, pageIndex, pageSize, totalPages);
    }

    public async Task<PagedResult<ReviewDto>> GetReviewsByUserAsync(int userId, int pageIndex = 1, int pageSize = 10)
    {
        using var conn = _dbFactory.CreateConnection();

        var totalCount = await conn.ExecuteScalarAsync<int>(
            "SELECT COUNT(1) FROM Reviews WHERE UserId = @UserId", new { UserId = userId });

        var sql = @"
            SELECT 
                r.Id, r.PlaceId, p.Name AS PlaceName,
                u.Id AS AuthorId, u.DisplayName AS AuthorName, u.AvatarUrl AS AuthorAvatar,
                r.Rating, r.Content, r.Status, r.CreatedAt, r.UpdatedAt
            FROM Reviews r
            JOIN Users u ON r.UserId = u.Id
            JOIN Places p ON r.PlaceId = p.Id
            WHERE r.UserId = @UserId
            ORDER BY r.CreatedAt DESC
            LIMIT @PageSize OFFSET @Offset";

        var rows = await conn.QueryAsync<dynamic>(sql, new
        {
            UserId = userId,
            PageSize = pageSize,
            Offset = (pageIndex - 1) * pageSize
        });

        var items = new List<ReviewDto>();
        foreach (var row in rows)
        {
            int revId = (int)row.Id;
            var images = (await conn.QueryAsync<ReviewImageDto>(
                "SELECT Id, ImageUrl, AltText FROM ReviewImages WHERE ReviewId = @ReviewId",
                new { ReviewId = revId })).AsList();

            items.Add(new ReviewDto(
                revId,
                (int)row.PlaceId,
                (string)row.PlaceName,
                new ReviewAuthorDto((int)row.AuthorId, (string)row.AuthorName, (string?)row.AuthorAvatar),
                (int)row.Rating,
                (string)row.Content,
                (ReviewStatus)(int)row.Status,
                DateTime.Parse((string)row.CreatedAt),
                DateTime.Parse((string)row.UpdatedAt),
                images
            ));
        }

        var totalPages = (int)Math.Ceiling(totalCount / (double)pageSize);
        return new PagedResult<ReviewDto>(items, totalCount, pageIndex, pageSize, totalPages);
    }
}
