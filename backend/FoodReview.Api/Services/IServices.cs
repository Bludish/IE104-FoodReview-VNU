using FoodReview.Api.DTOs;
using FoodReview.Api.Models;

namespace FoodReview.Api.Services;

public interface IAuthService
{
    Task<AuthResponse> RegisterAsync(RegisterRequest request);
    Task<AuthResponse> LoginAsync(LoginRequest request);
    Task<UserProfileDto?> GetProfileAsync(int userId);
    Task<UserProfileDto> UpdateProfileAsync(int userId, UpdateProfileRequest request);
    string GenerateJwtToken(User user);
}

public interface ICategoryAreaService
{
    Task<List<AreaDto>> GetAllAreasAsync();
    Task<AreaDto> CreateAreaAsync(CreateAreaRequest request);
    Task<List<CategoryDto>> GetAllCategoriesAsync();
    Task<CategoryDto> CreateCategoryAsync(CreateCategoryRequest request);
}

public interface IPlaceService
{
    Task<PagedResult<PlaceSummaryDto>> GetPlacesAsync(PlaceFilterParams filter);
    Task<PlaceDetailDto?> GetPlaceDetailAsync(int id, int? currentUserId = null);
    Task<PlaceDetailDto?> GetPlaceDetailBySlugAsync(string slug, int? currentUserId = null);
    Task<PlaceDetailDto> CreatePlaceAsync(CreatePlaceRequest request);
    Task<PlaceDetailDto> UpdatePlaceAsync(int id, UpdatePlaceRequest request);
    Task<bool> DeletePlaceAsync(int id);
}

public interface IReviewService
{
    Task<ReviewDto> CreateReviewAsync(int placeId, int userId, CreateReviewRequest request);
    Task<ReviewDto> UpdateReviewAsync(int reviewId, int userId, UpdateReviewRequest request);
    Task<bool> DeleteReviewAsync(int reviewId, int userId);
    Task<PagedResult<ReviewDto>> GetReviewsByPlaceAsync(int placeId, int pageIndex = 1, int pageSize = 10);
    Task<PagedResult<ReviewDto>> GetReviewsByUserAsync(int userId, int pageIndex = 1, int pageSize = 10);
}

public interface IInteractionService
{
    Task<bool> ToggleFavoriteAsync(int userId, int placeId);
    Task<List<PlaceSummaryDto>> GetUserFavoritesAsync(int userId);
    Task<ReportDto> ReportReviewAsync(int userId, CreateReportRequest request);
}

public interface IAdminService
{
    Task<List<ReportDto>> GetReportsAsync(ReportStatus? status = null);
    Task<bool> ResolveReportAsync(int reportId, int adminId, ResolveReportRequest request);
    Task<List<UserProfileDto>> GetAllUsersAsync();
    Task<bool> ToggleUserBanStatusAsync(int userId);
    Task<List<PlaceSummaryDto>> GetAllPlacesForAdminAsync();
    Task<bool> ToggleReviewVisibilityAsync(int reviewId, ReviewStatus newStatus);
}

