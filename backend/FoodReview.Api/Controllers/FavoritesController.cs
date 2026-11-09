using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using FoodReview.Api.DTOs;
using FoodReview.Api.Services;

namespace FoodReview.Api.Controllers;

[Authorize]
[ApiController]
[Route("api/[controller]")]
public class FavoritesController : ControllerBase
{
    private readonly IInteractionService _interactionService;

    public FavoritesController(IInteractionService interactionService)
    {
        _interactionService = interactionService;
    }

    /// <summary>
    /// Lấy danh sách các quán ăn mà người dùng hiện tại đã lưu yêu thích (Bookmark)
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<ApiResponse<List<PlaceSummaryDto>>>> GetMyFavorites()
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<List<PlaceSummaryDto>>(false, "Phiên đăng nhập không hợp lệ."));
        }

        var list = await _interactionService.GetUserFavoritesAsync(userId);
        return Ok(new ApiResponse<List<PlaceSummaryDto>>(true, "Lấy danh sách quán yêu thích thành công.", list));
    }
}

