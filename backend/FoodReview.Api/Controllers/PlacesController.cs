using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using FoodReview.Api.DTOs;
using FoodReview.Api.Services;

namespace FoodReview.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class PlacesController : ControllerBase
{
    private readonly IPlaceService _placeService;
    private readonly IInteractionService _interactionService;

    public PlacesController(IPlaceService placeService, IInteractionService interactionService)
    {
        _placeService = placeService;
        _interactionService = interactionService;
    }

    [HttpGet]
    public async Task<ActionResult<ApiResponse<PagedResult<PlaceSummaryDto>>>> GetPlaces([FromQuery] PlaceFilterParams filter)
    {
        var result = await _placeService.GetPlacesAsync(filter);
        return Ok(new ApiResponse<PagedResult<PlaceSummaryDto>>(true, "Lấy danh sách địa điểm thành công.", result));
    }

    [HttpGet("{id:int}")]
    public async Task<ActionResult<ApiResponse<PlaceDetailDto>>> GetById(int id)
    {
        int? userId = null;
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (int.TryParse(userIdStr, out var parsed)) userId = parsed;

        var place = await _placeService.GetPlaceDetailAsync(id, userId);
        if (place == null)
        {
            return NotFound(new ApiResponse<PlaceDetailDto>(false, "Không tìm thấy địa điểm quán ăn."));
        }

        return Ok(new ApiResponse<PlaceDetailDto>(true, "Lấy chi tiết địa điểm thành công.", place));
    }

    [HttpGet("slug/{slug}")]
    public async Task<ActionResult<ApiResponse<PlaceDetailDto>>> GetBySlug(string slug)
    {
        int? userId = null;
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (int.TryParse(userIdStr, out var parsed)) userId = parsed;

        var place = await _placeService.GetPlaceDetailBySlugAsync(slug, userId);
        if (place == null)
        {
            return NotFound(new ApiResponse<PlaceDetailDto>(false, "Không tìm thấy địa điểm quán ăn."));
        }

        return Ok(new ApiResponse<PlaceDetailDto>(true, "Lấy chi tiết địa điểm thành công.", place));
    }

    [Authorize(Roles = "Admin")]
    [HttpPost]
    public async Task<ActionResult<ApiResponse<PlaceDetailDto>>> Create([FromBody] CreatePlaceRequest request)
    {
        var place = await _placeService.CreatePlaceAsync(request);
        return CreatedAtAction(nameof(GetById), new { id = place.Id }, new ApiResponse<PlaceDetailDto>(true, "Tạo địa điểm thành công.", place));
    }

    [Authorize(Roles = "Admin")]
    [HttpPut("{id:int}")]
    public async Task<ActionResult<ApiResponse<PlaceDetailDto>>> Update(int id, [FromBody] UpdatePlaceRequest request)
    {
        try
        {
            var place = await _placeService.UpdatePlaceAsync(id, request);
            return Ok(new ApiResponse<PlaceDetailDto>(true, "Cập nhật địa điểm thành công.", place));
        }
        catch (KeyNotFoundException ex)
        {
            return NotFound(new ApiResponse<PlaceDetailDto>(false, ex.Message));
        }
    }

    [Authorize(Roles = "Admin")]
    [HttpDelete("{id:int}")]
    public async Task<ActionResult<ApiResponse<bool>>> Delete(int id)
    {
        var success = await _placeService.DeletePlaceAsync(id);
        if (!success)
        {
            return NotFound(new ApiResponse<bool>(false, "Không tìm thấy địa điểm để xóa/ẩn."));
        }
        return Ok(new ApiResponse<bool>(true, "Đã ẩn địa điểm thành công.", true));
    }

    [Authorize]
    [HttpPost("{id:int}/favorite")]
    public async Task<ActionResult<ApiResponse<object>>> ToggleFavorite(int id)
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<object>(false, "Vui lòng đăng nhập lại."));
        }

        try
        {
            var isAdded = await _interactionService.ToggleFavoriteAsync(userId, id);
            var msg = isAdded ? "Đã lưu vào danh sách quán yêu thích." : "Đã bỏ lưu khỏi danh sách yêu thích.";
            return Ok(new ApiResponse<object>(true, msg, new { isFavorite = isAdded }));
        }
        catch (KeyNotFoundException ex)
        {
            return NotFound(new ApiResponse<object>(false, ex.Message));
        }
    }
}

