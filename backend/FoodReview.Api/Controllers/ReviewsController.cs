using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using FoodReview.Api.DTOs;
using FoodReview.Api.Services;

namespace FoodReview.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class ReviewsController : ControllerBase
{
    private readonly IReviewService _reviewService;
    private readonly IInteractionService _interactionService;

    public ReviewsController(IReviewService reviewService, IInteractionService interactionService)
    {
        _reviewService = reviewService;
        _interactionService = interactionService;
    }

    [HttpGet("place/{placeId:int}")]
    public async Task<ActionResult<ApiResponse<PagedResult<ReviewDto>>>> GetByPlace(int placeId, [FromQuery] int page = 1, [FromQuery] int pageSize = 10)
    {
        var result = await _reviewService.GetReviewsByPlaceAsync(placeId, page, pageSize);
        return Ok(new ApiResponse<PagedResult<ReviewDto>>(true, "Lấy danh sách đánh giá thành công.", result));
    }

    [Authorize]
    [HttpGet("my")]
    public async Task<ActionResult<ApiResponse<PagedResult<ReviewDto>>>> GetMyReviews([FromQuery] int page = 1, [FromQuery] int pageSize = 10)
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<PagedResult<ReviewDto>>(false, "Phiên đăng nhập không hợp lệ."));
        }

        var result = await _reviewService.GetReviewsByUserAsync(userId, page, pageSize);
        return Ok(new ApiResponse<PagedResult<ReviewDto>>(true, "Lấy đánh giá của tôi thành công.", result));
    }

    [Authorize]
    [HttpPost("place/{placeId:int}")]
    public async Task<ActionResult<ApiResponse<ReviewDto>>> Create(int placeId, [FromBody] CreateReviewRequest request)
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<ReviewDto>(false, "Phiên đăng nhập không hợp lệ."));
        }

        try
        {
            var review = await _reviewService.CreateReviewAsync(placeId, userId, request);
            return Ok(new ApiResponse<ReviewDto>(true, "Đăng đánh giá thành công.", review));
        }
        catch (KeyNotFoundException ex)
        {
            return NotFound(new ApiResponse<ReviewDto>(false, ex.Message));
        }
        catch (UnauthorizedAccessException ex)
        {
            return StatusCode(403, new ApiResponse<ReviewDto>(false, ex.Message));
        }
    }

    [Authorize]
    [HttpPut("{id:int}")]
    public async Task<ActionResult<ApiResponse<ReviewDto>>> Update(int id, [FromBody] UpdateReviewRequest request)
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<ReviewDto>(false, "Phiên đăng nhập không hợp lệ."));
        }

        try
        {
            var review = await _reviewService.UpdateReviewAsync(id, userId, request);
            return Ok(new ApiResponse<ReviewDto>(true, "Cập nhật đánh giá thành công.", review));
        }
        catch (KeyNotFoundException ex)
        {
            return NotFound(new ApiResponse<ReviewDto>(false, ex.Message));
        }
        catch (UnauthorizedAccessException ex)
        {
            return StatusCode(403, new ApiResponse<ReviewDto>(false, ex.Message));
        }
    }

    [Authorize]
    [HttpDelete("{id:int}")]
    public async Task<ActionResult<ApiResponse<bool>>> Delete(int id)
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<bool>(false, "Phiên đăng nhập không hợp lệ."));
        }

        try
        {
            var success = await _reviewService.DeleteReviewAsync(id, userId);
            if (!success)
            {
                return NotFound(new ApiResponse<bool>(false, "Không tìm thấy đánh giá."));
            }
            return Ok(new ApiResponse<bool>(true, "Xóa đánh giá thành công.", true));
        }
        catch (UnauthorizedAccessException ex)
        {
            return StatusCode(403, new ApiResponse<bool>(false, ex.Message));
        }
    }

    [Authorize]
    [HttpPost("{id:int}/report")]
    public async Task<ActionResult<ApiResponse<ReportDto>>> ReportReview(int id, [FromBody] CreateReportRequest request)
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<ReportDto>(false, "Phiên đăng nhập không hợp lệ."));
        }

        try
        {
            var report = await _interactionService.ReportReviewAsync(userId, request with { ReviewId = id });
            return Ok(new ApiResponse<ReportDto>(true, "Đã gửi báo cáo vi phạm đến quản trị viên.", report));
        }
        catch (KeyNotFoundException ex)
        {
            return NotFound(new ApiResponse<ReportDto>(false, ex.Message));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new ApiResponse<ReportDto>(false, ex.Message));
        }
    }
}

