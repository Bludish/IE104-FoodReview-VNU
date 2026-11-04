using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using FoodReview.Api.DTOs;
using FoodReview.Api.Models;
using FoodReview.Api.Services;

namespace FoodReview.Api.Controllers;

[Authorize(Roles = "Admin")]
[ApiController]
[Route("api/[controller]")]
public class AdminController : ControllerBase
{
    private readonly IAdminService _adminService;

    public AdminController(IAdminService adminService)
    {
        _adminService = adminService;
    }

    [HttpGet("reports")]
    public async Task<ActionResult<ApiResponse<List<ReportDto>>>> GetReports([FromQuery] ReportStatus? status = null)
    {
        var reports = await _adminService.GetReportsAsync(status);
        return Ok(new ApiResponse<List<ReportDto>>(true, "Lấy danh sách báo cáo thành công.", reports));
    }

    [HttpPut("reports/{id:int}/resolve")]
    public async Task<ActionResult<ApiResponse<bool>>> ResolveReport(int id, [FromBody] ResolveReportRequest request)
    {
        var adminIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(adminIdStr, out var adminId))
        {
            return Unauthorized(new ApiResponse<bool>(false, "Phiên làm việc admin không hợp lệ."));
        }

        var success = await _adminService.ResolveReportAsync(id, adminId, request);
        if (!success)
        {
            return NotFound(new ApiResponse<bool>(false, "Không tìm thấy báo cáo."));
        }

        return Ok(new ApiResponse<bool>(true, "Đã xử lý báo cáo thành công.", true));
    }

    [HttpGet("users")]
    public async Task<ActionResult<ApiResponse<List<UserProfileDto>>>> GetUsers()
    {
        var users = await _adminService.GetAllUsersAsync();
        return Ok(new ApiResponse<List<UserProfileDto>>(true, "Lấy danh sách người dùng thành công.", users));
    }

    [HttpPut("users/{id:int}/toggle-ban")]
    public async Task<ActionResult<ApiResponse<bool>>> ToggleBanUser(int id)
    {
        var success = await _adminService.ToggleUserBanStatusAsync(id);
        if (!success)
        {
            return NotFound(new ApiResponse<bool>(false, "Không tìm thấy tài khoản người dùng."));
        }

        return Ok(new ApiResponse<bool>(true, "Đã thay đổi trạng thái tài khoản thành công.", true));
    }

    [HttpGet("places")]
    public async Task<ActionResult<ApiResponse<List<PlaceSummaryDto>>>> GetAllPlaces()
    {
        var places = await _adminService.GetAllPlacesForAdminAsync();
        return Ok(new ApiResponse<List<PlaceSummaryDto>>(true, "Lấy danh sách quản lý địa điểm thành công.", places));
    }

    [HttpPut("reviews/{id:int}/visibility")]
    public async Task<ActionResult<ApiResponse<bool>>> ToggleReviewVisibility(int id, [FromBody] ReviewStatus newStatus)
    {
        var success = await _adminService.ToggleReviewVisibilityAsync(id, newStatus);
        if (!success)
        {
            return NotFound(new ApiResponse<bool>(false, "Không tìm thấy đánh giá."));
        }

        return Ok(new ApiResponse<bool>(true, "Đã cập nhật trạng thái hiển thị của đánh giá.", true));
    }
}

