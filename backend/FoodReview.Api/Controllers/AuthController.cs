using System.Security.Claims;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using FoodReview.Api.DTOs;
using FoodReview.Api.Services;

namespace FoodReview.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AuthController : ControllerBase
{
    private readonly IAuthService _authService;

    public AuthController(IAuthService authService)
    {
        _authService = authService;
    }

    [HttpPost("register")]
    public async Task<ActionResult<ApiResponse<AuthResponse>>> Register([FromBody] RegisterRequest request)
    {
        try
        {
            var res = await _authService.RegisterAsync(request);
            return Ok(new ApiResponse<AuthResponse>(true, "Đăng ký tài khoản thành công.", res));
        }
        catch (InvalidOperationException ex)
        {
            return BadRequest(new ApiResponse<AuthResponse>(false, ex.Message));
        }
    }

    [HttpPost("login")]
    public async Task<ActionResult<ApiResponse<AuthResponse>>> Login([FromBody] LoginRequest request)
    {
        try
        {
            var res = await _authService.LoginAsync(request);
            return Ok(new ApiResponse<AuthResponse>(true, "Đăng nhập thành công.", res));
        }
        catch (UnauthorizedAccessException ex)
        {
            return Unauthorized(new ApiResponse<AuthResponse>(false, ex.Message));
        }
    }

    [Authorize]
    [HttpGet("me")]
    public async Task<ActionResult<ApiResponse<UserProfileDto>>> GetProfile()
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<UserProfileDto>(false, "Phiên đăng nhập không hợp lệ."));
        }

        var profile = await _authService.GetProfileAsync(userId);
        if (profile == null)
        {
            return NotFound(new ApiResponse<UserProfileDto>(false, "Không tìm thấy thông tin người dùng."));
        }

        return Ok(new ApiResponse<UserProfileDto>(true, "Lấy thông tin thành công.", profile));
    }

    [Authorize]
    [HttpPut("profile")]
    public async Task<ActionResult<ApiResponse<UserProfileDto>>> UpdateProfile([FromBody] UpdateProfileRequest request)
    {
        var userIdStr = User.FindFirstValue(ClaimTypes.NameIdentifier);
        if (!int.TryParse(userIdStr, out var userId))
        {
            return Unauthorized(new ApiResponse<UserProfileDto>(false, "Phiên đăng nhập không hợp lệ."));
        }

        var profile = await _authService.UpdateProfileAsync(userId, request);
        return Ok(new ApiResponse<UserProfileDto>(true, "Cập nhật hồ sơ thành công.", profile));
    }
}

