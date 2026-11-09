using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using FoodReview.Api.DTOs;
using FoodReview.Api.Services;

namespace FoodReview.Api.Controllers;

[Authorize]
[ApiController]
[Route("api/[controller]")]
public class UploadController : ControllerBase
{
    private readonly IFileStorageService _fileService;

    public UploadController(IFileStorageService fileService)
    {
        _fileService = fileService;
    }

    /// <summary>
    /// Tải tệp ảnh lên máy chủ (hỗ trợ JPG, PNG, WEBP, tối đa 5MB)
    /// </summary>
    [HttpPost]
    public async Task<ActionResult<ApiResponse<object>>> UploadFile(IFormFile file, [FromQuery] string folder = "images")
    {
        try
        {
            var relativeUrl = await _fileService.SaveFileAsync(file, folder);
            return Ok(new ApiResponse<object>(true, "Tải ảnh lên thành công.", new { url = relativeUrl }));
        }
        catch (Exception ex)
        {
            return BadRequest(new ApiResponse<object>(false, ex.Message));
        }
    }
}

