using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using FoodReview.Api.DTOs;
using FoodReview.Api.Services;

namespace FoodReview.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AreasController : ControllerBase
{
    private readonly ICategoryAreaService _service;

    public AreasController(ICategoryAreaService service)
    {
        _service = service;
    }

    /// <summary>
    /// Lấy danh sách các khu vực ẩm thực tại Làng Đại học (KTX Khu A, KTX Khu B, Chợ đêm,...)
    /// </summary>
    [HttpGet]
    public async Task<ActionResult<ApiResponse<List<AreaDto>>>> GetAll()
    {
        var areas = await _service.GetAllAreasAsync();
        return Ok(new ApiResponse<List<AreaDto>>(true, "Lấy danh sách khu vực thành công.", areas));
    }

    /// <summary>
    /// [Admin] Thêm mới khu vực ẩm thực
    /// </summary>
    [Authorize(Roles = "Admin")]
    [HttpPost]
    public async Task<ActionResult<ApiResponse<AreaDto>>> Create([FromBody] CreateAreaRequest request)
    {
        var area = await _service.CreateAreaAsync(request);
        return CreatedAtAction(nameof(GetAll), new ApiResponse<AreaDto>(true, "Thêm khu vực thành công.", area));
    }
}

