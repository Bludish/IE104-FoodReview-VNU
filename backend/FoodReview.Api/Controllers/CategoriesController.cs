using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using FoodReview.Api.DTOs;
using FoodReview.Api.Services;

namespace FoodReview.Api.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CategoriesController : ControllerBase
{
    private readonly ICategoryAreaService _service;

    public CategoriesController(ICategoryAreaService service)
    {
        _service = service;
    }

    [HttpGet]
    public async Task<ActionResult<ApiResponse<List<CategoryDto>>>> GetAll()
    {
        var categories = await _service.GetAllCategoriesAsync();
        return Ok(new ApiResponse<List<CategoryDto>>(true, "Lấy danh sách danh mục thành công.", categories));
    }

    [Authorize(Roles = "Admin")]
    [HttpPost]
    public async Task<ActionResult<ApiResponse<CategoryDto>>> Create([FromBody] CreateCategoryRequest request)
    {
        var cat = await _service.CreateCategoryAsync(request);
        return CreatedAtAction(nameof(GetAll), new ApiResponse<CategoryDto>(true, "Thêm danh mục thành công.", cat));
    }
}

