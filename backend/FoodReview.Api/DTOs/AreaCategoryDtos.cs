using System.ComponentModel.DataAnnotations;

namespace FoodReview.Api.DTOs;

public class AreaDto
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string? Description { get; set; }
    public int? ParentAreaId { get; set; }
    public int PlaceCount { get; set; }

    public AreaDto() { }
    public AreaDto(int id, string name, string? description, int? parentAreaId, int placeCount)
    {
        Id = id; Name = name; Description = description; ParentAreaId = parentAreaId; PlaceCount = placeCount;
    }
}

public record CreateAreaRequest(
    [Required][MaxLength(100)] string Name,
    [MaxLength(255)] string? Description,
    int? ParentAreaId
);

public class CategoryDto
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Slug { get; set; } = string.Empty;
    public string? Icon { get; set; }
    public int PlaceCount { get; set; }

    public CategoryDto() { }
    public CategoryDto(int id, string name, string slug, string? icon, int placeCount)
    {
        Id = id; Name = name; Slug = slug; Icon = icon; PlaceCount = placeCount;
    }
}

public record CreateCategoryRequest(
    [Required][MaxLength(100)] string Name,
    [Required][MaxLength(100)] string Slug,
    [MaxLength(100)] string? Icon
);
