using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace FoodReview.Api.Models;

public class Category
{
    public int Id { get; set; }

    [Required]
    [MaxLength(100)]
    public string Name { get; set; } = string.Empty;

    [Required]
    [MaxLength(100)]
    public string Slug { get; set; } = string.Empty;

    [MaxLength(100)]
    public string? Icon { get; set; }

    [JsonIgnore]
    public ICollection<Place> Places { get; set; } = new List<Place>();
}

