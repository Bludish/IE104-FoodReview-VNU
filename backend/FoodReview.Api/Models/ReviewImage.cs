using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace FoodReview.Api.Models;

public class ReviewImage
{
    public int Id { get; set; }

    public int ReviewId { get; set; }
    [JsonIgnore]
    public Review? Review { get; set; }

    [Required]
    [MaxLength(500)]
    public string ImageUrl { get; set; } = string.Empty;

    [MaxLength(200)]
    public string? AltText { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

