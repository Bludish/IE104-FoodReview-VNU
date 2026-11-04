using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace FoodReview.Api.Models;

public class Place
{
    public int Id { get; set; }

    [Required]
    [MaxLength(200)]
    public string Name { get; set; } = string.Empty;

    [Required]
    [MaxLength(200)]
    public string Slug { get; set; } = string.Empty;

    public string? Description { get; set; }

    [Required]
    [MaxLength(300)]
    public string Address { get; set; } = string.Empty;

    public int AreaId { get; set; }
    public Area? Area { get; set; }

    public int CategoryId { get; set; }
    public Category? Category { get; set; }

    [Column(TypeName = "decimal(18, 2)")]
    public decimal PriceMin { get; set; }

    [Column(TypeName = "decimal(18, 2)")]
    public decimal PriceMax { get; set; }

    [MaxLength(100)]
    public string OpeningHours { get; set; } = "07:00 - 21:00";

    public double? Latitude { get; set; }
    public double? Longitude { get; set; }

    public string? CoverImageUrl { get; set; }

    public PlaceStatus Status { get; set; } = PlaceStatus.Published;

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    // Navigation properties
    public ICollection<MenuItem> MenuItems { get; set; } = new List<MenuItem>();
    public ICollection<Review> Reviews { get; set; } = new List<Review>();
    [JsonIgnore]
    public ICollection<Favorite> Favorites { get; set; } = new List<Favorite>();

    // Computed / Helper helpers
    [NotMapped]
    public double AverageRating => Reviews.Any(r => r.Status == ReviewStatus.Published)
        ? Math.Round(Reviews.Where(r => r.Status == ReviewStatus.Published).Average(r => r.Rating), 1)
        : 0;

    [NotMapped]
    public int ReviewCount => Reviews.Count(r => r.Status == ReviewStatus.Published);
}

