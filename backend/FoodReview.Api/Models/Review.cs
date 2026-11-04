using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace FoodReview.Api.Models;

public class Review
{
    public int Id { get; set; }

    public int PlaceId { get; set; }
    [JsonIgnore]
    public Place? Place { get; set; }

    public int UserId { get; set; }
    public User? User { get; set; }

    [Range(1, 5)]
    public int Rating { get; set; }

    [Required]
    [MaxLength(2000)]
    public string Content { get; set; } = string.Empty;

    public ReviewStatus Status { get; set; } = ReviewStatus.Published;

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime UpdatedAt { get; set; } = DateTime.UtcNow;

    public ICollection<ReviewImage> Images { get; set; } = new List<ReviewImage>();
    [JsonIgnore]
    public ICollection<Report> Reports { get; set; } = new List<Report>();
}

