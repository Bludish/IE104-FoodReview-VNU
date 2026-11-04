using System.ComponentModel.DataAnnotations;
using FoodReview.Api.Models;

namespace FoodReview.Api.DTOs;

public class ReviewAuthorDto
{
    public int Id { get; set; }
    public string DisplayName { get; set; } = string.Empty;
    public string? AvatarUrl { get; set; }

    public ReviewAuthorDto() { }
    public ReviewAuthorDto(int id, string displayName, string? avatarUrl)
    {
        Id = id; DisplayName = displayName; AvatarUrl = avatarUrl;
    }
}

public class ReviewImageDto
{
    public int Id { get; set; }
    public string ImageUrl { get; set; } = string.Empty;
    public string? AltText { get; set; }

    public ReviewImageDto() { }
    public ReviewImageDto(int id, string imageUrl, string? altText)
    {
        Id = id; ImageUrl = imageUrl; AltText = altText;
    }
}

public class ReviewDto
{
    public int Id { get; set; }
    public int PlaceId { get; set; }
    public string PlaceName { get; set; } = string.Empty;
    public ReviewAuthorDto Author { get; set; } = new();
    public int Rating { get; set; }
    public string Content { get; set; } = string.Empty;
    public ReviewStatus Status { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime UpdatedAt { get; set; }
    public List<ReviewImageDto> Images { get; set; } = new();

    public ReviewDto() { }
    public ReviewDto(int id, int placeId, string placeName, ReviewAuthorDto author, int rating, string content, ReviewStatus status, DateTime createdAt, DateTime updatedAt, List<ReviewImageDto> images)
    {
        Id = id; PlaceId = placeId; PlaceName = placeName; Author = author;
        Rating = rating; Content = content; Status = status;
        CreatedAt = createdAt; UpdatedAt = updatedAt; Images = images;
    }
}

public record CreateReviewRequest(
    [Required][Range(1, 5)] int Rating,
    [Required][MaxLength(2000)] string Content,
    List<string>? ImageUrls
);

public record UpdateReviewRequest(
    [Required][Range(1, 5)] int Rating,
    [Required][MaxLength(2000)] string Content,
    List<string>? ImageUrls
);
