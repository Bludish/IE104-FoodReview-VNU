using System.ComponentModel.DataAnnotations;
using FoodReview.Api.Models;

namespace FoodReview.Api.DTOs;

public record RegisterRequest(
    [Required][EmailAddress] string Email,
    [Required][MinLength(6)] string Password,
    [Required][MaxLength(100)] string DisplayName
);

public record LoginRequest(
    [Required][EmailAddress] string Email,
    [Required] string Password
);

public record AuthResponse(
    int Id,
    string Email,
    string DisplayName,
    string? AvatarUrl,
    string Role,
    string Token
);

public record UserProfileDto(
    int Id,
    string Email,
    string DisplayName,
    string? AvatarUrl,
    string Role,
    DateTime CreatedAt,
    int TotalReviews,
    int TotalFavorites
);

public record UpdateProfileRequest(
    [Required][MaxLength(100)] string DisplayName,
    string? AvatarUrl
);

