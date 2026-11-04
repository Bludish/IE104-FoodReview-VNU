using System.ComponentModel.DataAnnotations;
using FoodReview.Api.Models;

namespace FoodReview.Api.DTOs;

public record CreateReportRequest(
    [Required] int ReviewId,
    [Required] ReportReason Reason,
    [MaxLength(500)] string? Description
);

public class ReportDto
{
    public int Id { get; set; }
    public int ReviewId { get; set; }
    public string ReviewContent { get; set; } = string.Empty;
    public int ReviewAuthorId { get; set; }
    public string ReviewAuthorName { get; set; } = string.Empty;
    public int ReporterId { get; set; }
    public string ReporterName { get; set; } = string.Empty;
    public ReportReason Reason { get; set; }
    public string? Description { get; set; }
    public ReportStatus Status { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? HandledAt { get; set; }
    public string? HandledByName { get; set; }

    public ReportDto() { }
    public ReportDto(int id, int reviewId, string reviewContent, int reviewAuthorId, string reviewAuthorName, int reporterId, string reporterName, ReportReason reason, string? description, ReportStatus status, DateTime createdAt, DateTime? handledAt, string? handledByName)
    {
        Id = id; ReviewId = reviewId; ReviewContent = reviewContent; ReviewAuthorId = reviewAuthorId;
        ReviewAuthorName = reviewAuthorName; ReporterId = reporterId; ReporterName = reporterName;
        Reason = reason; Description = description; Status = status; CreatedAt = createdAt;
        HandledAt = handledAt; HandledByName = handledByName;
    }
}

public record ResolveReportRequest(
    [Required] ReportStatus Status,
    bool HideReview = false
);
