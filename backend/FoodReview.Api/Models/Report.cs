using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace FoodReview.Api.Models;

public class Report
{
    public int Id { get; set; }

    public int ReporterId { get; set; }
    public User? Reporter { get; set; }

    public int ReviewId { get; set; }
    public Review? Review { get; set; }

    public ReportReason Reason { get; set; } = ReportReason.Other;

    [MaxLength(500)]
    public string? Description { get; set; }

    public ReportStatus Status { get; set; } = ReportStatus.Pending;

    public int? HandledBy { get; set; }
    public User? AdminHandler { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? HandledAt { get; set; }
}

