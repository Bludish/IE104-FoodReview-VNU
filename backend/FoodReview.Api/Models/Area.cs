using System.ComponentModel.DataAnnotations;
using System.Text.Json.Serialization;

namespace FoodReview.Api.Models;

public class Area
{
    public int Id { get; set; }

    [Required]
    [MaxLength(100)]
    public string Name { get; set; } = string.Empty;

    [MaxLength(255)]
    public string? Description { get; set; }

    public int? ParentAreaId { get; set; }

    [JsonIgnore]
    public Area? ParentArea { get; set; }

    [JsonIgnore]
    public ICollection<Area> SubAreas { get; set; } = new List<Area>();

    [JsonIgnore]
    public ICollection<Place> Places { get; set; } = new List<Place>();
}

