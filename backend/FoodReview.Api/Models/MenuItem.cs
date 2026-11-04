using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace FoodReview.Api.Models;

public class MenuItem
{
    public int Id { get; set; }

    public int PlaceId { get; set; }

    [JsonIgnore]
    public Place? Place { get; set; }

    [Required]
    [MaxLength(150)]
    public string Name { get; set; } = string.Empty;

    [Column(TypeName = "decimal(18, 2)")]
    public decimal Price { get; set; }

    [MaxLength(255)]
    public string? Note { get; set; }
}

