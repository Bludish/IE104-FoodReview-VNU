using System.Text.Json.Serialization;

namespace FoodReview.Api.Models;

public class Favorite
{
    public int UserId { get; set; }
    [JsonIgnore]
    public User? User { get; set; }

    public int PlaceId { get; set; }
    public Place? Place { get; set; }

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

