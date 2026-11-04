namespace FoodReview.Api.DTOs;

public record PagedResult<T>(
    IReadOnlyList<T> Items,
    int TotalCount,
    int PageIndex,
    int PageSize,
    int TotalPages
);

public record ApiResponse<T>(
    bool Success,
    string Message,
    T? Data = default
);

