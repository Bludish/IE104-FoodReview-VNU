using Dapper;
using FoodReview.Api.Data;
using FoodReview.Api.DTOs;
using FoodReview.Api.Models;

namespace FoodReview.Api.Services;

public class CategoryAreaService : ICategoryAreaService
{
    private readonly IDbConnectionFactory _dbFactory;

    public CategoryAreaService(IDbConnectionFactory dbFactory)
    {
        _dbFactory = dbFactory;
    }

    public async Task<List<AreaDto>> GetAllAreasAsync()
    {
        using var conn = _dbFactory.CreateConnection();

        var sql = @"
            SELECT a.Id, a.Name, a.Description, a.ParentAreaId,
                   COUNT(CASE WHEN p.Status = 1 THEN 1 END) AS PlaceCount
            FROM Areas a
            LEFT JOIN Places p ON p.AreaId = a.Id
            GROUP BY a.Id, a.Name, a.Description, a.ParentAreaId
            ORDER BY a.Id";

        var list = await conn.QueryAsync<AreaDto>(sql);
        return list.AsList();
    }

    public async Task<AreaDto> CreateAreaAsync(CreateAreaRequest request)
    {
        using var conn = _dbFactory.CreateConnection();

        var insertSql = @"
            INSERT INTO Areas (Name, Description, ParentAreaId)
            VALUES (@Name, @Description, @ParentAreaId)
            RETURNING Id;";

        var id = await conn.ExecuteScalarAsync<int>(insertSql, new
        {
            Name = request.Name.Trim(),
            Description = request.Description?.Trim(),
            request.ParentAreaId
        });

        return new AreaDto(id, request.Name.Trim(), request.Description?.Trim(), request.ParentAreaId, 0);
    }

    public async Task<List<CategoryDto>> GetAllCategoriesAsync()
    {
        using var conn = _dbFactory.CreateConnection();

        var sql = @"
            SELECT c.Id, c.Name, c.Slug, c.Icon,
                   COUNT(CASE WHEN p.Status = 1 THEN 1 END) AS PlaceCount
            FROM Categories c
            LEFT JOIN Places p ON p.CategoryId = c.Id
            GROUP BY c.Id, c.Name, c.Slug, c.Icon
            ORDER BY c.Id";

        var list = await conn.QueryAsync<CategoryDto>(sql);
        return list.AsList();
    }

    public async Task<CategoryDto> CreateCategoryAsync(CreateCategoryRequest request)
    {
        using var conn = _dbFactory.CreateConnection();

        var insertSql = @"
            INSERT INTO Categories (Name, Slug, Icon)
            VALUES (@Name, @Slug, @Icon)
            RETURNING Id;";

        var id = await conn.ExecuteScalarAsync<int>(insertSql, new
        {
            Name = request.Name.Trim(),
            Slug = request.Slug.Trim().ToLower(),
            Icon = request.Icon?.Trim()
        });

        return new CategoryDto(id, request.Name.Trim(), request.Slug.Trim().ToLower(), request.Icon?.Trim(), 0);
    }
}
