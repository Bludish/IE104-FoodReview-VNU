using System.Data;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;
using Dapper;
using Microsoft.IdentityModel.Tokens;
using FoodReview.Api.Data;
using FoodReview.Api.DTOs;
using FoodReview.Api.Models;

namespace FoodReview.Api.Services;

public class AuthService : IAuthService
{
    private readonly IDbConnectionFactory _dbFactory;
    private readonly IConfiguration _config;

    public AuthService(IDbConnectionFactory dbFactory, IConfiguration config)
    {
        _dbFactory = dbFactory;
        _config = config;
    }

    public async Task<AuthResponse> RegisterAsync(RegisterRequest request)
    {
        using var conn = _dbFactory.CreateConnection();

        var existingCount = await conn.ExecuteScalarAsync<int>(
            "SELECT COUNT(1) FROM Users WHERE LOWER(Email) = LOWER(@Email)",
            new { request.Email });

        if (existingCount > 0)
        {
            throw new InvalidOperationException("Email này đã được sử dụng.");
        }

        var passwordHash = BCrypt.Net.BCrypt.HashPassword(request.Password);

        var insertSql = @"
            INSERT INTO Users (Email, PasswordHash, DisplayName, Role, Status)
            VALUES (@Email, @PasswordHash, @DisplayName, 0, 0);
            SELECT last_insert_rowid();";

        var newUserId = await conn.ExecuteScalarAsync<int>(insertSql, new
        {
            Email = request.Email.Trim().ToLower(),
            PasswordHash = passwordHash,
            DisplayName = request.DisplayName.Trim()
        });

        var user = new User
        {
            Id = newUserId,
            Email = request.Email.Trim().ToLower(),
            DisplayName = request.DisplayName.Trim(),
            Role = UserRole.User,
            Status = UserStatus.Active
        };

        var token = GenerateJwtToken(user);
        return new AuthResponse(user.Id, user.Email, user.DisplayName, null, "User", token);
    }

    public async Task<AuthResponse> LoginAsync(LoginRequest request)
    {
        using var conn = _dbFactory.CreateConnection();

        var user = await conn.QueryFirstOrDefaultAsync<User>(
            "SELECT * FROM Users WHERE LOWER(Email) = LOWER(@Email)",
            new { request.Email });

        if (user == null || !BCrypt.Net.BCrypt.Verify(request.Password, user.PasswordHash))
        {
            throw new UnauthorizedAccessException("Email hoặc mật khẩu không chính xác.");
        }

        if (user.Status == UserStatus.Banned)
        {
            throw new UnauthorizedAccessException("Tài khoản của bạn đã bị khóa do vi phạm quy tắc cộng đồng.");
        }

        var token = GenerateJwtToken(user);
        return new AuthResponse(user.Id, user.Email, user.DisplayName, user.AvatarUrl, user.Role.ToString(), token);
    }

    public async Task<UserProfileDto?> GetProfileAsync(int userId)
    {
        using var conn = _dbFactory.CreateConnection();

        var user = await conn.QueryFirstOrDefaultAsync<User>(
            "SELECT * FROM Users WHERE Id = @Id", new { Id = userId });

        if (user == null) return null;

        var totalReviews = await conn.ExecuteScalarAsync<int>(
            "SELECT COUNT(1) FROM Reviews WHERE UserId = @UserId", new { UserId = userId });

        var totalFavorites = await conn.ExecuteScalarAsync<int>(
            "SELECT COUNT(1) FROM Favorites WHERE UserId = @UserId", new { UserId = userId });

        return new UserProfileDto(
            user.Id,
            user.Email,
            user.DisplayName,
            user.AvatarUrl,
            user.Role.ToString(),
            user.CreatedAt,
            totalReviews,
            totalFavorites
        );
    }

    public async Task<UserProfileDto> UpdateProfileAsync(int userId, UpdateProfileRequest request)
    {
        using var conn = _dbFactory.CreateConnection();

        var rowsAffected = await conn.ExecuteAsync(@"
            UPDATE Users 
            SET DisplayName = @DisplayName, 
                AvatarUrl = COALESCE(@AvatarUrl, AvatarUrl)
            WHERE Id = @Id", new
        {
            Id = userId,
            DisplayName = request.DisplayName.Trim(),
            request.AvatarUrl
        });

        if (rowsAffected == 0)
        {
            throw new KeyNotFoundException("Không tìm thấy người dùng.");
        }

        return (await GetProfileAsync(userId))!;
    }

    public string GenerateJwtToken(User user)
    {
        var jwtKey = _config["Jwt:Key"] ?? "FoodReviewLangDaiHocSuperSecretKeyForDevelopment2026!@#$%^";
        var issuer = _config["Jwt:Issuer"] ?? "FoodReviewApi";
        var audience = _config["Jwt:Audience"] ?? "FoodReviewClient";
        var expiresDays = int.TryParse(_config["Jwt:ExpiresInDays"], out var days) ? days : 7;

        var key = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey));
        var creds = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

        var claims = new List<Claim>
        {
            new(ClaimTypes.NameIdentifier, user.Id.ToString()),
            new(ClaimTypes.Email, user.Email),
            new(ClaimTypes.Name, user.DisplayName),
            new(ClaimTypes.Role, user.Role.ToString())
        };

        var token = new JwtSecurityToken(
            issuer: issuer,
            audience: audience,
            claims: claims,
            expires: DateTime.UtcNow.AddDays(expiresDays),
            signingCredentials: creds
        );

        return new JwtSecurityTokenHandler().WriteToken(token);
    }
}
