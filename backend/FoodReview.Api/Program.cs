using System.Text;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using Microsoft.OpenApi.Models;
using FoodReview.Api.Data;
using FoodReview.Api.Services;

var builder = WebApplication.CreateBuilder(args);

// 1. Add Dapper Database Connection Factory (Singleton / Scoped)
builder.Services.AddSingleton<IDbConnectionFactory, SqliteConnectionFactory>();

// 2. Add Business Services (Pure Dapper)
builder.Services.AddScoped<IAuthService, AuthService>();
builder.Services.AddScoped<ICategoryAreaService, CategoryAreaService>();
builder.Services.AddScoped<IPlaceService, PlaceService>();
builder.Services.AddScoped<IReviewService, ReviewService>();
builder.Services.AddScoped<IInteractionService, InteractionService>();
builder.Services.AddScoped<IAdminService, AdminService>();
builder.Services.AddScoped<IFileStorageService, LocalFileStorageService>();

// 3. Configure JWT Authentication
var jwtKey = builder.Configuration["Jwt:Key"] ?? "FoodReviewLangDaiHocSuperSecretKeyForDevelopment2026!@#$%^";
var jwtIssuer = builder.Configuration["Jwt:Issuer"] ?? "FoodReviewApi";
var jwtAudience = builder.Configuration["Jwt:Audience"] ?? "FoodReviewClient";

builder.Services.AddAuthentication(options =>
{
    options.DefaultAuthenticateScheme = JwtBearerDefaults.AuthenticationScheme;
    options.DefaultChallengeScheme = JwtBearerDefaults.AuthenticationScheme;
})
.AddJwtBearer(options =>
{
    options.RequireHttpsMetadata = false;
    options.SaveToken = true;
    options.TokenValidationParameters = new TokenValidationParameters
    {
        ValidateIssuerSigningKey = true,
        IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey)),
        ValidateIssuer = true,
        ValidIssuer = jwtIssuer,
        ValidateAudience = true,
        ValidAudience = jwtAudience,
        ValidateLifetime = true,
        ClockSkew = TimeSpan.Zero
    };
});

builder.Services.AddAuthorization();

// 4. Configure CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.AllowAnyOrigin()
              .AllowAnyHeader()
              .AllowAnyMethod();
    });
});

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();

// 5. Configure Swagger / OpenAPI with JWT Bearer support
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new OpenApiInfo
    {
        Title = "Food Review - Làng Đại học API (Dapper Architecture)",
        Version = "v1",
        Description = "Hệ thống API RESTful Backend cho Website Review Làng Đại học ĐHQG TP.HCM xây dựng bằng Dapper Micro-ORM và C# 13 .NET 9."
    });

    c.AddSecurityDefinition("Bearer", new OpenApiSecurityScheme
    {
        Description = "Nhập token JWT theo định dạng: Bearer {token}",
        Name = "Authorization",
        In = ParameterLocation.Header,
        Type = SecuritySchemeType.ApiKey,
        Scheme = "Bearer"
    });

    c.AddSecurityRequirement(new OpenApiSecurityRequirement
    {
        {
            new OpenApiSecurityScheme
            {
                Reference = new OpenApiReference
                {
                    Type = ReferenceType.SecurityScheme,
                    Id = "Bearer"
                }
            },
            Array.Empty<string>()
        }
    });
});

var app = builder.Build();

// 6. Database auto-initialization via schema.sql and Dapper Seeding
using (var scope = app.Services.CreateScope())
{
    var factory = scope.ServiceProvider.GetRequiredService<IDbConnectionFactory>();
    await DbInitializer.InitializeAsync(factory, app.Environment);
}

// 7. Configure HTTP Pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI(c =>
    {
        c.SwaggerEndpoint("/swagger/v1/swagger.json", "Food Review API (Dapper) v1");
        c.RoutePrefix = "swagger";
    });
}

app.UseCors("AllowAll");

// Serve uploaded images static files
app.UseStaticFiles();

app.UseAuthentication();
app.UseAuthorization();

app.MapControllers();

app.Run();
