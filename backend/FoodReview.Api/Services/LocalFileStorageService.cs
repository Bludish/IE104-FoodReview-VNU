namespace FoodReview.Api.Services;

public interface IFileStorageService
{
    Task<string> SaveFileAsync(IFormFile file, string subFolder = "images");
    bool DeleteFile(string relativeUrl);
}

public class LocalFileStorageService : IFileStorageService
{
    private readonly IWebHostEnvironment _env;

    public LocalFileStorageService(IWebHostEnvironment env)
    {
        _env = env;
    }

    public async Task<string> SaveFileAsync(IFormFile file, string subFolder = "images")
    {
        if (file == null || file.Length == 0)
        {
            throw new ArgumentException("Tệp tải lên không hợp lệ.");
        }

        // Limit size: 5MB
        if (file.Length > 5 * 1024 * 1024)
        {
            throw new InvalidOperationException("Dung lượng tệp vượt quá giới hạn 5MB.");
        }

        // Validate extension
        var ext = Path.GetExtension(file.FileName).ToLowerInvariant();
        var allowedExtensions = new[] { ".jpg", ".jpeg", ".png", ".webp" };
        if (!allowedExtensions.Contains(ext))
        {
            throw new InvalidOperationException("Chỉ chấp nhận các định dạng ảnh: .jpg, .jpeg, .png, .webp.");
        }

        var webRoot = _env.WebRootPath ?? Path.Combine(Directory.GetCurrentDirectory(), "wwwroot");
        var uploadsDir = Path.Combine(webRoot, "uploads", subFolder);

        if (!Directory.Exists(uploadsDir))
        {
            Directory.CreateDirectory(uploadsDir);
        }

        var uniqueFileName = $"{Guid.NewGuid():N}_{Path.GetFileName(file.FileName)}";
        var filePath = Path.Combine(uploadsDir, uniqueFileName);

        using (var stream = new FileStream(filePath, FileMode.Create))
        {
            await file.CopyToAsync(stream);
        }

        return $"/uploads/{subFolder}/{uniqueFileName}";
    }

    public bool DeleteFile(string relativeUrl)
    {
        try
        {
            var webRoot = _env.WebRootPath ?? Path.Combine(Directory.GetCurrentDirectory(), "wwwroot");
            var cleanPath = relativeUrl.TrimStart('/').Replace('/', Path.DirectorySeparatorChar);
            var fullPath = Path.Combine(webRoot, cleanPath);

            if (File.Exists(fullPath))
            {
                File.Delete(fullPath);
                return true;
            }
        }
        catch
        {
            // Ignore failure on deletion
        }
        return false;
    }
}

