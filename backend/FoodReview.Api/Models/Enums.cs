namespace FoodReview.Api.Models;

public enum UserRole
{
    User = 0,
    Admin = 1
}

public enum UserStatus
{
    Active = 0,
    Banned = 1
}

public enum PlaceStatus
{
    Draft = 0,
    Published = 1,
    Hidden = 2
}

public enum ReviewStatus
{
    Pending = 0,
    Published = 1,
    Hidden = 2
}

public enum ReportReason
{
    Spam = 0,
    Inappropriate = 1,
    Misleading = 2,
    Other = 3
}

public enum ReportStatus
{
    Pending = 0,
    Resolved = 1,
    Dismissed = 2
}

