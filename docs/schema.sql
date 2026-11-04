-- =====================================================================
-- DATABASE SCHEMA: FOOD REVIEW - LÀNG ĐẠI HỌC (ĐHQG TP.HCM)
-- Engine: SQLite / SQL Server compatible DDL
-- Single Source of Truth for Database-First & Dapper Architecture
-- =====================================================================

PRAGMA foreign_keys = ON;

-- 1. BẢNG NGƯỜI DÙNG (USERS)
CREATE TABLE IF NOT EXISTS Users (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Email TEXT NOT NULL UNIQUE,
    PasswordHash TEXT NOT NULL,
    DisplayName TEXT NOT NULL,
    AvatarUrl TEXT,
    Role INTEGER NOT NULL DEFAULT 0,    -- 0: User, 1: Admin
    Status INTEGER NOT NULL DEFAULT 0,  -- 0: Active, 1: Banned
    CreatedAt TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_users_email ON Users(Email);

-- 2. BẢNG KHU VỰC LÀNG ĐẠI HỌC (AREAS)
CREATE TABLE IF NOT EXISTS Areas (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Description TEXT,
    ParentAreaId INTEGER,
    FOREIGN KEY (ParentAreaId) REFERENCES Areas(Id) ON DELETE SET NULL
);

-- 3. BẢNG DANH MỤC QUÁN & LOẠI MÓN (CATEGORIES)
CREATE TABLE IF NOT EXISTS Categories (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Slug TEXT NOT NULL UNIQUE,
    Icon TEXT
);

CREATE INDEX IF NOT EXISTS idx_categories_slug ON Categories(Slug);

-- 4. BẢNG ĐỊA ĐIỂM QUÁN ĂN (PLACES)
CREATE TABLE IF NOT EXISTS Places (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    Name TEXT NOT NULL,
    Slug TEXT NOT NULL UNIQUE,
    Description TEXT,
    Address TEXT NOT NULL,
    AreaId INTEGER NOT NULL,
    CategoryId INTEGER NOT NULL,
    PriceMin REAL NOT NULL DEFAULT 0,
    PriceMax REAL NOT NULL DEFAULT 0,
    OpeningHours TEXT DEFAULT '07:00 - 21:00',
    Latitude REAL,
    Longitude REAL,
    CoverImageUrl TEXT,
    Status INTEGER NOT NULL DEFAULT 1,  -- 0: Draft, 1: Published, 2: Hidden
    CreatedAt TEXT NOT NULL DEFAULT (datetime('now')),
    UpdatedAt TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (AreaId) REFERENCES Areas(Id) ON DELETE RESTRICT,
    FOREIGN KEY (CategoryId) REFERENCES Categories(Id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_places_slug ON Places(Slug);
CREATE INDEX IF NOT EXISTS idx_places_area ON Places(AreaId);
CREATE INDEX IF NOT EXISTS idx_places_category ON Places(CategoryId);
CREATE INDEX IF NOT EXISTS idx_places_status ON Places(Status);

-- 5. BẢNG THỰC ĐƠN TIÊU BIỂU (MENU_ITEMS)
CREATE TABLE IF NOT EXISTS MenuItems (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    PlaceId INTEGER NOT NULL,
    Name TEXT NOT NULL,
    Price REAL NOT NULL DEFAULT 0,
    Note TEXT,
    FOREIGN KEY (PlaceId) REFERENCES Places(Id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_menuitems_place ON MenuItems(PlaceId);

-- 6. BẢNG ĐÁNH GIÁ (REVIEWS)
CREATE TABLE IF NOT EXISTS Reviews (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    PlaceId INTEGER NOT NULL,
    UserId INTEGER NOT NULL,
    Rating INTEGER NOT NULL CHECK (Rating >= 1 AND Rating <= 5),
    Content TEXT NOT NULL,
    Status INTEGER NOT NULL DEFAULT 1,  -- 0: Pending, 1: Published, 2: Hidden
    CreatedAt TEXT NOT NULL DEFAULT (datetime('now')),
    UpdatedAt TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (PlaceId) REFERENCES Places(Id) ON DELETE CASCADE,
    FOREIGN KEY (UserId) REFERENCES Users(Id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reviews_place ON Reviews(PlaceId);
CREATE INDEX IF NOT EXISTS idx_reviews_user ON Reviews(UserId);

-- 7. BẢNG HÌNH ẢNH REVIEW (REVIEW_IMAGES)
CREATE TABLE IF NOT EXISTS ReviewImages (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    ReviewId INTEGER NOT NULL,
    ImageUrl TEXT NOT NULL,
    AltText TEXT,
    CreatedAt TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (ReviewId) REFERENCES Reviews(Id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_reviewimages_review ON ReviewImages(ReviewId);

-- 8. BẢNG BÁO CÁO VI PHẠM (REPORTS)
CREATE TABLE IF NOT EXISTS Reports (
    Id INTEGER PRIMARY KEY AUTOINCREMENT,
    ReporterId INTEGER NOT NULL,
    ReviewId INTEGER NOT NULL,
    Reason INTEGER NOT NULL DEFAULT 3,  -- 0: Spam, 1: Inappropriate, 2: Misleading, 3: Other
    Description TEXT,
    Status INTEGER NOT NULL DEFAULT 0,  -- 0: Pending, 1: Resolved, 2: Dismissed
    HandledBy INTEGER,
    CreatedAt TEXT NOT NULL DEFAULT (datetime('now')),
    HandledAt TEXT,
    FOREIGN KEY (ReporterId) REFERENCES Users(Id) ON DELETE RESTRICT,
    FOREIGN KEY (ReviewId) REFERENCES Reviews(Id) ON DELETE CASCADE,
    FOREIGN KEY (HandledBy) REFERENCES Users(Id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_reports_review ON Reports(ReviewId);
CREATE INDEX IF NOT EXISTS idx_reports_status ON Reports(Status);

-- 9. BẢNG QUÁN YÊU THÍCH (FAVORITES)
CREATE TABLE IF NOT EXISTS Favorites (
    UserId INTEGER NOT NULL,
    PlaceId INTEGER NOT NULL,
    CreatedAt TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (UserId, PlaceId),
    FOREIGN KEY (UserId) REFERENCES Users(Id) ON DELETE CASCADE,
    FOREIGN KEY (PlaceId) REFERENCES Places(Id) ON DELETE CASCADE
);

