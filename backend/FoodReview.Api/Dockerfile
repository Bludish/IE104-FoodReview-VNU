FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
WORKDIR /app
EXPOSE 8080
ENV ASPNETCORE_URLS=http://+:8080

FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src
COPY ["backend/FoodReview.Api/FoodReview.Api.csproj", "backend/FoodReview.Api/"]
RUN dotnet restore "backend/FoodReview.Api/FoodReview.Api.csproj"
COPY . .
WORKDIR "/src/backend/FoodReview.Api"
RUN dotnet build "FoodReview.Api.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "FoodReview.Api.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
COPY docs/schema.sql /app/schema.sql
ENTRYPOINT ["dotnet", "FoodReview.Api.dll"]
