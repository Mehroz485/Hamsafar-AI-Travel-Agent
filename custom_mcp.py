from mcp.server.fastmcp import FastMCP
import requests

mcp = FastMCP("Weather MCP Server")



WMO = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
    75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "violent rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def get_coordinates(city: str):
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1},
    )
    results = response.json().get("results")
    if not results:
        return None
    place = results[0]
    return place["latitude"], place["longitude"], place["name"]


@mcp.tool()
def get_current_weather(city: str):
    coords = get_coordinates(city)
    if not coords:
        return {"error": f"Could not find location: {city}"}
    lat, lon, name = coords

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        },
    )

    data = response.json()

    if response.status_code != 200:
        return {"error": data.get("reason", "Could not fetch current weather")}

    current = data["current"]

    return {
        "city": name,
        "temperature_c": current["temperature_2m"],
        "feels_like_c": current["apparent_temperature"],
        "humidity": current["relative_humidity_2m"],
        "condition": WMO.get(current["weather_code"], "unknown"),
        "wind_speed": current["wind_speed_10m"]
    }


@mcp.tool()
def get_forecast(city: str):
    coords = get_coordinates(city)
    if not coords:
        return {"error": f"Could not find location: {city}"}
    lat, lon, name = coords

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "timezone": "auto",
            "forecast_days": 5,
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
        },
    )

    data = response.json()

    if response.status_code != 200:
        return {"error": data.get("reason", "Could not fetch forecast")}

    daily = data.get("daily", {})
    forecast = []

    for i in range(len(daily.get("time", []))):
        forecast.append(
            {
                "date": daily["time"][i],
                "max_temperature": daily["temperature_2m_max"][i],
                "min_temperature": daily["temperature_2m_min"][i],
                "weather": WMO.get(daily["weather_code"][i], "unknown"),
                "rain_chance_percent": daily["precipitation_probability_max"][i],
            }
        )

    return {
        "city": name,
        "forecast": forecast
    }


if __name__ == "__main__":
    mcp.run()

    







































































