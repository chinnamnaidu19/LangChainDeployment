import os
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# MOVIE SEARCH TOOL
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali",
        "romance": "Sita Ramam, Geetha Govindam, Jab We Met",
        "thriller": "Drishyam, Ratsasan, Andhadhun",
        "drama": "Taare Zameen Par, Dangal, 12th Fail",
        "fantasy": "Baahubali, Eega, Tumbbad",
        "horror": "Tumbbad, Stree, Bhool Bhulaiyaa"
    }

    requested_genre = genre.strip().lower()

    if requested_genre in movies:
        return (
            f"Indian {requested_genre} movies found: "
            f"{movies[requested_genre]}"
        )

    return (
        f"No Indian {requested_genre} movies were found "
        f"in the database."
    )


# ============================================================
# TEMPERATURE CONVERSION TOOL
# ============================================================

@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return round((temp_c * 1.8) + 32, 2)


# ============================================================
# WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather for an Indian city using Open-Meteo."""

    try:
        # ----------------------------------------------------
        # STEP 1: Clean city name
        # ----------------------------------------------------

        city_input = city.strip()

        # Common Indian city aliases
        city_aliases = {
            "vizag": "Visakhapatnam",
            "vishakapatnam": "Visakhapatnam",
            "vishakhapatnam": "Visakhapatnam",
            "bombay": "Mumbai",
            "calcutta": "Kolkata",
            "madras": "Chennai",
            "bengaluru": "Bangalore"
        }

        search_city = city_aliases.get(
            city_input.lower(),
            city_input
        )

        # ----------------------------------------------------
        # STEP 2: Find latitude and longitude
        # ----------------------------------------------------

        geo_url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        geo_params = {
            "name": search_city,
            "count": 10,
            "language": "en",
            "format": "json"
        }

        headers = {
            "User-Agent": "IndianWeatherCinemaAgent/1.0"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            headers=headers,
            timeout=20
        )

        print("Geocoding URL:", geo_response.url)
        print(
            "Geocoding status:",
            geo_response.status_code
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        # ----------------------------------------------------
        # STEP 3: Check search results
        # ----------------------------------------------------

        if not geo_data.get("results"):
            return (
                f"Could not find weather data for "
                f"{search_city}."
            )

        # ----------------------------------------------------
        # STEP 4: Find an Indian result
        # ----------------------------------------------------

        location = None

        for result in geo_data["results"]:
            if (
                result.get("country_code", "").upper()
                == "IN"
            ):
                location = result
                break

        if location is None:
            return (
                f"Could not find an Indian city named "
                f"{search_city}."
            )

        latitude = location["latitude"]
        longitude = location["longitude"]

        resolved_city = location.get(
            "name",
            search_city
        )

        # ----------------------------------------------------
        # STEP 5: Get current weather
        # ----------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "weather_code,"
                "wind_speed_10m"
            ),
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            headers=headers,
            timeout=20
        )

        print("Weather URL:", weather_response.url)
        print(
            "Weather status:",
            weather_response.status_code
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        # ----------------------------------------------------
        # STEP 6: Validate weather response
        # ----------------------------------------------------

        if "current" not in weather_data:
            return (
                f"Weather information is unavailable "
                f"for {resolved_city}."
            )

        current = weather_data["current"]

        temperature = current.get(
            "temperature_2m"
        )

        humidity = current.get(
            "relative_humidity_2m"
        )

        weather_code = current.get(
            "weather_code"
        )

        wind_speed = current.get(
            "wind_speed_10m"
        )

        # ----------------------------------------------------
        # STEP 7: Convert weather code to description
        # ----------------------------------------------------

        weather_descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            56: "Light freezing drizzle",
            57: "Dense freezing drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            66: "Light freezing rain",
            67: "Heavy freezing rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            77: "Snow grains",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            85: "Slight snow showers",
            86: "Heavy snow showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }

        description = weather_descriptions.get(
            weather_code,
            "Unknown weather condition"
        )

        # ----------------------------------------------------
        # STEP 8: Return normal text
        # ----------------------------------------------------

        return (
            f"Current weather in {resolved_city}: "
            f"{temperature}°C, {description}. "
            f"Humidity: {humidity}%. "
            f"Wind speed: {wind_speed} km/h."
        )

    # --------------------------------------------------------
    # HTTP/API ERROR
    # --------------------------------------------------------

    except requests.RequestException as e:

        print(
            f"WEATHER API ERROR for {city}: {e}"
        )

        return (
            f"Unable to retrieve weather for "
            f"{city} right now. "
            f"Please try again later."
        )

    # --------------------------------------------------------
    # OTHER ERROR
    # --------------------------------------------------------

    except Exception as e:

        print(
            f"WEATHER ERROR for {city}: {e}"
        )

        return (
            f"Could not retrieve weather "
            f"information for {city}."
        )


# ============================================================
# REGISTER TOOLS
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


# ============================================================
# GEMINI MODEL
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a specialized Indian Weather and Indian Cinema assistant.

You are authorized to answer questions about:

1. Indian weather
2. Weather of Indian cities
3. Indian movies
4. Indian cinema
5. Indian movie genres
6. Questions combining Indian weather and Indian movies

For weather questions, ALWAYS use the get_weather tool.

If the user mentions multiple Indian cities,
retrieve weather information for all requested cities
and include all of them in the final answer.

For movie questions, use the search_movies tool.

The user may request movie genres such as:

comedy, action, sci-fi, horror, romance,
thriller, drama, fantasy, etc.

An unavailable movie genre is NOT an unauthorized question.

If a requested movie genre is not available in the database,
tell the user that no Indian movies for that genre
were found in the database.

If the user asks for weather and movies together,
answer both parts in one normal text response.

For a valid movie genre,
provide the movies returned by the tool.

For completely unrelated topics outside
Indian weather and Indian cinema, respond EXACTLY:

I am not authorized to answer questions outside of Indian weather and cinema.

The final response must always be plain text.

Do not return JSON.
Do not return Python dictionaries.
Do not return internal agent state.
Do not return tool calls.
Do not return reasoning.
Do not return thinking.
Do not return metadata.
"""


# ============================================================
# CREATE AGENT
# ============================================================

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


# ============================================================
# INPUT MODEL
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description=(
            "Your message to the "
            "Indian Weather and Cinema Agent"
        )
    )


# ============================================================
# FORMAT USER INPUT
# ============================================================

def format_for_agent(x) -> dict:

    if isinstance(x, dict):
        user_input = x.get("input", "")
    else:
        user_input = x.input

    return {
        "messages": [
            ("user", user_input)
        ]
    }


# ============================================================
# EXTRACT FINAL AI RESPONSE
# ============================================================

def extract_text_response(agent_output) -> str:

    if isinstance(agent_output, str):
        return agent_output.strip()

    if not isinstance(agent_output, dict):
        return str(agent_output).strip()

    messages = agent_output.get("messages")

    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict):

                if "messages" in value:
                    messages = value["messages"]
                    break

    if not messages:
        return str(agent_output).strip()

    # Find the latest AI response
    for message in reversed(messages):

        if message.__class__.__name__ == "AIMessage":

            content = getattr(
                message,
                "content",
                ""
            )

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if isinstance(item, dict):

                        text = item.get("text")

                        if text:
                            text_parts.append(
                                str(text)
                            )

                    elif isinstance(item, str):

                        text_parts.append(item)

                return "\n".join(
                    text_parts
                ).strip()

        if isinstance(message, dict):

            message_type = message.get(
                "type",
                ""
            )

            if message_type in [
                "ai",
                "AIMessage"
            ]:

                content = message.get(
                    "content",
                    ""
                )

                if isinstance(content, str):
                    return content.strip()

    # Fallback
    last_message = messages[-1]

    content = getattr(
        last_message,
        "content",
        str(last_message)
    )

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


# ============================================================
# LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Indian Weather and Cinema Agent",
    version="1.0.0"
)


# ============================================================
# LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "running",
        "service": "Indian Weather and Cinema Agent"
    }


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )