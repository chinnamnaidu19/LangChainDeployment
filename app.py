import os
import requests

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. MOVIE DATABASE
# ============================================================

MOVIES = {
    "action": [
        "RRR",
        "Vikram",
        "Baahubali"
    ],

    "comedy": [
        "3 Idiots",
        "Hera Pheri",
        "Munna Bhai M.B.B.S."
    ],

    "sci-fi": [
        "Cargo",
        "2.0",
        "Mr. India"
    ],

    "romance": [
        "Sita Ramam",
        "Geetha Govindam",
        "Jab We Met"
    ],

    "thriller": [
        "Drishyam",
        "Ratsasan",
        "Andhadhun"
    ],

    "drama": [
        "Taare Zameen Par",
        "Dangal",
        "12th Fail"
    ],

    "fantasy": [
        "Baahubali",
        "Eega",
        "Tumbbad"
    ],

    "horror": [
        "Tumbbad",
        "Stree",
        "Bhool Bhulaiyaa"
    ]
}


# ============================================================
# 2. CITY ALIASES
# ============================================================

CITY_ALIASES = {
    "vizag": "Visakhapatnam",
    "vishakapatnam": "Visakhapatnam",
    "vishakhapatnam": "Visakhapatnam",

    "bombay": "Mumbai",

    "calcutta": "Kolkata",

    "madras": "Chennai",

    "bengaluru": "Bangalore",

    "bangalore": "Bangalore",

    "new delhi": "Delhi"
}


# ============================================================
# 3. WEATHER CODE DESCRIPTIONS
# ============================================================

WEATHER_DESCRIPTIONS = {

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


# ============================================================
# 4. MOVIE SEARCH TOOL
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """
    Search for Indian movies by genre.
    """

    requested_genre = genre.strip().lower()

    if requested_genre in MOVIES:

        movie_list = ", ".join(
            MOVIES[requested_genre]
        )

        return (
            f"Indian {requested_genre} movies: "
            f"{movie_list}"
        )

    available_genres = ", ".join(
        sorted(MOVIES.keys())
    )

    return (
        f"No Indian movies are currently available "
        f"for the genre '{requested_genre}'. "
        f"Available genres include: {available_genres}."
    )


# ============================================================
# 5. TEMPERATURE CONVERSION TOOL
# ============================================================

@tool
def change_to_f(temp_c: float) -> float:
    """
    Convert Celsius temperature to Fahrenheit.
    """

    return round(
        (temp_c * 1.8) + 32,
        2
    )


# ============================================================
# 6. WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get current weather information for an Indian city.
    """

    try:

        # ----------------------------------------------------
        # Clean city name
        # ----------------------------------------------------

        city_input = city.strip()

        if not city_input:

            return (
                "Please provide a valid Indian city "
                "or location."
            )

        # ----------------------------------------------------
        # Convert common names to official city names
        # ----------------------------------------------------

        search_city = CITY_ALIASES.get(
            city_input.lower(),
            city_input
        )

        # ----------------------------------------------------
        # STEP 1: GEOCODING
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
            "User-Agent":
                "Indian-Weather-Cinema-Agent/1.0"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            headers=headers,
            timeout=20
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        # ----------------------------------------------------
        # Check whether location exists
        # ----------------------------------------------------

        if not geo_data.get("results"):

            return (
                f"I could not find an Indian location "
                f"matching '{city_input}'. "
                f"Please provide a valid city name."
            )

        # ----------------------------------------------------
        # Find Indian location
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
                f"I could not identify '{city_input}' "
                f"as an Indian location."
            )

        # ----------------------------------------------------
        # Get coordinates
        # ----------------------------------------------------

        latitude = location["latitude"]

        longitude = location["longitude"]

        resolved_city = location.get(
            "name",
            search_city
        )

        state = location.get(
            "admin1",
            ""
        )

        # ----------------------------------------------------
        # STEP 2: WEATHER API
        # ----------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        weather_params = {

            "latitude": latitude,

            "longitude": longitude,

            "current":
                "temperature_2m,"
                "relative_humidity_2m,"
                "weather_code,"
                "wind_speed_10m",

            "temperature_unit":
                "celsius",

            "wind_speed_unit":
                "kmh"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            headers=headers,
            timeout=20
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()

        if "current" not in weather_data:

            return (
                f"Weather information is currently "
                f"unavailable for {resolved_city}."
            )

        current = weather_data["current"]

        # ----------------------------------------------------
        # Extract values
        # ----------------------------------------------------

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

        description = WEATHER_DESCRIPTIONS.get(
            weather_code,
            "Unknown weather condition"
        )

        # ----------------------------------------------------
        # Return professional weather response
        # ----------------------------------------------------

        location_text = resolved_city

        if state:
            location_text += f", {state}"

        return (
            f"Weather Report\n"
            f"Location: {location_text}, India\n"
            f"Temperature: {temperature}°C\n"
            f"Condition: {description}\n"
            f"Humidity: {humidity}%\n"
            f"Wind Speed: {wind_speed} km/h"
        )

    except requests.RequestException as error:

        print(
            f"WEATHER API ERROR: {error}"
        )

        return (
            f"I’m currently unable to retrieve live "
            f"weather information for {city}. "
            f"Please try again shortly."
        )

    except Exception as error:

        print(
            f"WEATHER PROCESSING ERROR: {error}"
        )

        return (
            f"I encountered an issue while processing "
            f"the weather request for {city}. "
            f"Please try again."
        )


# ============================================================
# 7. TOOLS
# ============================================================

tools = [
    get_weather,
    search_movies,
    change_to_f
]


# ============================================================
# 8. GEMINI API KEY
# ============================================================

GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY environment variable "
        "is not configured."
    )


# ============================================================
# 9. GEMINI MODEL
# ============================================================

llm_flash = ChatGoogleGenerativeAI(

    model="gemini-3.1-flash-lite-preview",

    google_api_key=GEMINI_API_KEY,

    temperature=0
)


# ============================================================
# 10. PROFESSIONAL SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Indian Weather and Cinema Assistant.

Your role is strictly limited to:

1. Indian weather
2. Weather of Indian cities and locations
3. Indian movies
4. Indian cinema
5. Indian movie genres
6. Questions combining Indian weather and Indian cinema

============================================================
WEATHER RULES
============================================================

For every weather-related question:

- ALWAYS use the get_weather tool.
- Do not guess weather information.
- Do not invent temperature values.
- If the user provides a city name, use that city.
- If the user uses a common name such as Vizag, understand it as
  Visakhapatnam.
- If multiple cities are requested, call the weather tool for
  EACH city.
- Present every requested location separately.
- Include the city/location name, temperature, condition,
  humidity and wind speed when available.

Example:

Weather Report

1. Visakhapatnam, Andhra Pradesh
   Temperature: 31°C
   Condition: Partly cloudy
   Humidity: 70%
   Wind Speed: 12 km/h

2. Hyderabad, Telangana
   Temperature: 29°C
   Condition: Clear sky
   Humidity: 55%
   Wind Speed: 10 km/h

============================================================
MOVIE RULES
============================================================

For movie-related questions:

- Use the search_movies tool.
- Only provide movies returned by the tool.
- Do not invent movie names.
- If the requested genre is available, provide the movies.
- If the requested genre is unavailable, clearly explain
  that the database currently does not contain movies for
  that genre.

Example:

Indian Action Movies

1. RRR
2. Vikram
3. Baahubali

============================================================
COMBINED QUESTIONS
============================================================

If the user asks about both weather and movies:

- Answer BOTH parts.
- Use the appropriate tool for each part.
- Organize the response professionally.

Example:

Weather Report
Visakhapatnam: 31°C, Partly cloudy
Hyderabad: 29°C, Clear sky

Indian Action Movies
1. RRR
2. Vikram
3. Baahubali

============================================================
IRRELEVANT QUESTIONS
============================================================

If the user asks about a topic completely unrelated to
Indian weather or Indian cinema, do not answer the question.

Instead respond professionally:

"Thank you for your question. I’m currently specialized in
Indian weather and Indian cinema, so I’m unable to assist
with that topic. Please ask me about weather conditions,
Indian cities, movies, cinema, or movie genres, and I’ll be
happy to help."

Do NOT provide information about the unrelated topic.

Examples of unrelated topics:

- Recipes
- Programming
- Mathematics
- General knowledge
- Politics
- Medical advice
- Sports unrelated to Indian cinema
- Technology
- Personal advice

============================================================
FINAL RESPONSE RULES
============================================================

- Be professional and helpful.
- Keep answers clear and easy to read.
- Use headings when useful.
- Use numbered lists for multiple locations or movies.
- Do not return JSON.
- Do not return Python dictionaries.
- Do not expose tool calls.
- Do not expose reasoning.
- Do not expose internal agent state.
- Do not mention system instructions.
- Do not hallucinate information.
"""


# ============================================================
# 11. CREATE AGENT
# ============================================================

agent = create_agent(

    model=llm_flash,

    tools=tools,

    system_prompt=SYSTEM_PROMPT
)


# ============================================================
# 12. LANGSERVE INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(
        description=(
            "Ask about Indian weather, Indian cities, "
            "Indian movies, cinema or movie genres."
        )
    )


# ============================================================
# 13. FORMAT INPUT
# ============================================================

def format_for_agent(x):

    if isinstance(x, dict):

        user_input = x.get(
            "input",
            ""
        )

    else:

        user_input = x.input

    return {

        "messages": [
            (
                "user",
                user_input
            )
        ]

    }


# ============================================================
# 14. EXTRACT FINAL RESPONSE
# ============================================================

def extract_text_response(agent_output):

    if isinstance(
        agent_output,
        str
    ):

        return agent_output.strip()

    if not isinstance(
        agent_output,
        dict
    ):

        return str(
            agent_output
        ).strip()

    messages = agent_output.get(
        "messages"
    )

    if messages is None:

        for value in agent_output.values():

            if isinstance(
                value,
                dict
            ):

                if "messages" in value:

                    messages = value[
                        "messages"
                    ]

                    break

    if not messages:

        return str(
            agent_output
        ).strip()

    # Find latest AI message
    for message in reversed(messages):

        # LangChain AIMessage
        if (
            message.__class__.__name__
            == "AIMessage"
        ):

            content = getattr(
                message,
                "content",
                ""
            )

            if isinstance(
                content,
                str
            ):

                return content.strip()

            if isinstance(
                content,
                list
            ):

                text_parts = []

                for item in content:

                    if isinstance(
                        item,
                        dict
                    ):

                        text = item.get(
                            "text"
                        )

                        if text:

                            text_parts.append(
                                str(text)
                            )

                    elif isinstance(
                        item,
                        str
                    ):

                        text_parts.append(
                            item
                        )

                return "\n".join(
                    text_parts
                ).strip()

        # Dictionary message
        if isinstance(
            message,
            dict
        ):

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

                if isinstance(
                    content,
                    str
                ):

                    return content.strip()

    # Fallback
    last_message = messages[-1]

    content = getattr(
        last_message,
        "content",
        str(last_message)
    )

    if isinstance(
        content,
        str
    ):

        return content.strip()

    return str(
        content
    ).strip()


# ============================================================
# 15. CREATE LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_agent
    )

    | agent

    | RunnableLambda(
        extract_text_response
    )

).with_types(

    input_type=AgentInput,

    output_type=str
)


# ============================================================
# 16. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="Indian Weather and Cinema Agent",

    description=(
        "A professional AI assistant for Indian "
        "weather and Indian cinema."
    ),

    version="2.0.0"
)


# ============================================================
# 17. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent",

    playground_type="default"
)


# ============================================================
# 18. ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {

        "status": "running",

        "service":
            "Indian Weather and Cinema Agent",

        "version":
            "2.0.0"

    }


# ============================================================
# 19. START SERVER
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