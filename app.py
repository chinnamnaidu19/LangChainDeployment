import os
import json
import requests

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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
# 3. WEATHER DESCRIPTIONS
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
# 5. CELSIUS TO FAHRENHEIT TOOL
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

        city_input = city.strip()

        if not city_input:

            return (
                "Please provide a valid Indian city "
                "or location."
            )


        # ----------------------------------------------------
        # City alias
        # ----------------------------------------------------

        search_city = CITY_ALIASES.get(
            city_input.lower(),
            city_input
        )


        # ----------------------------------------------------
        # Geocoding
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
                result.get(
                    "country_code",
                    ""
                ).upper()
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
        # Coordinates
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
        # Weather API
        # ----------------------------------------------------

        weather_url = (
            "https://api.open-meteo.com/v1/forecast"
        )


        weather_params = {

            "latitude":
                latitude,

            "longitude":
                longitude,

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


        weather_data = (
            weather_response.json()
        )


        if "current" not in weather_data:

            return (
                f"Weather information is currently "
                f"unavailable for {resolved_city}."
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


        description = (
            WEATHER_DESCRIPTIONS.get(
                weather_code,
                "Unknown weather condition"
            )
        )


        location_text = resolved_city


        if state:

            location_text += (
                f", {state}"
            )


        # ----------------------------------------------------
        # Structured JSON returned to agent
        # ----------------------------------------------------

        result = {

            "location":
                location_text,

            "country":
                "India",

            "temperature_celsius":
                temperature,

            "condition":
                description,

            "humidity":
                humidity,

            "wind_speed_kmh":
                wind_speed
        }


        return json.dumps(
            result
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

You are a professional AI assistant specialized ONLY in:

1. Indian weather
2. Weather of Indian cities and locations
3. Indian movies
4. Indian cinema
5. Indian movie genres
6. Celsius to Fahrenheit temperature conversion
7. Questions combining these supported capabilities


============================================================
WEATHER
============================================================

For every weather question:

- ALWAYS use get_weather.
- Never guess weather.
- Never invent temperature.
- Use the city requested by the user.
- Understand common names such as Vizag = Visakhapatnam.
- If multiple cities are requested, retrieve EACH city separately.
- Never ignore a requested city.

For multiple cities, clearly list every location.

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
MOVIES
============================================================

For movie questions:

- ALWAYS use search_movies.
- Only provide movies returned by the tool.
- Never invent movie names.
- Support available genres.
- If a genre is unavailable, clearly say that it is not
  currently available in the database.


Example:

Indian Action Movies

1. RRR
2. Vikram
3. Baahubali


============================================================
TEMPERATURE CONVERSION
============================================================

For Celsius to Fahrenheit questions:

- ALWAYS use change_to_f.
- Do not calculate manually.
- Clearly show both Celsius and Fahrenheit.

Example:

Temperature Conversion

Celsius: 30°C
Fahrenheit: 86°F


============================================================
COMBINED QUESTIONS
============================================================

If a user asks multiple supported questions:

- Answer EVERY supported part.
- Use every required tool.
- Do not ignore any part.

Example:

User:
"What is the weather in Vizag and Hyderabad and give me
some action movies?"

Response:

Weather Report

1. Visakhapatnam
   Temperature: ...
   Condition: ...

2. Hyderabad
   Temperature: ...
   Condition: ...


Indian Action Movies

1. RRR
2. Vikram
3. Baahubali


============================================================
IRRELEVANT QUESTIONS
============================================================

The following are outside your supported capabilities:

- Programming
- Coding
- Mathematics
- Recipes
- Politics
- Sports unrelated to Indian cinema
- General knowledge
- Medical advice
- Personal advice
- Technology questions
- Homework unrelated to the supported topics
- Any other unrelated topic

For an irrelevant question, DO NOT answer it.

Instead respond professionally:

"Thank you for your question. I’m currently specialized in
Indian weather, Indian cinema, and temperature conversion,
so I’m unable to assist with that topic. Please ask me about
weather, Indian cities, movies, movie genres, or temperature
conversion, and I’ll be happy to help."


============================================================
FINAL RESPONSE
============================================================

- Be professional.
- Be concise but useful.
- Use headings.
- Use numbered lists where appropriate.
- Do not return JSON to the user.
- Do not return Python dictionaries.
- Do not expose tool calls.
- Do not expose reasoning.
- Do not expose internal state.
- Do not hallucinate.
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
# 12. INPUT MODEL
# ============================================================

class AgentInput(BaseModel):

    input: str = Field(

        description=(
            "Ask about Indian weather, Indian movies, "
            "movie genres or temperature conversion."
        )
    )


# ============================================================
# 13. FORMAT AGENT INPUT
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

def extract_text_response(
    agent_output
):

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


    for message in reversed(
        messages
    ):

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
# 15. FIND TOOLS USED
# ============================================================

def get_tools_used(
    agent_output
):

    tool_names = []


    if not isinstance(
        agent_output,
        dict
    ):

        return tool_names


    messages = agent_output.get(
        "messages",
        []
    )


    for message in messages:

        # LangChain ToolMessage

        if (
            message.__class__.__name__
            == "ToolMessage"
        ):

            name = getattr(
                message,
                "name",
                ""
            )

            if name and name not in tool_names:

                tool_names.append(
                    name
                )


        # Dictionary fallback

        elif isinstance(
            message,
            dict
        ):

            name = message.get(
                "name",
                ""
            )

            if name and name not in tool_names:

                tool_names.append(
                    name
                )


    return tool_names


# ============================================================
# 16. LANGSERVE CHAIN
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
# 17. FASTAPI APP
# ============================================================

app = FastAPI(

    title="Indian Weather & Cinema AI",

    description=(
        "Professional AI assistant for Indian weather, "
        "Indian cinema and temperature conversion."
    ),

    version="3.0.0"
)


# ============================================================
# 18. PROFESSIONAL WEB UI
# ============================================================

HTML = r"""

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>IndiaAI • Weather & Cinema</title>


<style>

/* ============================================================
   RESET
============================================================ */

* {

    box-sizing:
        border-box;

    margin:
        0;

    padding:
        0;
}


/* ============================================================
   ROOT
============================================================ */

:root {

    --primary:
        #4f46e5;

    --primary2:
        #7c3aed;

    --blue:
        #2563eb;

    --green:
        #16a34a;

    --orange:
        #ea580c;

    --red:
        #dc2626;

    --text:
        #172033;

    --muted:
        #667085;

    --border:
        #e4e7ec;

    --background:
        #f6f8fc;

    --white:
        #ffffff;
}


/* ============================================================
   BODY
============================================================ */

body {

    min-height:
        100vh;

    font-family:
        Inter,
        "Segoe UI",
        Arial,
        sans-serif;

    color:
        var(--text);

    background:

        radial-gradient(
            circle at 10% 0%,
            rgba(99,102,241,0.16),
            transparent 30%
        ),

        radial-gradient(
            circle at 90% 100%,
            rgba(14,165,233,0.12),
            transparent 30%
        ),

        var(--background);
}


/* ============================================================
   HEADER
============================================================ */

.header {

    height:
        72px;

    padding:
        0 5%;

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;

    background:
        rgba(255,255,255,0.88);

    backdrop-filter:
        blur(16px);

    border-bottom:
        1px solid var(--border);

    position:
        sticky;

    top:
        0;

    z-index:
        100;
}


.logo-area {

    display:
        flex;

    align-items:
        center;

    gap:
        12px;
}


.logo {

    width:
        44px;

    height:
        44px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        13px;

    color:
        white;

    font-size:
        18px;

    font-weight:
        800;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary2)
        );

    box-shadow:
        0 8px 22px
        rgba(79,70,229,0.25);
}


.brand h1 {

    font-size:
        18px;

    margin-bottom:
        2px;
}


.brand p {

    color:
        var(--muted);

    font-size:
        10px;
}


.status {

    display:
        flex;

    align-items:
        center;

    gap:
        7px;

    padding:
        8px 13px;

    border-radius:
        30px;

    background:
        #ecfdf3;

    color:
        #15803d;

    font-size:
        11px;

    font-weight:
        700;
}


.status-dot {

    width:
        8px;

    height:
        8px;

    border-radius:
        50%;

    background:
        #22c55e;
}


/* ============================================================
   MAIN
============================================================ */

.container {

    max-width:
        1150px;

    margin:
        auto;

    padding:
        45px 20px 60px;
}


/* ============================================================
   HERO
============================================================ */

.hero {

    text-align:
        center;

    margin-bottom:
        35px;
}


.badge {

    display:
        inline-block;

    padding:
        7px 13px;

    border-radius:
        30px;

    background:
        #eef2ff;

    color:
        var(--primary);

    font-size:
        10px;

    font-weight:
        800;

    margin-bottom:
        15px;
}


.hero h2 {

    font-size:
        clamp(30px, 5vw, 46px);

    letter-spacing:
        -1.5px;

    margin-bottom:
        12px;
}


.hero p {

    max-width:
        690px;

    margin:
        auto;

    color:
        var(--muted);

    font-size:
        14px;

    line-height:
        1.7;
}


/* ============================================================
   CAPABILITIES
============================================================ */

.capabilities {

    display:
        grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap:
        14px;

    margin-bottom:
        25px;
}


.capability {

    padding:
        17px;

    border:
        1px solid var(--border);

    border-radius:
        17px;

    background:
        rgba(255,255,255,0.9);

    display:
        flex;

    align-items:
        center;

    gap:
        12px;

    transition:
        0.2s;
}


.capability:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 10px 25px
        rgba(16,24,40,0.06);
}


.cap-icon {

    width:
        40px;

    height:
        40px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        12px;

    font-size:
        19px;
}


.weather-icon {

    background:
        #eff6ff;
}


.movie-icon {

    background:
        #fdf2f8;
}


.temp-icon {

    background:
        #fff7ed;
}


.capability strong {

    display:
        block;

    font-size:
        13px;
}


.capability span {

    color:
        var(--muted);

    font-size:
        10px;

    display:
        block;

    margin-top:
        3px;
}


/* ============================================================
   CHAT CARD
============================================================ */

.chat-card {

    background:
        white;

    border:
        1px solid var(--border);

    border-radius:
        23px;

    overflow:
        hidden;

    box-shadow:
        0 18px 50px
        rgba(16,24,40,0.07);
}


/* ============================================================
   CHAT HEADER
============================================================ */

.chat-header {

    padding:
        17px 21px;

    border-bottom:
        1px solid var(--border);

    display:
        flex;

    align-items:
        center;

    justify-content:
        space-between;
}


.assistant-title {

    display:
        flex;

    align-items:
        center;

    gap:
        10px;
}


.bot {

    width:
        39px;

    height:
        39px;

    border-radius:
        12px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    background:
        #eef2ff;

    font-size:
        18px;
}


.assistant-title strong {

    display:
        block;

    font-size:
        13px;
}


.assistant-title span {

    color:
        var(--muted);

    font-size:
        10px;
}


.clear {

    border:
        none;

    background:
        transparent;

    color:
        #98a2b3;

    cursor:
        pointer;

    font-size:
        11px;
}


.clear:hover {

    color:
        var(--primary);
}


/* ============================================================
   CHAT BODY
============================================================ */

.chat-body {

    min-height:
        380px;

    max-height:
        550px;

    overflow-y:
        auto;

    padding:
        24px;

    background:
        linear-gradient(
            180deg,
            #fbfcff,
            #f8f9fd
        );
}


/* ============================================================
   WELCOME
============================================================ */

.welcome {

    max-width:
        600px;

    margin:
        35px auto;

    text-align:
        center;

    color:
        var(--muted);
}


.welcome-icon {

    width:
        65px;

    height:
        65px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    margin:
        0 auto 15px;

    border-radius:
        20px;

    background:
        linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff
        );

    font-size:
        28px;
}


.welcome h3 {

    color:
        var(--text);

    font-size:
        19px;

    margin-bottom:
        8px;
}


.welcome p {

    font-size:
        12px;

    line-height:
        1.7;
}


/* ============================================================
   SUGGESTIONS
============================================================ */

.suggestions {

    display:
        flex;

    flex-wrap:
        wrap;

    justify-content:
        center;

    gap:
        8px;

    margin-top:
        18px;
}


.suggestion {

    padding:
        9px 12px;

    border:
        1px solid #dfe3eb;

    background:
        white;

    border-radius:
        20px;

    cursor:
        pointer;

    color:
        #475467;

    font-size:
        10px;

    transition:
        0.2s;
}


.suggestion:hover {

    border-color:
        #a5b4fc;

    color:
        var(--primary);
}


/* ============================================================
   USER MESSAGE
============================================================ */

.message {

    display:
        flex;

    margin-bottom:
        18px;
}


.message.user {

    justify-content:
        flex-end;
}


.bubble {

    max-width:
        80%;

    padding:
        12px 15px;

    border-radius:
        17px;

    font-size:
        12px;

    line-height:
        1.6;

    white-space:
        pre-wrap;
}


.user .bubble {

    color:
        white;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary2)
        );

    border-bottom-right-radius:
        5px;
}


/* ============================================================
   RESULT CARDS
============================================================ */

.result {

    margin-bottom:
        18px;

    border:
        1px solid var(--border);

    border-radius:
        17px;

    background:
        white;

    overflow:
        hidden;

    box-shadow:
        0 7px 22px
        rgba(16,24,40,0.04);
}


.result-header {

    padding:
        14px 16px;

    display:
        flex;

    align-items:
        center;

    gap:
        10px;

    border-bottom:
        1px solid #edf0f4;
}


.result-icon {

    width:
        37px;

    height:
        37px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    border-radius:
        11px;

    font-size:
        17px;
}


.weather-card .result-icon {

    background:
        #eff6ff;
}


.movie-card .result-icon {

    background:
        #fdf2f8;
}


.temp-card .result-icon {

    background:
        #fff7ed;
}


.irrelevant-card .result-icon {

    background:
        #fef2f2;
}


.result-header strong {

    display:
        block;

    font-size:
        13px;
}


.result-header span {

    color:
        #98a2b3;

    font-size:
        9px;
}


.result-body {

    padding:
        16px;
}


/* ============================================================
   WEATHER
============================================================ */

.weather-grid {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        10px;
}


.weather-location {

    border:
        1px solid #e7ebf2;

    border-radius:
        14px;

    padding:
        14px;

    background:
        #fbfcff;
}


.weather-location h4 {

    font-size:
        13px;

    margin-bottom:
        12px;
}


.weather-temp {

    font-size:
        26px;

    font-weight:
        800;

    color:
        #2563eb;

    margin-bottom:
        6px;
}


.weather-condition {

    font-size:
        11px;

    color:
        #475467;

    margin-bottom:
        12px;
}


.weather-meta {

    display:
        grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap:
        7px;
}


.meta {

    padding:
        8px;

    border-radius:
        9px;

    background:
        white;

    border:
        1px solid #edf0f4;

    font-size:
        9px;

    color:
        #667085;
}


.meta strong {

    display:
        block;

    color:
        #344054;

    font-size:
        10px;

    margin-top:
        2px;
}


/* ============================================================
   MOVIES
============================================================ */

.movie-list {

    display:
        grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap:
        10px;
}


.movie {

    padding:
        13px;

    border:
        1px solid #f0d8e5;

    border-radius:
        13px;

    background:
        #fff8fb;

    text-align:
        center;

    font-size:
        11px;

    font-weight:
        700;

    color:
        #9d174d;
}


/* ============================================================
   TEMPERATURE
============================================================ */

.conversion {

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    gap:
        18px;

    padding:
        18px;

    border-radius:
        15px;

    background:
        #fffaf5;

    border:
        1px solid #fed7aa;
}


.conversion-value {

    text-align:
        center;
}


.conversion-value strong {

    display:
        block;

    font-size:
        27px;

    color:
        #c2410c;
}


.conversion-value span {

    color:
        #9a3412;

    font-size:
        10px;
}


.arrow {

    font-size:
        22px;

    color:
        #f97316;
}


/* ============================================================
   IRRELEVANT
============================================================ */

.irrelevant {

    padding:
        15px;

    border-radius:
        13px;

    background:
        #fff7f7;

    border:
        1px solid #fecaca;

    color:
        #7f1d1d;

    font-size:
        12px;

    line-height:
        1.7;
}


/* ============================================================
   INPUT
============================================================ */

.input-area {

    padding:
        17px;

    border-top:
        1px solid var(--border);

    background:
        white;
}


.input-box {

    display:
        flex;

    align-items:
        flex-end;

    gap:
        8px;

    padding:
        8px;

    border:
        1px solid #dfe3eb;

    border-radius:
        16px;

    background:
        #fbfcff;

    transition:
        0.2s;
}


.input-box:focus-within {

    border-color:
        #818cf8;

    box-shadow:
        0 0 0 4px
        rgba(99,102,241,0.08);
}


textarea {

    flex:
        1;

    min-height:
        43px;

    max-height:
        130px;

    resize:
        none;

    border:
        none;

    outline:
        none;

    background:
        transparent;

    padding:
        9px;

    font:
        inherit;

    font-size:
        12px;
}


.send {

    width:
        43px;

    height:
        43px;

    border:
        none;

    border-radius:
        12px;

    background:
        linear-gradient(
            135deg,
            var(--primary),
            var(--primary2)
        );

    color:
        white;

    cursor:
        pointer;

    font-size:
        17px;

    transition:
        0.2s;
}


.send:hover {

    transform:
        translateY(-2px);

    box-shadow:
        0 7px 17px
        rgba(79,70,229,0.25);
}


.send:disabled {

    opacity:
        0.5;

    cursor:
        not-allowed;
}


.footer-note {

    margin-top:
        7px;

    display:
        flex;

    justify-content:
        space-between;

    color:
        #98a2b3;

    font-size:
        9px;
}


/* ============================================================
   TYPING
============================================================ */

.typing {

    display:
        flex;

    gap:
        5px;

    padding:
        10px;
}


.typing span {

    width:
        7px;

    height:
        7px;

    border-radius:
        50%;

    background:
        #98a2b3;

    animation:
        bounce 1.2s infinite;
}


.typing span:nth-child(2) {

    animation-delay:
        0.15s;
}


.typing span:nth-child(3) {

    animation-delay:
        0.3s;
}


@keyframes bounce {

    0%,
    60%,
    100% {

        transform:
            translateY(0);

    }

    30% {

        transform:
            translateY(-5px);

    }
}


/* ============================================================
   RESPONSIVE
============================================================ */

@media(max-width: 750px) {

    .capabilities {

        grid-template-columns:
            1fr;

    }


    .weather-grid {

        grid-template-columns:
            1fr;

    }


    .movie-list {

        grid-template-columns:
            1fr;

    }


    .status {

        display:
            none;

    }

}


</style>

</head>


<body>


<!-- ==========================================================
     HEADER
========================================================== -->

<header class="header">

    <div class="logo-area">

        <div class="logo">
            IA
        </div>

        <div class="brand">

            <h1>
                IndiaAI Assistant
            </h1>

            <p>
                Weather • Cinema • Temperature
            </p>

        </div>

    </div>


    <div class="status">

        <span class="status-dot"></span>

        AI Online

    </div>

</header>


<!-- ==========================================================
     MAIN
========================================================== -->

<main class="container">


    <section class="hero">

        <div class="badge">
            ✦ SPECIALIZED AI ASSISTANT
        </div>

        <h2>
            Indian Weather & Cinema
        </h2>

        <p>
            Get live weather information for Indian cities,
            discover Indian movies by genre, and convert
            Celsius temperatures to Fahrenheit.
        </p>

    </section>


    <!-- CAPABILITIES -->

    <section class="capabilities">


        <div class="capability">

            <div class="cap-icon weather-icon">
                🌤️
            </div>

            <div>

                <strong>
                    Indian Weather
                </strong>

                <span>
                    Live city weather information
                </span>

            </div>

        </div>


        <div class="capability">

            <div class="cap-icon movie-icon">
                🎬
            </div>

            <div>

                <strong>
                    Indian Movies
                </strong>

                <span>
                    Movies by genre
                </span>

            </div>

        </div>


        <div class="capability">

            <div class="cap-icon temp-icon">
                🌡️
            </div>

            <div>

                <strong>
                    Temperature Conversion
                </strong>

                <span>
                    Celsius → Fahrenheit
                </span>

            </div>

        </div>


    </section>


    <!-- CHAT -->

    <section class="chat-card">


        <div class="chat-header">

            <div class="assistant-title">

                <div class="bot">
                    🤖
                </div>

                <div>

                    <strong>
                        IndiaAI
                    </strong>

                    <span>
                        Weather & Cinema Assistant
                    </span>

                </div>

            </div>


            <button
                class="clear"
                onclick="clearChat()"
            >
                Clear
            </button>

        </div>


        <!-- CHAT BODY -->

        <div
            class="chat-body"
            id="chatBody"
        >


            <div
                class="welcome"
                id="welcome"
            >

                <div class="welcome-icon">
                    ✨
                </div>

                <h3>
                    How can I help you?
                </h3>

                <p>
                    Ask about Indian weather, movies,
                    movie genres or temperature conversion.
                </p>


                <div class="suggestions">


                    <button
                        class="suggestion"
                        onclick="suggest('What is the weather in Vizag?')"
                    >
                        🌤️ Weather in Vizag
                    </button>


                    <button
                        class="suggestion"
                        onclick="suggest('Give me some action movies')"
                    >
                        🎬 Action movies
                    </button>


                    <button
                        class="suggestion"
                        onclick="suggest('Convert 30 Celsius to Fahrenheit')"
                    >
                        🌡️ Convert 30°C
                    </button>


                    <button
                        class="suggestion"
                        onclick="suggest('What is the weather in Delhi and Mumbai?')"
                    >
                        📍 Multiple cities
                    </button>

                </div>

            </div>


        </div>


        <!-- INPUT -->

        <div class="input-area">

            <div class="input-box">

                <textarea
                    id="question"
                    rows="1"
                    placeholder="Ask about Indian weather, movies or temperature..."
                    oninput="resizeInput(this)"
                ></textarea>


                <button
                    class="send"
                    id="send"
                    onclick="askQuestion()"
                >
                    ➤
                </button>

            </div>


            <div class="footer-note">

                <span>
                    IndiaAI • Specialized Assistant
                </span>

                <span>
                    Enter to send
                </span>

            </div>

        </div>


    </section>


</main>


<script>


// ==========================================================
// ELEMENTS
// ==========================================================

const question =
    document.getElementById(
        "question"
    );


const chatBody =
    document.getElementById(
        "chatBody"
    );


const sendButton =
    document.getElementById(
        "send"
    );


const welcome =
    document.getElementById(
        "welcome"
    );


// ==========================================================
// ASK QUESTION
// ==========================================================

async function askQuestion() {


    const text =
        question.value.trim();


    if (!text) {

        question.focus();

        return;

    }


    // Hide welcome

    if (welcome) {

        welcome.style.display =
            "none";

    }


    // Add user message

    addUserMessage(
        text
    );


    // Clear input

    question.value =
        "";

    resizeInput(
        question
    );


    // Disable

    sendButton.disabled =
        true;


    // Typing

    const typing =
        addTyping();


    try {


        const response =
            await fetch(
                "/ask",
                {

                    method:
                        "POST",

                    headers:
                        {
                            "Content-Type":
                                "application/json"
                        },

                    body:
                        JSON.stringify({
                            question:
                                text
                        })

                }
            );


        if (!response.ok) {

            throw new Error(
                "Server error"
            );

        }


        const data =
            await response.json();


        removeTyping(
            typing
        );


        // Display professional result

        displayResult(
            data
        );


    } catch (error) {


        console.error(
            error
        );


        removeTyping(
            typing
        );


        displayIrrelevant(
            "I’m sorry, I could not connect to the AI service right now. Please try again."
        );


    } finally {

        sendButton.disabled =
            false;

        question.focus();

    }

}


// ==========================================================
// USER MESSAGE
// ==========================================================

function addUserMessage(
    text
) {


    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message user";


    const bubble =
        document.createElement(
            "div"
        );


    bubble.className =
        "bubble";


    bubble.textContent =
        text;


    wrapper.appendChild(
        bubble
    );


    chatBody.appendChild(
        wrapper
    );


    scrollBottom();

}


// ==========================================================
// TYPING
// ==========================================================

function addTyping() {


    const id =
        "typing-" +
        Date.now();


    const wrapper =
        document.createElement(
            "div"
        );


    wrapper.className =
        "message";


    wrapper.id =
        id;


    wrapper.innerHTML = `

        <div class="bubble">

            <div class="typing">

                <span></span>
                <span></span>
                <span></span>

            </div>

        </div>

    `;


    chatBody.appendChild(
        wrapper
    );


    scrollBottom();


    return id;

}


// ==========================================================
// REMOVE TYPING
// ==========================================================

function removeTyping(
    id
) {

    const element =
        document.getElementById(
            id
        );


    if (element) {

        element.remove();

    }

}


// ==========================================================
// DISPLAY RESULT
// ==========================================================

function displayResult(
    data
) {


    const tools =
        data.tools_used || [];


    const answer =
        data.answer || "";


    const lower =
        answer.toLowerCase();


    // ------------------------------------------------------
    // Weather
    // ------------------------------------------------------

    if (
        tools.includes(
            "get_weather"
        )
    ) {

        displayWeather(
            answer
        );

    }


    // ------------------------------------------------------
    // Movies
    // ------------------------------------------------------

    if (
        tools.includes(
            "search_movies"
        )
    ) {

        displayMovies(
            answer
        );

    }


    // ------------------------------------------------------
    // Temperature
    // ------------------------------------------------------

    if (
        tools.includes(
            "change_to_f"
        )
    ) {

        displayTemperature(
            answer
        );

    }


    // ------------------------------------------------------
    // If no tool was used
    // ------------------------------------------------------

    if (
        tools.length === 0
    ) {

        displayIrrelevant(
            answer
        );

    }


    scrollBottom();

}


// ==========================================================
// WEATHER CARD
// ==========================================================

function displayWeather(
    answer
) {


    const result =
        document.createElement(
            "div"
        );


    result.className =
        "result weather-card";


    result.innerHTML = `

        <div class="result-header">

            <div class="result-icon">
                🌤️
            </div>

            <div>

                <strong>
                    Weather Report
                </strong>

                <span>
                    Live Indian weather information
                </span>

            </div>

        </div>


        <div class="result-body">

            <div class="weather-location">

                <h4>
                    Current Weather
                </h4>

                <div class="weather-condition">
                    ${formatAnswer(answer)}
                </div>

            </div>

        </div>

    `;


    chatBody.appendChild(
        result
    );

}


// ==========================================================
// MOVIE CARD
// ==========================================================

function displayMovies(
    answer
) {


    const result =
        document.createElement(
            "div"
        );


    result.className =
        "result movie-card";


    result.innerHTML = `

        <div class="result-header">

            <div class="result-icon">
                🎬
            </div>

            <div>

                <strong>
                    Indian Cinema
                </strong>

                <span>
                    Movies matching your request
                </span>

            </div>

        </div>


        <div class="result-body">

            <div class="test"
                 style="
                    white-space:pre-wrap;
                    font-size:12px;
                    line-height:1.7;
                 "
            >
                ${formatAnswer(answer)}
            </div>

        </div>

    `;


    chatBody.appendChild(
        result
    );

}


// ==========================================================
// TEMPERATURE CARD
// ==========================================================

function displayTemperature(
    answer
) {


    const result =
        document.createElement(
            "div"
        );


    result.className =
        "result temp-card";


    result.innerHTML = `

        <div class="result-header">

            <div class="result-icon">
                🌡️
            </div>

            <div>

                <strong>
                    Temperature Conversion
                </strong>

                <span>
                    Celsius to Fahrenheit
                </span>

            </div>

        </div>


        <div class="result-body">

            <div class="conversion">

                <div class="conversion-value">

                    <strong>
                        Result
                    </strong>

                    <span>
                        ${formatAnswer(answer)}
                    </span>

                </div>

            </div>

        </div>

    `;


    chatBody.appendChild(
        result
    );

}


// ==========================================================
// IRRELEVANT CARD
// ==========================================================

function displayIrrelevant(
    answer
) {


    const result =
        document.createElement(
            "div"
        );


    result.className =
        "result irrelevant-card";


    result.innerHTML = `

        <div class="result-header">

            <div class="result-icon">
                ℹ️
            </div>

            <div>

                <strong>
                    Outside Supported Topics
                </strong>

                <span>
                    IndiaAI Assistant
                </span>

            </div>

        </div>


        <div class="result-body">

            <div class="irrelevant">

                ${escapeHtml(answer)}

            </div>

        </div>

    `;


    chatBody.appendChild(
        result
    );

}


// ==========================================================
// FORMAT ANSWER
// ==========================================================

function formatAnswer(
    text
) {

    return escapeHtml(
        text
    );

}


// ==========================================================
// ESCAPE HTML
// ==========================================================

function escapeHtml(
    text
) {


    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        String(text);


    return div.innerHTML;

}


// ==========================================================
// SUGGESTION
// ==========================================================

function suggest(
    text
) {

    question.value =
        text;

    resizeInput(
        question
    );

    question.focus();

}


// ==========================================================
// CLEAR CHAT
// ==========================================================

function clearChat() {

    chatBody.innerHTML =
        "";

    const welcomeClone =
        document.createElement(
            "div"
        );


    welcomeClone.className =
        "welcome";


    welcomeClone.innerHTML = `

        <div class="welcome-icon">
            ✨
        </div>

        <h3>
            How can I help you?
        </h3>

        <p>
            Ask about Indian weather, movies,
            movie genres or temperature conversion.
        </p>

        <div class="suggestions">

            <button
                class="suggestion"
                onclick="suggest('What is the weather in Vizag?')"
            >
                🌤️ Weather in Vizag
            </button>

            <button
                class="suggestion"
                onclick="suggest('Give me some action movies')"
            >
                🎬 Action movies
            </button>

            <button
                class="suggestion"
                onclick="suggest('Convert 30 Celsius to Fahrenheit')"
            >
                🌡️ Convert 30°C
            </button>

            <button
                class="suggestion"
                onclick="suggest('What is the weather in Delhi and Mumbai?')"
            >
                📍 Multiple cities
            </button>

        </div>

    `;


    chatBody.appendChild(
        welcomeClone
    );

}


// ==========================================================
// INPUT RESIZE
// ==========================================================

function resizeInput(
    element
) {

    element.style.height =
        "auto";


    element.style.height =
        Math.min(
            element.scrollHeight,
            130
        ) + "px";

}


// ==========================================================
// ENTER KEY
// ==========================================================

question.addEventListener(
    "keydown",
    function(event) {

        if (
            event.key === "Enter"
            &&
            !event.shiftKey
        ) {

            event.preventDefault();

            askQuestion();

        }

    }
);


// ==========================================================
// SCROLL
// ==========================================================

function scrollBottom() {

    setTimeout(
        function() {

            chatBody.scrollTo({

                top:
                    chatBody.scrollHeight,

                behavior:
                    "smooth"

            });

        },
        50
    );

}


</script>

</body>

</html>

"""


# ============================================================
# 19. ROOT PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    return HTML


# ============================================================
# 20. CUSTOM ASK API
# ============================================================

@app.post("/ask")
async def ask(
    payload: dict
):

    question = (
        payload.get(
            "question",
            ""
        )
        .strip()
    )


    if not question:

        return {

            "answer":
                "Please enter a question.",

            "tools_used":
                []

        }


    # --------------------------------------------------------
    # Run agent
    # --------------------------------------------------------

    agent_input = {

        "messages": [

            {
                "role":
                    "user",

                "content":
                    question
            }

        ]

    }


    result = agent.invoke(
        agent_input
    )


    answer = extract_text_response(
        result
    )


    tools_used = get_tools_used(
        result
    )


    return {

        "answer":
            answer,

        "tools_used":
            tools_used

    }


# ============================================================
# 21. HEALTH
# ============================================================

@app.get("/health")
async def health():

    return {

        "status":
            "ok",

        "service":
            "Indian Weather and Cinema AI",

        "tools": [

            "get_weather",

            "search_movies",

            "change_to_f"

        ]

    }


# ============================================================
# 22. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/agent",

    playground_type="default"

)


# ============================================================
# 23. SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn


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