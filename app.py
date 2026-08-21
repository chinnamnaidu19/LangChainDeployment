import os
import json
import requests
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langserve import add_routes
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""

    movies = {
        "sci-fi": "Cargo, 2.0, Mr. India",
        "comedy": "3 Idiots, Hera Pheri, Munna Bhai M.B.B.S.",
        "action": "RRR, Vikram, Baahubali"
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


@tool
def change_to_f(temp_c: float) -> float:
    """Convert Celsius temperature to Fahrenheit."""

    return round((temp_c * 1.8) + 32, 2)


@tool
def get_weather(city: str) -> str:
    """Get current temperature for an Indian city."""

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"

        geo_params = {
            "name": city,
            "count": 10,
            "language": "en",
            "format": "json"
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=10
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find weather data for city: {city}"

        location = None

        for result in geo_data["results"]:
            if result.get("country_code", "").upper() == "IN":
                location = result
                break

        if location is None:
            location = geo_data["results"][0]

        latitude = location["latitude"]
        longitude = location["longitude"]
        resolved_city = location.get("name", city)

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "temperature_unit": "celsius"
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()
        current = weather_data["current"]

        result = {
            "resolved_city": resolved_city,
            "temperature_celsius": current["temperature_2m"],
            "weather_code": current["weather_code"]
        }

        return json.dumps(result)

    except requests.RequestException:
        return (
            f"Unable to retrieve weather for {city}. "
            f"Please try again later."
        )

    except Exception:
        return f"Could not retrieve weather information for {city}."


tools = [
    get_weather,
    search_movies,
    change_to_f
]


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set."
    )


llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)


SYSTEM_PROMPT = """
You are a specialized Indian Weather and Indian Cinema assistant.

You are authorized to answer questions about:

1. Indian weather
2. Weather of Indian cities
3. Indian movies
4. Indian cinema
5. Indian movie genres
6. Questions combining Indian weather and Indian movies

For weather questions, use the get_weather tool.

If the user mentions multiple Indian cities, retrieve weather information
for all requested cities and include all of them in the final answer.

For movie questions, use the search_movies tool.

The user may request any movie genre such as:
comedy, action, sci-fi, horror, romance, thriller, drama, fantasy, etc.

An unavailable movie genre is NOT an unauthorized question.

If a requested movie genre is not available in the database, tell the user
that no Indian movies for that genre were found in the database.

For example:

Unfortunately, no Indian horror movies were found in the database.

If the user asks for weather and movies together, answer both parts in
one normal text response.

For example:

The current weather is:
Vizag: 32°C
Hyderabad: 29°C
Chennai: 34°C

Unfortunately, no Indian horror movies were found in the database.

For a valid movie genre, provide the movies returned by the tool.

For completely unrelated topics outside Indian weather and Indian cinema,
respond EXACTLY:

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


agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)


class AgentInput(BaseModel):
    input: str = Field(
        description="Your message to the Indian Weather and Cinema Agent"
    )


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

    for message in reversed(messages):

        if message.__class__.__name__ == "AIMessage":

            content = getattr(message, "content", "")

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if isinstance(item, dict):
                        text = item.get("text")

                        if text:
                            text_parts.append(str(text))

                    elif isinstance(item, str):
                        text_parts.append(item)

                return "\n".join(text_parts).strip()

        if isinstance(message, dict):

            message_type = message.get("type", "")

            if message_type in ["ai", "AIMessage"]:

                content = message.get("content", "")

                if isinstance(content, str):
                    return content.strip()

    last_message = messages[-1]

    content = getattr(
        last_message,
        "content",
        str(last_message)
    )

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


app = FastAPI(
    title="Indian Weather and Cinema Agent",
    version="1.0.0"
)


add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default"
)


@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Indian Weather and Cinema Agent"
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )