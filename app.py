import os
import json
from urllib.parse import quote

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from langserve import add_routes
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# ============================================================
# 1. INDIAN MOVIE DATABASE
# ============================================================

MOVIES = {
    "action": ["RRR", "Vikram", "Baahubali"],
    "comedy": ["3 Idiots", "Hera Pheri", "Munna Bhai M.B.B.S."],
    "sci-fi": ["Cargo", "2.0", "Mr. India"],
    "romance": ["Sita Ramam", "Geetha Govindam", "Jab We Met"],
    "thriller": ["Drishyam", "Ratsasan", "Andhadhun"],
    "drama": ["Taare Zameen Par", "Dangal", "12th Fail"],
    "fantasy": ["Baahubali", "Eega", "Tumbbad"],
    "horror": ["Tumbbad", "Stree", "Bhool Bhulaiyaa"],
}


# ============================================================
# 2. CITY ALIASES
# ============================================================

CITY_ALIASES = {
    "vizag": "Visakhapatnam",
    "vishakapatnam": "Visakhapatnam",
    "bombay": "Mumbai",
    "calcutta": "Kolkata",
    "madras": "Chennai",
    "bengaluru": "Bangalore",
    "bangalore": "Bangalore",
    "new delhi": "Delhi",
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
    99: "Thunderstorm with heavy hail",
}


# ============================================================
# 4. MOVIE SEARCH TOOL
# ============================================================

@tool
def search_movies(genre: str) -> str:
    """Search for Indian movies by genre."""
    requested_genre = (genre or "").strip().lower()

    if requested_genre in MOVIES:
        movie_list = ", ".join(MOVIES[requested_genre])
        return f"Indian {requested_genre} movies: {movie_list}"

    available_genres = ", ".join(sorted(MOVIES.keys()))
    return (
        f"No Indian movies are currently available for the genre "
        f"'{requested_genre}'. Available genres include: {available_genres}."
    )


# ============================================================
# 5. LIVE INDIAN WEATHER TOOL
# ============================================================

@tool
def get_weather(city: str) -> str:
    """
    Get live weather for an Indian city.

    Strategy:
    1. Use known Indian-city coordinates when available.
    2. Otherwise use Open-Meteo geocoding.
    3. Try Open-Meteo forecast API.
    4. Fall back to wttr.in if needed.
    """
    city_input = (city or "").strip()
    if not city_input:
        return "Please provide a valid Indian city or location."

    search_city = CITY_ALIASES.get(city_input.lower(), city_input)

    CITY_COORDINATES = {
        "delhi": (28.6139, 77.2090, "Delhi", "Delhi"),
        "new delhi": (28.6139, 77.2090, "Delhi", "Delhi"),
        "mumbai": (19.0760, 72.8777, "Mumbai", "Maharashtra"),
        "hyderabad": (17.3850, 78.4867, "Hyderabad", "Telangana"),
        "bangalore": (12.9716, 77.5946, "Bangalore", "Karnataka"),
        "bengaluru": (12.9716, 77.5946, "Bangalore", "Karnataka"),
        "chennai": (13.0827, 80.2707, "Chennai", "Tamil Nadu"),
        "kolkata": (22.5726, 88.3639, "Kolkata", "West Bengal"),
        "pune": (18.5204, 73.8567, "Pune", "Maharashtra"),
        "visakhapatnam": (17.6868, 83.2185, "Visakhapatnam", "Andhra Pradesh"),
        "vizag": (17.6868, 83.2185, "Visakhapatnam", "Andhra Pradesh"),
        "vijayawada": (16.5062, 80.6480, "Vijayawada", "Andhra Pradesh"),
        "tadepalligudem": (16.8147, 81.5260, "Tadepalligudem", "Andhra Pradesh"),
        "kochi": (9.9312, 76.2673, "Kochi", "Kerala"),
        "ahmedabad": (23.0225, 72.5714, "Ahmedabad", "Gujarat"),
        "jaipur": (26.9124, 75.7873, "Jaipur", "Rajasthan"),
        "lucknow": (26.8467, 80.9462, "Lucknow", "Uttar Pradesh"),
        "bhubaneswar": (20.2961, 85.8245, "Bhubaneswar", "Odisha"),
        "patna": (25.5941, 85.1376, "Patna", "Bihar"),
        "nagpur": (21.1458, 79.0882, "Nagpur", "Maharashtra"),
        "tirupati": (13.6288, 79.4192, "Tirupati", "Andhra Pradesh"),
        "warangal": (17.9689, 79.5941, "Warangal", "Telangana"),
        "guntur": (16.3067, 80.4365, "Guntur", "Andhra Pradesh"),
        "nellore": (14.4426, 79.9865, "Nellore", "Andhra Pradesh"),
    }

    headers = {
        "User-Agent": "India-Weather-Movies-Assistant/4.0",
        "Accept": "application/json",
    }

    normalized = search_city.lower().strip()
    latitude = longitude = None
    resolved_city = search_city
    state = ""

    # Known coordinates first.
    if normalized in CITY_COORDINATES:
        latitude, longitude, resolved_city, state = CITY_COORDINATES[normalized]

    # Geocode other Indian cities.
    if latitude is None:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": search_city,
            "count": 10,
            "language": "en",
            "format": "json",
            "countryCode": "IN",
        }

        try:
            response = requests.get(
                geo_url, params=geo_params, headers=headers, timeout=10
            )
            response.raise_for_status()
            payload = response.json()
            results = payload.get("results") or []
            indian = next(
                (r for r in results if str(r.get("country_code", "")).upper() == "IN"),
                None,
            )
            if indian:
                latitude = float(indian["latitude"])
                longitude = float(indian["longitude"])
                resolved_city = indian.get("name", search_city)
                state = indian.get("admin1", "")
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            print(f"GEOCODING ERROR: {exc}")

    # Open-Meteo weather.
    if latitude is not None and longitude is not None:
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
            "forecast_days": 1,
        }

        try:
            response = requests.get(
                weather_url, params=weather_params, headers=headers, timeout=15
            )
            response.raise_for_status()
            payload = response.json()
            current = payload.get("current")

            if isinstance(current, dict):
                location_text = resolved_city + (f", {state}" if state else "")
                result = {
                    "location": location_text,
                    "country": "India",
                    "temperature_celsius": current.get("temperature_2m"),
                    "condition": WEATHER_DESCRIPTIONS.get(
                        current.get("weather_code"), "Unknown weather condition"
                    ),
                    "humidity": current.get("relative_humidity_2m"),
                    "wind_speed_kmh": current.get("wind_speed_10m"),
                }
                return json.dumps(result)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            print(f"OPEN-METEO ERROR: {exc}")

    # Backup provider.
    try:
        wttr_url = f"https://wttr.in/{quote(city_input)}"
        response = requests.get(
            wttr_url,
            params={"format": "j1"},
            headers=headers,
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        current_list = payload.get("current_condition") or []

        if current_list:
            current = current_list[0]
            desc_list = current.get("weatherDesc") or []
            description = (
                desc_list[0].get("value", "Current conditions")
                if desc_list and isinstance(desc_list[0], dict)
                else "Current conditions"
            )

            area_list = payload.get("nearest_area") or []
            area_name, area_state = resolved_city, state
            if area_list:
                area = area_list[0]
                if area.get("areaName") and isinstance(area["areaName"][0], dict):
                    area_name = area["areaName"][0].get("value", area_name)
                if area.get("region") and isinstance(area["region"][0], dict):
                    area_state = area["region"][0].get("value", area_state)

            result = {
                "location": area_name + (f", {area_state}" if area_state else ""),
                "country": "India",
                "temperature_celsius": float(current.get("temp_C")),
                "condition": description,
                "humidity": float(current.get("humidity")),
                "wind_speed_kmh": float(current.get("windspeedKmph")),
            }
            return json.dumps(result)
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        print(f"WTTR.IN ERROR: {exc}")

    return (
        f"Live weather for {city_input} is temporarily unavailable. "
        "Please try again shortly."
    )


# ============================================================
# 6. AGENT CONFIGURATION
# ============================================================

TOOLS = [get_weather, search_movies]

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not configured.")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

SYSTEM_PROMPT = """
You are India Weather & Movies Assistant, a professional AI assistant specialized ONLY in:

1. Live weather in India
2. Weather of Indian cities and locations
3. Indian movies and Indian cinema
4. Indian movie genres and recommendations based only on the movie database tool
5. Questions combining Indian weather and Indian movies

============================================================
WEATHER RULES
============================================================

For every weather question:
- ALWAYS use get_weather.
- Never guess or invent weather information.
- Use the exact city/location requested by the user.
- Understand common city aliases such as Vizag = Visakhapatnam.
- If multiple cities are requested, retrieve EACH city separately.
- Clearly list every requested location.
- Weather temperatures should be shown in Celsius.

Example:

Weather Report

1. Visakhapatnam, Andhra Pradesh
   Temperature: 31°C
   Condition: Partly cloudy
   Humidity: 70%
   Wind Speed: 12 km/h

============================================================
MOVIE RULES
============================================================

For movie questions:
- ALWAYS use search_movies when the user asks for movies by genre.
- Only provide movie names returned by the tool.
- Never invent movie names.
- Supported genres include action, comedy, sci-fi, romance, thriller, drama, fantasy and horror.
- If the requested genre is unavailable, clearly explain that it is not currently available.

Example:

Indian Action Movies

1. RRR
2. Vikram
3. Baahubali

============================================================
COMBINED QUESTIONS
============================================================

If a user asks about weather and movies together:
- Answer EVERY supported part.
- Use every required tool.
- Do not ignore any part.

============================================================
OUT-OF-SCOPE QUESTIONS
============================================================

Do not answer questions about programming, coding, mathematics, recipes, politics,
sports unrelated to Indian cinema, medical advice, personal advice, technology,
homework unrelated to weather/movies, or other unrelated topics.

For an irrelevant question, respond:

"Thank you for your question. I’m currently specialized in Indian weather and Indian
movies, so I’m unable to assist with that topic. Please ask me about Indian weather,
Indian cities, Indian movies, or movie genres, and I’ll be happy to help."

============================================================
FINAL RESPONSE STYLE
============================================================

- Be professional, friendly and concise.
- Use clear headings.
- Use numbered lists where appropriate.
- Do not return JSON to the user.
- Do not expose tool calls, reasoning, internal state or API details.
- Do not hallucinate.
"""

agent = create_agent(
    model=llm_flash,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


# ============================================================
# 7. INPUT / OUTPUT HELPERS
# ============================================================

class AgentInput(BaseModel):
    input: str = Field(
        description="Ask about Indian weather, Indian cities, Indian movies or movie genres."
    )


def format_for_agent(x):
    user_input = x.get("input", "") if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}


def extract_text_response(agent_output):
    if isinstance(agent_output, str):
        return agent_output.strip()

    if not isinstance(agent_output, dict):
        return str(agent_output).strip()

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
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
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                    elif isinstance(item, str):
                        parts.append(item)
                return "\n".join(parts).strip()

    content = getattr(messages[-1], "content", str(messages[-1]))
    return content.strip() if isinstance(content, str) else str(content).strip()


def get_tools_used(agent_output):
    tool_names = []
    if not isinstance(agent_output, dict):
        return tool_names

    messages = agent_output.get("messages", [])
    for message in messages:
        if message.__class__.__name__ == "ToolMessage":
            name = getattr(message, "name", "")
            if name and name not in tool_names:
                tool_names.append(name)
        elif isinstance(message, dict):
            name = message.get("name", "")
            if name and name not in tool_names:
                tool_names.append(name)
    return tool_names


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)


# ============================================================
# 8. FASTAPI APP
# ============================================================

app = FastAPI(
    title="India Weather & Movies Assistant",
    description="Professional AI assistant for Indian weather and Indian movies.",
    version="4.0.0",
)


# ============================================================
# 9. MODERN WEB UI
# ============================================================

HTML = r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>India Weather & Movies Assistant</title>
<style>
:root{
    --navy:#071a2f;
    --blue:#087ea4;
    --cyan:#13b8d4;
    --sky:#eaf8ff;
    --gold:#f4b942;
    --text:#142033;
    --muted:#667085;
    --border:#dce7ef;
    --white:#ffffff;
    --bg:#f4f9fc;
}
*{box-sizing:border-box;margin:0;padding:0}
body{
    min-height:100vh;
    font-family:Inter,"Segoe UI",Arial,sans-serif;
    color:var(--text);
    background:
      radial-gradient(circle at 10% 0%,rgba(19,184,212,.16),transparent 30%),
      radial-gradient(circle at 95% 90%,rgba(244,185,66,.13),transparent 28%),
      var(--bg);
}
.header{
    height:76px;padding:0 5%;display:flex;align-items:center;justify-content:space-between;
    background:rgba(255,255,255,.92);backdrop-filter:blur(16px);
    border-bottom:1px solid var(--border);position:sticky;top:0;z-index:20
}
.brand-wrap{display:flex;align-items:center;gap:12px}
.logo{
    width:46px;height:46px;border-radius:14px;display:grid;place-items:center;color:white;
    font-weight:900;background:linear-gradient(135deg,var(--navy),var(--blue));
    box-shadow:0 9px 24px rgba(8,126,164,.24)
}
.brand h1{font-size:18px;line-height:1.1}.brand p{font-size:10px;color:var(--muted);margin-top:4px}
.online{padding:8px 13px;border-radius:30px;background:#ecfdf3;color:#15803d;font-size:11px;font-weight:800}
.online span{display:inline-block;width:7px;height:7px;background:#22c55e;border-radius:50%;margin-right:6px}
.container{max-width:1180px;margin:auto;padding:42px 20px 60px}
.hero{text-align:center;margin-bottom:30px}
.badge{display:inline-block;padding:7px 13px;border-radius:30px;background:#e7f8fd;color:#067392;font-size:10px;font-weight:900;letter-spacing:.4px;margin-bottom:14px}
.hero h2{font-size:clamp(31px,5vw,48px);letter-spacing:-1.8px;margin-bottom:11px;color:var(--navy)}
.hero h2 em{font-style:normal;color:var(--blue)}
.hero p{max-width:720px;margin:auto;color:var(--muted);font-size:14px;line-height:1.7}
.features{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:24px}
.feature{background:rgba(255,255,255,.9);border:1px solid var(--border);border-radius:18px;padding:18px;display:flex;align-items:center;gap:13px;transition:.2s}
.feature:hover{transform:translateY(-2px);box-shadow:0 12px 28px rgba(16,24,40,.07)}
.icon{width:44px;height:44px;border-radius:13px;display:grid;place-items:center;font-size:21px}.weather{background:#eaf6ff}.movie{background:#fff4df}
.feature strong{display:block;font-size:14px}.feature span{display:block;color:var(--muted);font-size:10px;margin-top:4px}
.chat{background:white;border:1px solid var(--border);border-radius:24px;overflow:hidden;box-shadow:0 20px 60px rgba(16,24,40,.08)}
.chat-head{padding:17px 21px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.assistant{display:flex;align-items:center;gap:10px}.bot{width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:#eaf8ff;font-size:19px}.assistant strong{display:block;font-size:13px}.assistant small{color:var(--muted);font-size:10px}
.clear{border:0;background:transparent;color:#98a2b3;cursor:pointer;font-size:11px}.clear:hover{color:var(--blue)}
.body{min-height:410px;max-height:570px;overflow-y:auto;padding:25px;background:linear-gradient(180deg,#fbfeff,#f7fafc)}
.welcome{max-width:650px;margin:42px auto;text-align:center;color:var(--muted)}
.welcome-icon{width:72px;height:72px;margin:0 auto 15px;border-radius:22px;display:grid;place-items:center;font-size:32px;background:linear-gradient(135deg,#e7f8fd,#fff4df)}
.welcome h3{font-size:20px;color:var(--navy);margin-bottom:8px}.welcome p{font-size:12px;line-height:1.7}
.suggestions{display:flex;flex-wrap:wrap;justify-content:center;gap:8px;margin-top:19px}
.suggestion{padding:10px 13px;border:1px solid #d8e5ed;background:white;border-radius:22px;cursor:pointer;color:#475467;font-size:10px;transition:.2s}
.suggestion:hover{border-color:#6bd3e4;color:var(--blue);background:#f5fdff}
.message{display:flex;margin-bottom:18px}.message.user{justify-content:flex-end}.bubble{max-width:82%;padding:12px 15px;border-radius:17px;font-size:12px;line-height:1.65;white-space:pre-wrap}
.user .bubble{color:white;background:linear-gradient(135deg,var(--navy),var(--blue));border-bottom-right-radius:5px}
.result{margin-bottom:18px;border:1px solid var(--border);border-radius:17px;background:white;overflow:hidden;box-shadow:0 7px 22px rgba(16,24,40,.04)}
.result-head{padding:14px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #edf0f4}.result-icon{width:37px;height:37px;border-radius:11px;display:grid;place-items:center;font-size:17px}.weather-card .result-icon{background:#eaf6ff}.movie-card .result-icon{background:#fff4df}.result-head strong{display:block;font-size:13px}.result-head span{color:#98a2b3;font-size:9px}.result-body{padding:16px}
.weather-box{border:1px solid #e7edf2;border-radius:15px;padding:15px;background:#fbfdff}.weather-title{font-size:13px;font-weight:800;margin-bottom:10px}.weather-text{font-size:12px;line-height:1.75;color:#475467;white-space:pre-wrap}
.movie-list{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.movie{padding:14px;border:1px solid #f2dfb8;border-radius:13px;background:#fffaf0;text-align:center;font-size:11px;font-weight:800;color:#8a5a00}
.answer-text{font-size:12px;line-height:1.75;color:#475467;white-space:pre-wrap}
.input-area{padding:17px;border-top:1px solid var(--border);background:white}.input-box{display:flex;align-items:flex-end;gap:8px;padding:8px;border:1px solid #d8e5ed;border-radius:17px;background:#fbfdff}.input-box:focus-within{border-color:#50bfd5;box-shadow:0 0 0 4px rgba(19,184,212,.08)}
textarea{flex:1;min-height:43px;max-height:130px;resize:none;border:0;outline:0;background:transparent;padding:9px;font:inherit;font-size:12px}.send{width:44px;height:44px;border:0;border-radius:13px;background:linear-gradient(135deg,var(--navy),var(--blue));color:white;cursor:pointer;font-size:17px;transition:.2s}.send:hover{transform:translateY(-2px);box-shadow:0 8px 18px rgba(8,126,164,.25)}.send:disabled{opacity:.5;cursor:not-allowed}
.footer{margin-top:7px;display:flex;justify-content:space-between;color:#98a2b3;font-size:9px}.typing{display:flex;gap:5px;padding:10px}.typing span{width:7px;height:7px;border-radius:50%;background:#98a2b3;animation:bounce 1.2s infinite}.typing span:nth-child(2){animation-delay:.15s}.typing span:nth-child(3){animation-delay:.3s}@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}
@media(max-width:750px){.features{grid-template-columns:1fr}.movie-list{grid-template-columns:1fr}.online{display:none}.header{padding:0 18px}.container{padding:30px 14px 45px}.body{min-height:430px}}
</style>
</head>
<body>
<header class="header">
  <div class="brand-wrap">
    <div class="logo">IN</div>
    <div class="brand">
      <h1>India Weather & Movies</h1>
      <p>Live Weather • Indian Cinema</p>
    </div>
  </div>
  <div class="online"><span></span>AI ONLINE</div>
</header>

<main class="container">
  <section class="hero">
    <div class="badge">✦ INDIA SPECIALIST ASSISTANT</div>
    <h2>India <em>Weather</em> & Movies</h2>
    <p>Ask for live weather in Indian cities or discover Indian movies by genre. Simple, fast and focused on India.</p>
  </section>

  <section class="features">
    <div class="feature">
      <div class="icon weather">🌦️</div>
      <div><strong>Indian Weather</strong><span>Live conditions for Indian cities</span></div>
    </div>
    <div class="feature">
      <div class="icon movie">🎬</div>
      <div><strong>Indian Movies</strong><span>Explore movies by genre</span></div>
    </div>
  </section>

  <section class="chat">
    <div class="chat-head">
      <div class="assistant">
        <div class="bot">🤖</div>
        <div><strong>India Weather & Movies Assistant</strong><small>Ask me about Indian weather or cinema</small></div>
      </div>
      <button class="clear" onclick="clearChat()">Clear chat</button>
    </div>

    <div class="body" id="chatBody">
      <div class="welcome" id="welcome">
        <div class="welcome-icon">🇮🇳</div>
        <h3>How can I help you?</h3>
        <p>I can check Indian city weather and help you discover Indian movies by genre.</p>
        <div class="suggestions">
          <button class="suggestion" onclick="suggest('What is the weather in Hyderabad?')">🌦️ Hyderabad weather</button>
          <button class="suggestion" onclick="suggest('What is the weather in Vizag?')">🌤️ Vizag weather</button>
          <button class="suggestion" onclick="suggest('Give me some action movies')">🎬 Action movies</button>
          <button class="suggestion" onclick="suggest('Give me some thriller movies')">🍿 Thriller movies</button>
        </div>
      </div>
    </div>

    <div class="input-area">
      <div class="input-box">
        <textarea id="question" rows="1" placeholder="Ask about Indian weather or movies..." oninput="resizeInput(this)"></textarea>
        <button class="send" id="send" onclick="askQuestion()">➤</button>
      </div>
      <div class="footer"><span>India Weather & Movies AI</span><span>Enter to send</span></div>
    </div>
  </section>
</main>

<script>
const question = document.getElementById('question');
const chatBody = document.getElementById('chatBody');
const sendButton = document.getElementById('send');
const welcome = document.getElementById('welcome');

async function askQuestion(){
  const text = question.value.trim();
  if(!text){ question.focus(); return; }
  if(welcome) welcome.style.display='none';
  addUserMessage(text);
  question.value='';
  resizeInput(question);
  sendButton.disabled=true;
  const typing=addTyping();

  try{
    const response=await fetch('/ask',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:text})
    });
    if(!response.ok) throw new Error('Server error');
    const data=await response.json();
    removeTyping(typing);
    displayResult(data);
  }catch(error){
    console.error(error);
    removeTyping(typing);
    displayIrrelevant('I could not connect to the AI service right now. Please try again.');
  }finally{
    sendButton.disabled=false;
    question.focus();
  }
}

function addUserMessage(text){
  const wrapper=document.createElement('div'); wrapper.className='message user';
  const bubble=document.createElement('div'); bubble.className='bubble'; bubble.textContent=text;
  wrapper.appendChild(bubble); chatBody.appendChild(wrapper); scrollBottom();
}

function addTyping(){
  const id='typing-'+Date.now();
  const wrapper=document.createElement('div'); wrapper.className='message'; wrapper.id=id;
  wrapper.innerHTML='<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>';
  chatBody.appendChild(wrapper); scrollBottom(); return id;
}

function removeTyping(id){ const el=document.getElementById(id); if(el) el.remove(); }

function displayResult(data){
  const tools=data.tools_used||[];
  const answer=data.answer||'';
  if(tools.includes('get_weather')) displayWeather(answer);
  if(tools.includes('search_movies')) displayMovies(answer);
  if(tools.length===0) displayIrrelevant(answer);
  scrollBottom();
}

function displayWeather(answer){
  const result=document.createElement('div'); result.className='result weather-card';
  result.innerHTML=`<div class="result-head"><div class="result-icon">🌦️</div><div><strong>Indian Weather Report</strong><span>Live weather information</span></div></div><div class="result-body"><div class="weather-box"><div class="weather-title">Current conditions</div><div class="weather-text">${escapeHtml(answer)}</div></div></div>`;
  chatBody.appendChild(result);
}

function displayMovies(answer){
  const result=document.createElement('div'); result.className='result movie-card';
  const movies=extractMovies(answer);
  let content='';
  if(movies.length){
    content='<div class="movie-list">'+movies.map(m=>`<div class="movie">🎬 ${escapeHtml(m)}</div>`).join('')+'</div>';
  }else{
    content=`<div class="answer-text">${escapeHtml(answer)}</div>`;
  }
  result.innerHTML=`<div class="result-head"><div class="result-icon">🎬</div><div><strong>Indian Cinema</strong><span>Movies by genre</span></div></div><div class="result-body">${content}</div>`;
  chatBody.appendChild(result);
}

function extractMovies(answer){
  const lines=answer.split('\n').map(x=>x.trim()).filter(Boolean);
  return lines.map(line=>line.replace(/^\d+[.)]\s*/,'').replace(/^[-•]\s*/,'')).filter(line=>{
    const l=line.toLowerCase();
    return l && !l.includes('indian action movies') && !l.includes('indian comedy movies') && !l.includes('indian thriller movies') && !l.includes('indian romance movies') && !l.includes('indian drama movies') && !l.includes('indian horror movies') && !l.includes('indian fantasy movies') && !l.includes('indian sci-fi movies');
  }).slice(0,12);
}

function displayIrrelevant(answer){
  const result=document.createElement('div'); result.className='result';
  result.innerHTML=`<div class="result-head"><div class="result-icon">💬</div><div><strong>Assistant</strong><span>India Weather & Movies</span></div></div><div class="result-body"><div class="answer-text">${escapeHtml(answer)}</div></div>`;
  chatBody.appendChild(result);
}

function escapeHtml(text){
  const div=document.createElement('div'); div.textContent=text; return div.innerHTML;
}

function suggest(text){ question.value=text; resizeInput(question); askQuestion(); }
function resizeInput(el){ el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,130)+'px'; }
function clearChat(){
  chatBody.innerHTML='';
  const div=document.createElement('div'); div.className='welcome'; div.id='welcome';
  div.innerHTML='<div class="welcome-icon">🇮🇳</div><h3>How can I help you?</h3><p>I can check Indian city weather and help you discover Indian movies by genre.</p><div class="suggestions"><button class="suggestion" onclick="suggest(\'What is the weather in Hyderabad?\')">🌦️ Hyderabad weather</button><button class="suggestion" onclick="suggest(\'What is the weather in Vizag?\')">🌤️ Vizag weather</button><button class="suggestion" onclick="suggest(\'Give me some action movies\')">🎬 Action movies</button><button class="suggestion" onclick="suggest(\'Give me some thriller movies\')">🍿 Thriller movies</button></div>';
  chatBody.appendChild(div);
}
question.addEventListener('keydown',function(event){ if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();askQuestion();} });
function scrollBottom(){ setTimeout(()=>chatBody.scrollTo({top:chatBody.scrollHeight,behavior:'smooth'}),50); }
</script>
</body>
</html>
'''


# ============================================================
# 10. ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML


@app.post("/ask")
async def ask(payload: dict):
    question = (payload.get("question", "") or "").strip()

    if not question:
        return {"answer": "Please enter a question.", "tools_used": []}

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })

    return {
        "answer": extract_text_response(result),
        "tools_used": get_tools_used(result),
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "India Weather & Movies Assistant",
        "version": "4.0.0",
        "tools": ["get_weather", "search_movies"],
    }


add_routes(
    app,
    formatted_agent_chain,
    path="/agent",
    playground_type="default",
)


# ============================================================
# 11. SERVER
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
