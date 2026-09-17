# ============================================================
# IMPORTS & SETUP
# ============================================================

import os                          # lets us read environment variables (like API keys)
import re                          # regex — used for pattern matching and cleaning text
import certifi                     # gives us a trusted list of SSL certificates
import airportsdata                # a ready-made database of every airport in the world
import pycountry                   # a ready-made database of countries and their codes
import requests                    # used to make HTTP calls to the flight API
from dotenv import load_dotenv     # loads secret values (like API keys) from a .env file

load_dotenv()  
# this actually reads the .env file and puts its values into the environment
# so we can grab them with os.getenv() below

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
# these two lines fix a common Windows problem where secure (HTTPS) connections
# fail because Windows doesn't have a proper certificate list —
# we're telling Python "use certifi's certificate list instead"

API_KEY = os.getenv("AVIATIONSTACK_API_KEY")  
# grabs our flight API's secret key from the .env file
DEFAULT_ORIGIN_IATA = os.getenv("DEFAULT_ORIGIN_IATA", "KHI")
# if the user only mentions a destination (not where they're flying FROM),
# we assume they're flying from this airport (Dhaka, DAC, unless changed in .env)

BASE_URL = "https://api.aviationstack.com/v1/flights"  
# the web address we send flight search requests to

AIRPORTS = airportsdata.load("IATA")  
# loads EVERY airport in the world into a big dictionary in memory,
# so we can instantly look up airport info without needing the internet


# ============================================================
# CHEAT SHEET 1: COUNTRY_ALIASES
# Maps common nicknames people actually type -> official 2-letter country code
# ============================================================
COUNTRY_ALIASES = {
    "usa": "US", "u.s.a": "US", "u.s.": "US", "america": "US", "united states": "US",
    "uk": "GB", "u.k.": "GB", "britain": "GB", "england": "GB",
    "uae": "AE", "dubai": "AE",  
    # ^ dubai is a CITY not a country, but people often say "Dubai" meaning UAE, so we allow it
    "south korea": "KR", "korea": "KR",
    "russia": "RU", "vietnam": "VN", "bangladesh": "BD", "india": "IN",
    "japan": "JP", "china": "CN", "singapore": "SG", "malaysia": "MY",
    "thailand": "TH", "indonesia": "ID", "nepal": "NP", "qatar": "QA",
    "saudi arabia": "SA", "turkey": "TR", "canada": "CA", "australia": "AU",
    "germany": "DE", "france": "FR", "italy": "IT", "spain": "ES",
}
# this exists so we don't have to do a slow/smart search every time —
# if someone types a common nickname, we can look it up instantly here


# ============================================================
# CHEAT SHEET 2: COUNTRY_MAIN_AIRPORT
# If someone only mentions a COUNTRY (not a city), which airport do we assume?
# ============================================================
COUNTRY_MAIN_AIRPORT = {
    "BD": "DAC", "IN": "DEL", "JP": "NRT", "US": "JFK", "GB": "LHR",
    "AE": "DXB", "SG": "SIN", "MY": "KUL", "TH": "BKK", "ID": "CGK",
    "CN": "PEK", "KR": "ICN", "NP": "KTM", "QA": "DOH", "SA": "JED",
    "TR": "IST", "CA": "YYZ", "AU": "SYD", "DE": "FRA", "FR": "CDG",
    "IT": "FCO", "ES": "MAD",
    "PK": "KHI",  
}
# this is our "pre-decided answer" so we don't have to calculate it every time


# ============================================================
# CHEAT SHEET 3: CITY_MAIN_AIRPORT
# If someone mentions a CITY, which airport do we assume?
# (important for cities that have MULTIPLE airports, like Tokyo)
# ============================================================
CITY_MAIN_AIRPORT = {
    "dhaka": "DAC", "delhi": "DEL", "new delhi": "DEL", "mumbai": "BOM",
    "kolkata": "CCU", "chennai": "MAA", "bangalore": "BLR", "bengaluru": "BLR",
    "tokyo": "NRT", "osaka": "KIX", "kyoto": "KIX", "new york": "JFK",
    "london": "LHR", "dubai": "DXB", "singapore": "SIN", "kuala lumpur": "KUL",
    "bangkok": "BKK", "doha": "DOH", "istanbul": "IST", "toronto": "YYZ",
    "sydney": "SYD", "paris": "CDG", "rome": "FCO", "madrid": "MAD",
    "frankfurt": "FRA",
}


# ============================================================
# FUNCTION: clean_text
# JOB: take messy user text and tidy it up so it's easier to work with
# ============================================================
def clean_text(text: str) -> str:
    # make everything lowercase, remove extra spaces at start/end
    # so "Japan" and "japan " are treated the same
    text = text.lower().strip()

    # remove anything that ISN'T a letter, number, or space
    # (this kills punctuation like ! , . $ & etc.)
    # we replace with a SPACE (not nothing) so words don't accidentally join together
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # if removing punctuation left behind multiple spaces in a row,
    # squash them down into just one space
    text = re.sub(r"\s+", " ", text)

    # these are common "filler" words in travel questions that don't
    # actually tell us WHERE someone wants to go — so we throw them away
    stop_words = [
        "flight", "flights", "ticket", "tickets", "trip", "travel",
        "plan", "complete", "days", "day", "including", "hotel",
        "hotels", "sightseeing", "under", "budget", "info", "information"
    ]

    # split the sentence into individual words, and only KEEP the ones
    # that are NOT in our filler-word list above
    words = [w for w in text.split() if w not in stop_words]

    # glue the remaining words back together into one clean string
    return " ".join(words).strip()
    # EXAMPLE: "Plan a complete 7 day trip to Japan!!" becomes just "a 7 to japan"


# ============================================================
# FUNCTION: country_name_to_code
# JOB: take ANY text and try to figure out what country it's talking about,
# then return that country's official 2-letter code (like "JP" for Japan)
# ============================================================
def country_name_to_code(text: str):
    text = clean_text(text)  # tidy up the input first using the function above

    # METHOD 1 (fastest): is this EXACTLY one of our known nicknames?
    if text in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[text]

    # METHOD 2: ask the pycountry library if it recognizes this as
    # an official country name or code
    try:
        country = pycountry.countries.lookup(text)
        return country.alpha_2  # alpha_2 = the 2-letter code, e.g. "JP"
    except LookupError:
        # if pycountry didn't recognize it, don't crash —
        # just move on and try the next method
        pass

    # METHOD 3: maybe the country name is buried INSIDE a longer sentence.
    # loop through literally every country pycountry knows about,
    # and check if that country's name appears anywhere in our text
    for country in pycountry.countries:
        country_name = country.name.lower()
        if country_name in text:
            return country.alpha_2

    # METHOD 4: same idea as method 3, but checking our own alias list
    # instead of official pycountry names (catches short forms like "uae")
    for alias, code in COUNTRY_ALIASES.items():
        if alias in text:
            return code

    # if NONE of the 4 methods worked, we genuinely don't know — give up
    return None


# ============================================================
# FUNCTION: airport_country_matches
# JOB: check if a specific airport belongs to a specific country
# PROBLEM IT SOLVES: airport data is messy — some entries store the
# country as a CODE ("PK"), others as a full NAME ("Pakistan")
# ============================================================
def airport_country_matches(airport: dict, country_code: str) -> bool:
    # grab the airport's stored country value, make it uppercase and trimmed
    airport_country = str(airport.get("country", "")).upper().strip()

    # CHECK 1: does it match the code directly? e.g. "PK" == "PK"
    if airport_country == country_code:
        return True

    # CHECK 2: maybe it's stored as a full country name instead of a code
    try:
        # look up what country this code actually represents
        country = pycountry.countries.get(alpha_2=country_code)
        # compare the airport's country field against that country's real name
        if country and airport_country.lower() == country.name.lower():
            return True
    except Exception:
        # if something goes wrong in the lookup, just ignore it and move on
        pass

    # neither check matched — this airport does NOT belong to that country
    return False


# ============================================================
# FUNCTION: get_best_airport_for_country
# JOB: figure out the single "main" airport for a given country
# ============================================================
def get_best_airport_for_country(country_code: str):
    # FAST PATH: check if we already have this country's answer
    # pre-decided in our cheat sheet
    preferred = COUNTRY_MAIN_AIRPORT.get(country_code)

    # only use it if it's actually a real airport in our database too
    if preferred and preferred in AIRPORTS:
        return preferred

    # SLOW PATH: we don't have a pre-decided answer, so we need to
    # search and score every airport ourselves
    candidates = []  # will store (score, airport_code) pairs

    # go through literally every airport in the world
    for iata, airport in AIRPORTS.items():
        if not iata:
            continue  # skip broken entries with no airport code

        # only bother scoring this airport if it's actually in our target country
        if airport_country_matches(airport, country_code):
            name = str(airport.get("name", "")).lower()
            city = str(airport.get("city", "")).lower()

            score = 0  # start at zero, add points based on clues

            # airports that say "International" in their name are
            # usually the big, important ones — give lots of points
            if "international" in name:
                score += 50

            # "Intl" is just short for "International" — same idea, smaller boost
            if "intl" in name:
                score += 40

            # airports serving a capital city are often more important
            if "capital" in name:
                score += 20

            # having a city name listed at all is a small sign of
            # more complete/reliable data — tiny bonus
            if city:
                score += 5

            # save this airport's score for comparison later
            candidates.append((score, iata))

    # if we found literally zero airports for this country, give up
    if not candidates:
        return None

    # sort so the HIGHEST score comes first
    candidates.sort(reverse=True)

    # return just the airport code of the winner (highest score)
    return candidates[0][1]


# ============================================================
# FUNCTION: resolve_location_to_iata
# JOB: the "master converter" — takes literally ANY location text
# (country, city, or already-an-airport-code) and turns it into
# one final, usable 3-letter airport code
# ============================================================
def resolve_location_to_iata(location: str):
    """
    Converts country/city/airport/IATA into IATA code.
    Examples: Bangladesh -> DAC, Japan -> NRT, Dhaka -> DAC, Tokyo -> NRT, DAC -> DAC
    """

    if not location:
        return None  # nothing was given, nothing to do

    raw_location = location.strip()  # remove extra spaces around the input

    # STEP 1: check if this is ALREADY a 3-letter airport code, like "DAC"
    if re.fullmatch(r"[A-Za-z]{3}", raw_location):
        code = raw_location.upper()
        # double check it's a REAL airport code in our database, not random letters
        if code in AIRPORTS:
            return code

    # clean up the text for the remaining checks
    location_clean = clean_text(raw_location)

    if not location_clean:
        return None  # cleaning removed everything — nothing usable left

    # STEP 2: is this a city we already know the answer for? (fast cheat sheet)
    if location_clean in CITY_MAIN_AIRPORT:
        return CITY_MAIN_AIRPORT[location_clean]

    # STEP 3: is this actually a country? if so, find that country's main airport
    country_code = country_name_to_code(location_clean)
    if country_code:
        airport = get_best_airport_for_country(country_code)
        if airport:
            return airport

    # STEP 4 (last resort): search through EVERY airport in the world,
    # trying to find one whose city or name looks like what was typed
    city_matches = []

    for iata, airport in AIRPORTS.items():
        city = str(airport.get("city", "")).lower().strip()
        name = str(airport.get("name", "")).lower().strip()

        score = 0

        # exact city name match is the strongest signal — biggest points
        if city == location_clean:
            score += 100
        # if our text appears somewhere INSIDE the city name, weaker signal
        elif location_clean in city:
            score += 70

        # if our text appears inside the airport's actual NAME, some points
        if location_clean in name:
            score += 50

        # small preference for airports that are labeled "International"
        if "international" in name:
            score += 10

        # only keep airports that scored something (i.e., actually matched somehow)
        if score > 0:
            city_matches.append((score, iata))

    # if we found any matches, return the highest-scoring one
    if city_matches:
        city_matches.sort(reverse=True)
        return city_matches[0][1]

    # absolutely nothing matched — give up
    return None


# ============================================================
# FUNCTION: find_location_mentions
# JOB: scan a FULL SENTENCE and pull out every country/city name
# it can spot mentioned inside it — used as a backup plan
# ============================================================
def find_location_mentions(query: str):
    """Finds country or city names inside a natural language query."""

    q = query.lower()  # work in lowercase for consistent matching
    mentions = []       # will collect every location name we find

    # check every alias in our cheat sheet — \b...\b means "whole word only"
    # (so "in" wouldn't accidentally match inside a word like "sightseeing")
    for alias in COUNTRY_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", q):
            mentions.append(alias)

    # check every OFFICIAL country name pycountry knows about
    # (only names 4+ letters long, to avoid short/confusing false matches)
    for country in pycountry.countries:
        name = country.name.lower()
        if len(name) >= 4 and re.search(rf"\b{re.escape(name)}\b", q):
            mentions.append(name)

    # check every city name from our city cheat sheet too
    for city in CITY_MAIN_AIRPORT:
        if re.search(rf"\b{re.escape(city)}\b", q):
            mentions.append(city)

    # remove any duplicate mentions, but keep the ORDER they appeared in
    unique_mentions = []
    for item in mentions:
        if item not in unique_mentions:
            unique_mentions.append(item)

    return unique_mentions


# ============================================================
# FUNCTION: parse_route
# JOB: THE BRAIN of the whole system. Takes a full sentence and
# figures out: where is the person flying FROM, and where TO?
# Tries several different "pattern guesses" in order, most specific first
# ============================================================
def parse_route(query: str):
    """
    Returns: dep_iata, arr_iata
    None, None  -> show global/all flights
    DAC, NRT    -> specific route
    DAC, None   -> all flights FROM this airport
    None, NRT   -> all flights TO this airport
    """

    q = query.strip()
    q_lower = q.lower()

    # ATTEMPT 1: does the sentence contain "global"/"all flights" type words?
    # if so, we don't need a specific route — show everything
    global_keywords = [
        "all country", "all countries", "global flight", "global flights",
        "all flight", "all flights", "worldwide flight", "worldwide flights",
    ]
    if any(keyword in q_lower for keyword in global_keywords):
        return None, None

    # ATTEMPT 2: does the sentence contain two RAW airport codes already?
    # like "DAC to NRT" — this is the clearest possible signal
    codes = re.findall(r"\b[A-Z]{3}\b", q)  
    # note: uses "q" not "q_lower" because airport codes are conventionally UPPERCASE,
    # so searching the original-case text catches real codes and avoids
    # matching random lowercase 3-letter words
    if len(codes) >= 2:
        dep = codes[0].upper()
        arr = codes[1].upper()
        return dep, arr

    # ATTEMPT 3: look for the pattern "from X to Y"
    # example: "flights from Dhaka to Tokyo"
    match = re.search(
        r"\bfrom\s+(.+?)\s+\bto\s+(.+?)(?:\s+(?:on|for|under|including|with|in|at)\b|[.!?]|$)",
        q_lower,
    )
    if match:
        origin_text = match.group(1)   # the text between "from" and "to"
        dest_text = match.group(2)     # the text after "to" (until a stop word or end)
        dep_iata = resolve_location_to_iata(origin_text)
        arr_iata = resolve_location_to_iata(dest_text)
        return dep_iata, arr_iata

    # ATTEMPT 4: look for the REVERSED pattern "to Y from X"
    # example: "flights to Tokyo from Dhaka"
    match = re.search(
        r"\bto\s+(.+?)\s+\bfrom\s+(.+?)(?:\s+(?:on|for|under|including|with|in|at)\b|[.!?]|$)",
        q_lower,
    )
    if match:
        dest_text = match.group(1)
        origin_text = match.group(2)
        dep_iata = resolve_location_to_iata(origin_text)
        arr_iata = resolve_location_to_iata(dest_text)
        return dep_iata, arr_iata

    # ATTEMPT 5: only "from X" is present, no "to" —
    # treat it as "show me all flights leaving from this place"
    match = re.search(r"\bfrom\s+(.+?)(?:[.!?]|$)", q_lower)
    if match:
        origin_text = match.group(1)
        dep_iata = resolve_location_to_iata(origin_text)
        return dep_iata, None

    # ATTEMPT 6: only "to X" is present, no "from" —
    # treat it as "show me all flights arriving at this place"
    match = re.search(r"\bto\s+(.+?)(?:[.!?]|$)", q_lower)
    if match:
        dest_text = match.group(1)
        arr_iata = resolve_location_to_iata(dest_text)
        return None, arr_iata

    # ATTEMPT 7 (last resort): no clear "from"/"to" wording at all —
    # just scan the whole sentence for any place names we recognize
    mentions = find_location_mentions(q)

    # if we found 2+ places mentioned, guess the FIRST one is the origin
    # and the SECOND one is the destination
    if len(mentions) >= 2:
        dep_iata = resolve_location_to_iata(mentions[0])
        arr_iata = resolve_location_to_iata(mentions[1])
        return dep_iata, arr_iata

    # if only ONE place was mentioned, assume it's the destination,
    # and use our default origin airport (set at the top of the file)
    if len(mentions) == 1:
        arr_iata = resolve_location_to_iata(mentions[0])
        return DEFAULT_ORIGIN_IATA, arr_iata

    # absolutely nothing recognizable found — give up, return nothing
    return None, None


# ============================================================
# FUNCTION: format_flight
# JOB: take ONE raw flight record from the API (which is a messy
# nested dictionary) and turn it into a clean, readable text block
# ============================================================
def format_flight(flight: dict):
    # dig into nested dictionaries safely — if any piece is missing,
    # fall back to a friendly "Unknown ___" placeholder instead of crashing
    airline = flight.get("airline", {}).get("name") or "Unknown airline"
    flight_number = flight.get("flight", {}).get("iata") or "Unknown flight number"
    status = flight.get("flight_status") or "Unknown"

    # grab the departure and arrival info blocks (each is its own dictionary)
    dep = flight.get("departure", {}) or {}
    arr = flight.get("arrival", {}) or {}

    # pull out each departure detail, with safe fallbacks
    dep_airport = dep.get("airport") or "Unknown departure airport"
    dep_iata = dep.get("iata") or "Unknown"
    dep_terminal = dep.get("terminal") or "N/A"
    dep_gate = dep.get("gate") or "N/A"
    dep_scheduled = dep.get("scheduled") or "Unknown"
    dep_delay = dep.get("delay")
    # special case: delay could legitimately be 0 (no delay), so we
    # check "is not None" instead of just checking if it's truthy
    dep_delay_text = f"{dep_delay} minutes" if dep_delay is not None else "N/A"

    # same idea, but for arrival details
    arr_airport = arr.get("airport") or "Unknown arrival airport"
    arr_iata = arr.get("iata") or "Unknown"
    arr_terminal = arr.get("terminal") or "N/A"
    arr_gate = arr.get("gate") or "N/A"
    arr_scheduled = arr.get("scheduled") or "Unknown"
    arr_delay = arr.get("delay")
    arr_delay_text = f"{arr_delay} minutes" if arr_delay is not None else "N/A"

    # build one big readable text block with all this info laid out nicely
    return f"""
Airline: {airline}
Flight: {flight_number}
Status: {status}

Departure:
- Airport: {dep_airport}
- IATA: {dep_iata}
- Terminal: {dep_terminal}
- Gate: {dep_gate}
- Scheduled: {dep_scheduled}
- Delay: {dep_delay_text}

Arrival:
- Airport: {arr_airport}
- IATA: {arr_iata}
- Terminal: {arr_terminal}
- Gate: {arr_gate}
- Scheduled: {arr_scheduled}
- Delay: {arr_delay_text}
""".strip()


# ============================================================
# FUNCTION: search_flights
# JOB: THE MAIN ENTRY POINT. This is the function you actually call.
# It ties everything together: understand the query, call the API,
# and hand back a nice readable answer
# ============================================================
def search_flights(query: str, limit: int = 10):
    # if we don't even have an API key set up, don't bother calling anything —
    # just tell the user how to fix it
    if not API_KEY:
        return (
            "Flight API error: AVIATIONSTACK_API_KEY is missing.\n"
            "Please add this in your .env file:\n"
            "AVIATIONSTACK_API_KEY=your_api_key_here"
        )

    # figure out the departure and arrival airports from the user's sentence
    dep_iata, arr_iata = parse_route(query)

    # build the parameters we'll send to the flight API
    params = {
        "access_key": API_KEY,
        "limit": min(limit, 100),  # never ask for more than 100 (API's max)
    }

    # only add these filters if we actually figured them out
    if dep_iata:
        params["dep_iata"] = dep_iata
    if arr_iata:
        params["arr_iata"] = arr_iata

    # actually make the web request to the flight API
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        data = response.json()
    except requests.exceptions.RequestException as e:
        # something went wrong with the internet connection/request itself
        return f"Flight API request failed: {e}"
    except ValueError:
        # the API sent back something that isn't valid JSON — unexpected
        return "Flight API returned invalid JSON."

    # if the API itself reported an error (like bad API key, rate limit, etc.)
    if "error" in data:
        error = data["error"]
        return (
            "Flight API error:\n"
            f"Code: {error.get('code', 'Unknown')}\n"
            f"Message: {error.get('message', 'Unknown error')}"
        )

    # grab the actual list of flights from the response
    flight_data = data.get("data", [])

    # if there are literally no flights matching our search
    if not flight_data:
        route_text = ""
        if dep_iata and arr_iata:
            route_text = f" for route {dep_iata} to {arr_iata}"
        elif dep_iata:
            route_text = f" from {dep_iata}"
        elif arr_iata:
            route_text = f" to {arr_iata}"

        return (
            f"No live flight data found{route_text}.\n\n"
            "Note: AviationStack provides live/status flight data, not ticket prices. "
            "For actual fare prices, use a flight-pricing API such as Amadeus."
        )

    # build a friendly header line describing what was searched
    route_info = "Global live flights"
    if dep_iata and arr_iata:
        route_info = f"Live flights from {dep_iata} to {arr_iata}"
    elif dep_iata:
        route_info = f"Live flights from {dep_iata}"
    elif arr_iata:
        route_info = f"Live flights to {arr_iata}"

    # format each flight nicely, but only up to our "limit" number of them
    formatted_flights = [format_flight(flight) for flight in flight_data[:limit]]

    # combine the header + all formatted flights, separated by dashed lines
    return f"{route_info}\n\n" + "\n\n---\n\n".join(formatted_flights)


# ============================================================
# SCRIPT ENTRY POINT
# This block ONLY runs if you run this file directly with "python file.py"
# It will NOT run if this file is imported into another file
# ============================================================
if __name__ == "__main__":
    # TEST 1: should figure out dep = Bangladesh's main airport (DAC),
    # arr = Japan's main airport (NRT)

    print("\n" + "=" * 80 + "\n")  # just a visual divider line

    # TEST 2: should trigger the "global flights" keyword match,
    # so it searches with NO specific route filter
    print(search_flights("all country flight info"))



    # ============================================================
# COMPLETE FLOW OF THE FLIGHT TOOL — STEP BY STEP
# ============================================================

# STEP 0: SETUP (happens once, when the file is first loaded)
# - Load environment variables from .env (API keys, default origin)
# - Fix SSL certificate issues on Windows using certifi
# - Load the entire world airport database into memory (AIRPORTS)
# - Define cheat-sheet dictionaries: COUNTRY_ALIASES, COUNTRY_MAIN_AIRPORT,
#   CITY_MAIN_AIRPORT — these let us skip slow searches for common cases


# STEP 1: USER CALLS search_flights(query)
# This is the single entry point. Example query:
# "Plan a 7 day Japan trip from Bangladesh"


# STEP 2: search_flights() checks if API_KEY exists
# - If missing, immediately return an error message telling the user
#   to add AVIATIONSTACK_API_KEY to their .env file
# - If present, continue


# STEP 3: search_flights() calls parse_route(query)
# This is the "brain" — its job is to figure out two things:
#   - dep_iata  (departure airport code, or None)
#   - arr_iata  (arrival airport code, or None)
#
# parse_route() tries several strategies IN ORDER, stopping at the
# first one that produces a result:
#
#   3a. Check for "global/all flights" keywords
#       -> if found, return (None, None) immediately, no route needed
#
#   3b. Check if the query already contains two raw IATA codes
#       (e.g. "DAC to NRT") -> use them directly, done
#
#   3c. Look for the pattern "from X to Y"
#       -> extract text X and text Y as raw strings
#       -> pass EACH one individually into resolve_location_to_iata()
#          to convert them into real airport codes
#
#   3d. Look for the reversed pattern "to Y from X"
#       -> same idea, just swapped order
#
#   3e. Look for just "from X" (no "to") -> departure-only search
#
#   3f. Look for just "to X" (no "from") -> arrival-only search
#
#   3g. FALLBACK: if none of the above patterns matched at all,
#       call find_location_mentions(query) to scan the ENTIRE sentence
#       for any recognizable country/city names, in any order
#       -> if 2+ places found, assume first = origin, second = destination
#       -> if only 1 place found, assume it's the destination and use
#          DEFAULT_ORIGIN_IATA as the origin
#       -> if 0 places found, give up and return (None, None)


# STEP 4: EVERY raw location string found in Step 3 gets passed through
# resolve_location_to_iata(location) — the "universal converter"
# This function tries, in order:
#
#   4a. Is the text already a valid 3-letter airport code? (e.g. "DAC")
#       -> if yes and it exists in AIRPORTS, return it directly
#
#   4b. Clean the text using clean_text() (lowercase, remove punctuation,
#       remove filler words like "trip", "days", "budget", etc.)
#
#   4c. Is the cleaned text a known CITY in CITY_MAIN_AIRPORT?
#       -> if yes, return that city's pre-decided main airport
#
#   4d. Is the cleaned text a COUNTRY? Call country_name_to_code() to find out:
#       - check COUNTRY_ALIASES for an exact match
#       - try pycountry's official lookup
#       - search if any real country name appears INSIDE the text
#         (this is what allows "ridion pakistan" to still find Pakistan)
#       - search if any of our custom aliases appear inside the text
#       -> if a country code IS found, call get_best_airport_for_country()
#          to get that country's main airport:
#            - check COUNTRY_MAIN_AIRPORT cheat sheet first (fast)
#            - if not found there, scan ALL airports in that country,
#              score each one based on keywords in its name
#              ("international" = +50, "intl" = +40, "capital" = +20,
#               has a city listed = +5), and return the highest scorer
#              (airport_country_matches() is used here to filter airports
#               by country, since airport data sometimes stores country
#               as a code and sometimes as a full name)
#
#   4e. LAST RESORT: if nothing above matched, scan the ENTIRE airport
#       database directly, scoring airports based on whether the cleaned
#       text appears in their city name or airport name
#       -> return the highest-scoring match, or None if nothing matched


# STEP 5: back in search_flights(), we now have dep_iata and/or arr_iata
# (either could be None if unresolved)
# - Build the request parameters for the AviationStack API,
#   only including dep_iata/arr_iata filters if they were actually found


# STEP 6: search_flights() sends the HTTP GET request to AviationStack
# - If the request fails (network error) -> return an error message
# - If the response isn't valid JSON -> return an error message
# - If the API itself returns an "error" field -> return that error info


# STEP 7: check if any flights were actually returned
# - If the flight list is empty -> return a friendly "no flights found"
#   message, including a note that this API gives live status data,
#   not ticket prices


# STEP 8: if flights WERE found, build a header describing the route
# searched (e.g. "Live flights from DAC to NRT"), then format each
# individual flight using format_flight()
# - format_flight() safely extracts airline, flight number, status,
#   and all departure/arrival details (airport, terminal, gate,
#   scheduled time, delay), using fallback text like "Unknown" or "N/A"
#   for any missing fields


# STEP 9: join the header + all formatted flights together with
# separator lines, and return this final combined string as the answer


# STEP 10 (only when running the file directly, not importing it):
# - Run two test example queries through search_flights() and print
#   the results, just to sanity-check the whole pipeline works end-to-end