# Import the TavilyClient class, which lets us call the Tavily search API
from tavily import TavilyClient

# Import the os module so we can read environment variables (like API keys)
import os

# Import load_dotenv so we can load variables from a .env file into the environment
from dotenv import load_dotenv

# Actually load the variables from the .env file into the environment
# (this must run before we try to read any env vars below)
load_dotenv()

# Create a TavilyClient instance, authenticated using the API key
# pulled from the environment variable "TAVILY_API_KEY"
client = TavilyClient(os.getenv("TAVILY_API_KEY"))

# Define a function that takes a search query string and returns
# a formatted string of search results
def tavily_search(query):
    # Call Tavily's search API with the given query,
    # asking for a maximum of 5 results back
    response = client.search(
        query=query,
        max_results=5
    )

    # Create an empty list to hold our formatted result strings
    results = []

    # Loop through each result in response["results"],
    # numbering them starting from 1 (that's what the "1" in enumerate does)
    for i, r in enumerate(response["results"], 1):
        # Get the "title" field from this result; if it doesn't exist, default to "Unknown"
        title   = r.get("title", "Unknown")

        # Get the "url" field from this result; if it doesn't exist, default to an empty string
        url     = r.get("url", "")

        # Get the "content" field (the snippet/summary text);
        # default to empty string if missing, then strip leading/trailing whitespace
        snippet = r.get("content", "").strip()

        # If the snippet is longer than 300 characters, trim it down
        # so results don't become a huge wall of text
        if len(snippet) > 300:
            # Cut the snippet to the first 300 characters,
            # then rsplit(" ", 1)[0] removes any trailing partial word
            # (splits on the last space, keeps everything before it),
            # and "..." is added to show the text was truncated
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        # Build a formatted string for this result:
        # numbered title (bold), then URL on its own line, then the snippet
        # and add it to our results list
        results.append(f"{i}. **{title}**\n   {url}\n   {snippet}")

    # Join all the formatted result strings together,
    # separated by a blank line (\n\n) between each one,
    # and return the final combined string
    return "\n\n".join(results)