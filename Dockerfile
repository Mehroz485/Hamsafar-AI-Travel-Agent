# ============================================================
# BASE IMAGE
# ============================================================

# Start from an official Python 3.11 image, using the "slim" variant.
# "slim" means it's a stripped-down Debian Linux image with only the
# bare minimum needed to run Python — much smaller than the full
# "python:3.11" image, which includes lots of extra tools/libraries
# you probably don't need. This keeps your final Docker image size down.
FROM python:3.11-slim


# ============================================================
# WORKING DIRECTORY
# ============================================================

# Sets /app as the "current directory" inside the container for all
# following instructions (COPY, RUN, CMD, etc.). If /app doesn't exist
# yet inside the container, Docker creates it automatically.
# This is just good practice — keeps your app's files organized in
# one predictable location instead of scattered in the container's root.
WORKDIR /app


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

# Tells Python NOT to create .pyc bytecode cache files (the __pycache__
# folders you normally see). Inside a container, these compiled files
# don't provide any real benefit (the container is rebuilt from scratch
# each time anyway), so disabling them saves disk space and avoids
# clutter in your image.
ENV PYTHONDONTWRITEBYTECODE=1

# Forces Python's print statements and logs to be sent straight to the
# terminal/output immediately, instead of being "buffered" (held back
# and released in batches). This is important in Docker because without
# it, your logs might not show up in real-time when you run
# `docker logs` — they could appear delayed or all at once at the end.
ENV PYTHONUNBUFFERED=1


# ============================================================
# SYSTEM-LEVEL DEPENDENCIES
# ============================================================

# apt-get update: refreshes the list of available packages and their
# versions from Debian's package repositories (needed before installing
# anything new, otherwise apt might try to fetch outdated/broken package info)
#
# apt-get install -y: installs the following packages, with -y meaning
# "automatically answer yes to any confirmation prompts" (needed since
# there's no human present to type "y" during a Docker build)
#
#   build-essential: a bundle of compilers and build tools (like gcc, make).
#     Some Python packages need to compile C code during installation if
#     a pre-built binary ("wheel") isn't available for your exact system.
#     Without this, those installs would fail with cryptic compiler errors.
#
#   libpq-dev: development headers/libraries for PostgreSQL. Your
#     requirements.txt includes psycopg[binary] and psycopg_pool, which
#     talk to a Postgres database. Even though "[binary]" usually means
#     a pre-compiled version is used (so this often isn't strictly
#     needed), including libpq-dev acts as a safety net in case pip
#     can't find a matching binary wheel for your exact CPU
#     architecture/OS combo and needs to build from source instead.
#
#   curl: a command-line tool for making HTTP requests. Not required by
#     your Python code directly, but commonly used for things like
#     Docker healthchecks (a way to verify the container is running
#     correctly) or manual debugging inside the container later.
#
# && rm -rf /var/lib/apt/lists/*: after installing everything, this
#   deletes the package list cache that apt-get update downloaded.
#   This cache is no longer needed once installation is done, and
#   removing it significantly reduces the final image size. It's
#   standard practice to always clean this up in the SAME RUN command
#   as the install (chaining with &&) — if you did it in a separate
#   RUN step, Docker's layer caching would still keep the bloat from
#   the earlier layer, wasting the space savings.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# COPY REQUIREMENTS FILE (separately from the rest of the code)
# ============================================================

# Copies ONLY requirements.txt from your project folder into the
# container's current directory (/app). Notice this happens BEFORE
# copying the rest of your application code (further down).
#
# WHY separate this step? Docker builds images in layers, and caches
# each layer. If requirements.txt hasn't changed since your last build,
# Docker will reuse the cached "pip install" layer instead of
# re-running it — which saves a LOT of time, since installing packages
# is usually the slowest part of a build. If you copied all your code
# first (including requirements.txt mixed in), ANY code change would
# invalidate the cache and force a full reinstall of every dependency,
# even if requirements.txt itself didn't change.
COPY requirements.txt .


# ============================================================
# INSTALL PYTHON DEPENDENCIES
# ============================================================

# Upgrades pip itself (the Python package installer) to the latest
# version before installing your actual dependencies. This helps avoid
# bugs or incompatibilities that can happen with older pip versions,
# especially with newer packages that expect modern pip features.
#
# --no-cache-dir: tells pip not to keep a local cache of downloaded
# packages after installing them. Normally pip caches packages so
# future installs are faster, but inside a Docker image that cache is
# just wasted space (you won't be reinstalling packages again in this
# same container) — so this flag keeps the image smaller.
RUN pip install --no-cache-dir --upgrade pip

# Installs every package listed in requirements.txt, using the exact
# versions you pinned (e.g. langgraph==1.2.11, langchain==1.4.0, etc.)
# Again using --no-cache-dir for the same size-saving reason as above.
# This is typically the slowest step in the whole build, since it has
# to download and set up potentially dozens of packages and their
# sub-dependencies (which is exactly why we cache this layer separately
# from your actual application code, as explained above).
RUN pip install --no-cache-dir -r requirements.txt


# ============================================================
# COPY APPLICATION CODE
# ============================================================

# NOW copy everything else from your project folder (all your .py
# files, subfolders, etc.) into the container's /app directory.
# This happens AFTER the pip install step specifically so that
# changing your application code (which happens often) doesn't
# force Docker to redo the slow dependency installation (which
# only needs to happen when requirements.txt itself changes).
COPY . .


# ============================================================
# EXPOSE PORT
# ============================================================

# This tells Docker (and anyone reading this Dockerfile) that the
# application inside this container listens on port 8000.
# IMPORTANT: this line by itself does NOT actually publish/open the
# port to the outside world — it's essentially documentation/metadata.
# You still need to map the port when you actually RUN the container,
# e.g. `docker run -p 8000:8000 your-image-name`, for it to be
# reachable from outside the container.
EXPOSE 8000


# ============================================================
# STARTUP COMMAND
# ============================================================

# This is the command that runs automatically when a container is
# started from this image. It uses the array/exec form (square
# brackets), which is the recommended way to write CMD because it
# runs the command directly rather than through a shell — this makes
# signal handling (like Ctrl+C or Docker stop commands) work correctly.
#
# uvicorn: an ASGI web server used to run Python web applications
#   (commonly used with FastAPI, which your project likely uses given
#   the "app:app" pattern below)
#
# "app:app": this means "look inside a Python file/module called
#   app.py, and find a variable inside it named 'app'" — that variable
#   should be your FastAPI() application instance. If your actual
#   entry-point file is named something else (like main.py with a
#   variable called "api"), this would need to change to "main:api"
#   or whatever matches your project structure.
#
# --host 0.0.0.0: tells uvicorn to accept connections from ANY network
#   interface, not just "localhost". This is REQUIRED inside Docker —
#   if you used the default "127.0.0.1" instead, the server would only
#   accept connections from INSIDE the container itself, making it
#   completely unreachable from outside (even with the port mapped).
#
# --port 8000: tells uvicorn to listen on port 8000, matching the
#   EXPOSE line above so everything lines up consistently.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]