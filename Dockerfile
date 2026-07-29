# Dockerfile — capstone handbook Section 16.3 (containerization)
#
# Base: Python 3.11, matching requirements.txt's floor (numpy/pandas/etc.
# pin against 3.11+ wheel availability).
FROM python:3.11-slim

# System libraries WeasyPrint needs for its Pango/cairo/GObject bindings
# (src/agents/report_generator.py's macOS DYLD_LIBRARY_PATH fix is the
# Homebrew/dylib equivalent of this on macOS; here we install the actual
# Debian shared libraries WeasyPrint's cffi bindings load at runtime).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpango-1.0-0 \
    libpangoft2-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached across
# code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code. Every path in this codebase is resolved relative
# to __file__ at runtime (see app/home.py, app/dashboard_helpers.py,
# app/pages/*.py) rather than hardcoded as an absolute filesystem path, so
# nothing here needs adjusting for the container's /app root — this
# already satisfies the handbook's "no absolute system file paths, all
# file tracking via relative workspace directories" requirement.
COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app/home.py", "--server.address=0.0.0.0"]
