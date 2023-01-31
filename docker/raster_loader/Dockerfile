FROM osgeo/gdal:3.2.0

ENV HOMEAPP=/code
ENV PATH=$PATH:$HOMEAPP/.local/bin
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR $HOMEAPP/

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
    bash \
    build-essential \
    gcc \
    git \
    libpq-dev \
    python3-dev \
    postgresql-client \
    wget \
    python3-pip \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Using a non-privileged user to own our code
RUN useradd -d $HOMEAPP -N non-privileged

# Update non-privileged user folder permission
RUN chown -R non-privileged $HOMEAPP

# Copy the requirements file into the container
COPY requirements.txt .
COPY requirements-dev.txt .
COPY . .

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy the rest of the files into the container
USER non-privileged
