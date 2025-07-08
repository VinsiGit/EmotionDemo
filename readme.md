# Emotion App

This project is an emotion detection demo that uses a Python backend with FastAPI to serve an emotion recognition model and a simple HTML/CSS/JavaScript frontend to interact with it.

## Table of Contents

- [Emotion App](#emotion-app)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
    - [Backend (FastAPI Server)](#backend-fastapi-server)
    - [Frontend](#frontend)
  - [Project Structure](#project-structure)
  <!-- - [Research](#research) -->

## Features

- Records audio from the user's microphone.
- Sends audio to a backend API for emotion prediction.
- Displays the predicted emotion and its confidence.

## Prerequisites

- Python 3.11.13 (or compatible version)
- `pip` (Python package installer)
<!-- - `npm` or `yarn` (for frontend dependencies, though not strictly necessary for this project as it's pure HTML/CSS/JS) -->

## Installation

1.  **Clone the repository (if applicable):**

    ```bash
    git clone <repository_url>
    cd emotionApp
    ```

2.  **Create a Python virtual environment:**

    It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python -m venv emotion-env
    ```

3.  **Activate the virtual environment:**

    - On Windows:

      ```bash
      .\emotion-env\Scripts\activate
      ```

    - On macOS/Linux:

      ```bash
      source emotion-env/bin/activate
      ```

4.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    This will install `fastapi`, `soundfile`, `transformers`, `librosa`, and `torch`.

## Running the Application

### Backend (FastAPI Server)

The backend server handles the audio processing and emotion prediction.

1.  **Navigate to the `server` directory:**

    ```bash
    cd server
    ```

2.  **Run the FastAPI application using Uvicorn:**

    ```bash
    uvicorn server:app --reload --port 5002
    ```

    The `--reload` flag will automatically restart the server on code changes, and `--port 5002` specifies the port the server will listen on. You should see output indicating that the server is running, typically on `http://127.0.0.1:5002`.

### Frontend

The frontend is a simple web page that allows users to record audio and see the emotion prediction.

1.  **Open the `frontend/index.html` file in your web browser.**

    You can do this by navigating to the file in your file explorer and double-clicking it, or by opening your browser and using `File > Open File...` to select [frontend/index.html](frontend/index.html).

2.  **Allow microphone access** when prompted by your browser.

3.  **Click the "record" button** to start recording your voice.

4.  **Click the "stop" button** to stop recording and send the audio for processing. The predicted emotion will be displayed on the page.

## Project Structure

- `frontend/`: Contains the web-based user interface.
  - [index.html](frontend/index.html): The main HTML file.
  - [script.js](frontend/script.js): JavaScript for audio recording and API interaction.
  - [style.css](frontend/style.css): CSS for styling the frontend.
  - [reset.css](frontend/reset.css): CSS reset for consistent styling.
- `server/`: Contains the FastAPI backend.
  - [server.py](server/server.py): The FastAPI application that handles emotion prediction.
  - `saved_audio/`: Directory for saving audio files (currently not used in the provided code).
- [requirements.txt](requirements.txt): Lists all Python dependencies.
- [readme.md](readme.md): This README file.
- `.gitignore`: Specifies files and directories to be ignored by Git.

<!-- ## Research

The [research/research.ipynb](research/research.ipynb) notebook will be used to test audio files and get a better understanding of how ML models learn. -->
