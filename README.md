# PhishGuard AI

PhishGuard AI is an **offline URL phishing classifier**. It uses machine learning to analyze URLs and determine if they are safe or potentially malicious (phishing). 

The entire application runs completely offline, meaning no URLs are sent to external services or third-party APIs. It checks structural features of the URL and compares them against a local database of known trusted brands to catch look-alikes.

## Features

*   **Offline Detection**: No internet required. Complete privacy for scanned URLs.
*   **Machine Learning Powered**: Uses an underlying ML model to evaluate URL structure (length, special characters, suspicious patterns).
*   **Look-alike Detection**: Compares URLs against known safe domains to catch deceptive spellings (e.g., `micr0soft.com`).
*   **Batch Scanning**: Allows you to upload a list of URLs and get bulk results back.
*   **Modern Web UI**: A clean, easy-to-read dashboard built with Flask, HTML, CSS, and Vanilla JavaScript.

## How to Install

1. Make sure you have Python installed.
2. Clone or download this repository.
3. Open your terminal or command prompt in the project folder.
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Start the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and go to: `http://127.0.0.1:5000`

## How to Use

*   **Single Scan**: Enter a URL into the main search bar on the home page and click "Analyze URL". The dashboard will show a risk score, a safety assessment, and the reasons (evidence) behind the score.
*   **Batch Scan**: Go to the "Batch Scan" section, paste a list of URLs (one per line), and click "Scan Batch". You will see a list of results for every URL.

## Tech Stack

*   **Backend**: Python, Flask
*   **Machine Learning**: Scikit-Learn (Joblib)
*   **Frontend**: HTML, CSS, Vanilla JavaScript
