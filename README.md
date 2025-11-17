# Cricket AI Backend 🏏

This project is a Python-based backend designed to provide AI-powered predictions and insights for cricket matches. It uses machine learning models trained on historical match and player data.

## Overview

The core of this backend consists of several pre-trained machine learning models (`.pkl` files) that predict various aspects of a cricket match, such as:
* Match winners
* Player scores
* Bowler performance
* Dismissal types

The application is likely built using a Python web framework (like **Flask** or **FastAPI**) to serve these predictions via an API.

## Project Structure

* `main.py` / `main2.py`: The main Python application file(s) that run the web server and load the models.
* `requirements.txt`: A list of all Python dependencies required to run the project.
* `*.pkl` files: Serialized, pre-trained machine learning models (e.g., `score_model.pkl`, `bowler_model_optimized.pkl`).
* `*.csv` files: Data files used for training, reference, or statistical lookups (e.g., `player_aggregate_stats.csv`, `head_to_head_stats.csv`).
* `match_winner_output/`: A directory that may be used for storing prediction results or logs.

## Technologies Used

* **Python**
* **Scikit-learn** (for machine learning models)
* **Pandas** (for data manipulation)
* **[Flask / FastAPI]** (As the web framework - *please update this!*)

## Getting Started

Follow these instructions to get a local copy of the project up and running.

### Prerequisites

You must have **Python 3.x** and **pip** (the Python package installer) installed on your system.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/khizr80/cricket-ai-backend.git](https://github.com/khizr80/cricket-ai-backend.git)
    cd cricket-ai-backend
    ```

2.  **Create a virtual environment (Recommended):**
    ```sh
    # For Windows
    python -m venv venv
    venv\Scripts\activate
    
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the backend server, execute the main Python file:

```sh
python main.py
