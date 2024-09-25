# Sales Performance Analysis Application

This project provides a comprehensive solution for analyzing sales performance data using various LLMs models. It consists of three main components: a Streamlit application, a transformer model for direct queries, and a Flask API for integrating the functionalities into other applications.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [llama.py](#llamapy)
  - [transformer.py](#transformerpy)
  - [app.py](#apipy)
- [API Endpoints](#api-endpoints)


## Overview

This application is designed to:
- Allow users to upload sales performance data files and query insights using a Streamlit interface.
- Provide direct query capabilities through a Python script using transformers.
- Expose functionalities via a Flask API, enabling integration with other services.

## Requirements

Ensure you have the following Python packages installed:

- Flask
- pandas
- sentence-transformers
- scikit-learn
- streamlit
- numpy

You can install these packages using:
    pip install -r requirements.txt

# Structure And Techniques
### 1. `llama.py`
- Purpose: Integrates Streamlit with the LLaMA model to create an interactive web application.
- Functionality:
  - Users can upload their sales performance data files through a web interface.
  - After uploading, users can ask specific questions regarding the sales performance data.
  - The LLaMA model processes these queries and returns insightful responses based on the uploaded data.
  Run:
    streamlit run llama.py

### 2. `Huggingface.py`
- Purpose: Allows users to run NLP queries directly from the command line.
- Functionality:
  - Users can interact with the model without a web interface by executing the script in their terminal.
  - It processes sales performance queries and returns answers based on the pre-loaded model and data.
  - This is useful for users who prefer a straightforward approach to querying without setting up a web app.
  Run:
    python Huggingface.py


### 3. `app.py`
- Purpose: Sets up a Flask API that provides RESTful endpoints for querying sales performance data.
- Functionality:
  - Three key API endpoints are defined:
    - `/performance_feedback`: Accepts a POST request with the sales representative's name to retrieve performance feedback.
    - `/team_performance`: Returns an assessment of the overall team performance upon a GET request.
    - `/sales_trends_forecasting`: Accepts a POST request to analyze sales trends over a specified period and return forecasting insights.
  - The API uses a pre-loaded dataset to generate responses, making it accessible for various applications requiring sales data insights.

  Run:
    python app.py
