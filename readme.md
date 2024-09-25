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

```bash
pip install -r requirements.txt
