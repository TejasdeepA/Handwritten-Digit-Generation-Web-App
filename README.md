# Handwritten Digit Generation Web App

This is a web application that generates handwritten digits based on user input, leveraging a deep learning model trained from scratch.

## 🚀 Overview

The application provides a simple web interface where a user can select a digit from 0 to 9. Upon selection, a pre-trained Conditional Generative Adversarial Network (CGAN) generates five unique, realistic images of the chosen digit.

The project demonstrates an end-to-end workflow: from training a generative model in a cloud environment to deploying it as a publicly accessible web service.

## ✨ Features

-   **User Input**: Select a digit (0-9) from a dropdown menu.
-   **Image Generation**: Generates 5 unique, 28x28 grayscale images of the selected digit.
-   **Live Deployment**: The application is deployed and publicly accessible via Streamlit Community Cloud.
-   **Custom Model**: Uses a custom-trained PyTorch model, not pre-trained weights from the internet.

## 🛠️ Technology Stack

-   **Machine Learning Framework**: PyTorch
-   **Model Architecture**: Conditional Generative Adversarial Network (CGAN)
-   **Dataset**: MNIST
-   **Training Environment**: Google Colab (with T4 GPU)
-   **Web Framework**: Streamlit
-   **Deployment**: Streamlit Community Cloud & GitHub

## Website link:

https://handwritten-digit-generation-web-app-66lfnz2nldfcx74korzsra.streamlit.app/
