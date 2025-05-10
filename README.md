# NextGen Social Network

## Overview

NextGen Social Network is a cutting-edge social networking platform designed to redefine the digital landscape by prioritizing user experience, safety, inclusivity, and creativity. Developed as a major project by students of Cochin University of Science and Technology, this web-based application leverages advanced machine learning techniques to introduce innovative features that empower users and foster a positive online community.

## Features

The platform integrates several advanced features to enhance user engagement and accessibility:

1. **Automatic Hate Speech Detection**

   - Utilizes machine learning models (Logistic Regression, SVM, XGBoost, LinearSVC, Multinomial Naive Bayes) to proactively detect and mitigate hate speech in user-generated content, ensuring a safer online environment.
   - Trained on the Jigsaw Toxic Comment Classification dataset, with experiments conducted on balanced and unbalanced datasets.

2. **Image Caption Generation with Audio**

   - Employs a CNN-LSTM architecture trained on the Flickr8k dataset to generate descriptive captions for images, enhancing accessibility for visually impaired users.
   - Captions can be converted to audio, making visual content more inclusive.

3. **Sentiment Analysis of Trending Posts**

   - Analyzes the emotional tone of trending posts using Logistic Regression and Bernoulli Naive Bayes models, trained on the Sentiment140 dataset.
   - Provides users with insights into the sentiment (positive, negative, or neutral) of discussions, fostering informed engagement.

4. **User-Style Text Generation**

   - Allows users to generate text in a personalized style using an LSTM-based language model trained on "The Art of War" by Sun Tzu.
   - Enables creative expression by producing content tailored to user preferences.

5. **User-Friendly Content Management**

   - Features such as exporting posts to the inbox streamline content sharing and improve usability.
   - Includes user authentication, profile management, follow system, hashtags, mentions, and a responsive design for seamless access across devices.

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**:
  - Libraries: scikit-learn, TensorFlow/Keras, NLTK, pandas, NumPy
  - Models: CNN, LSTM, Logistic Regression, SVM, XGBoost, LinearSVC, Multinomial Naive Bayes
- **Datasets**:
  - Jigsaw Toxic Comment Classification (Hate Speech Detection)
  - Flickr8k (Image Caption Generation)
  - Sentiment140 (Sentiment Analysis)
  - "The Art of War" by Sun Tzu (Text Generation)
- **Tools**: Google Colab (for model training), GitHub (version control)

## Installation

To set up the NextGen Social Network locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Shaliinii28/NextGen-Social-Network.git
   cd NextGen-Social-Network
   ```

2. **Install Dependencies**: Ensure you have Python 3.8+ installed. Install the required packages using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up the Environment**:

   - Configure Flask environment variables (e.g., `FLASK_APP`, `FLASK_ENV`).
   - Ensure access to the trained machine learning models and datasets (available in the repository or downloadable from specified sources).

4. **Run the Application**:

   ```bash
   flask run
   ```

   The application will be accessible at `http://localhost:5000`.

## Usage

- **Sign Up/Login**: Create an account using email or social media credentials.
- **Profile Management**: Customize your profile with a username, profile picture, bio, and cover photo.
- **Post Content**: Share text, images, or links, and engage with posts via likes, comments, or retweets.
- **Explore Features**:
  - Upload images to generate captions automatically.
  - Check sentiment analysis of trending hashtags.
  - Generate creative text in a specific style.
  - Experience real-time hate speech moderation.

## Project Structure

```
NextGen-Social-Network/
├── app.py                 # Flask application
├── templates/             # HTML templates for frontend
├── static/                # CSS, JavaScript, and static assets
├── models/                # Trained ML models
├── datasets/              # Dataset files or links
├── scripts/               # Python scripts for ML tasks
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Challenges Faced

- **Data Imbalance**: Addressed using SMOTE for hate speech detection to balance the dataset.
- **Model Training Time**: Optimized by fine-tuning hyperparameters and leveraging batch processing.
- **Resource Limitations**: Managed by using Google Colab's GPU resources and implementing checkpointing.
- **Runtime Errors**: Debugged using logging, print statements, and careful code inspection.

## Future Scope

- **Enhanced ML Capabilities**: Improve model accuracy with advanced NLP techniques.
- **Multilingual Support**: Add support for multiple languages to enhance inclusivity.
- **AR/VR Integration**: Create immersive user experiences with augmented and virtual reality.
- **Blockchain Authentication**: Use blockchain for secure content authentication.
- **Interactive AI Assistants**: Implement AI-driven recommendations and voice commands.