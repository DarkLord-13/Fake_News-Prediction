# Fake News Prediction

This project aims to predict whether a news article is real or fake using natural language processing techniques and machine learning models.

## Project Overview

The project follows these main steps:
1. **Data Collection and Preparation**:
    - Load the dataset containing news articles.
    - Handle missing values.
2. **Data Preprocessing**:
    - Combine author and title to create content.
    - Perform stemming to reduce words to their root form.
    - Convert textual data to feature vectors using TF-IDF.
3. **Model Training**:
    - Split the data into training and testing sets.
    - Train a Logistic Regression model.
4. **Model Evaluation**:
    - Evaluate the model's accuracy on training and testing data.
5. **Prediction**:
    - Make predictions using the trained model.

## Dependencies

The project requires the following dependencies:
- Python 3.x
- NumPy
- Pandas
- NLTK
- Scikit-learn

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/DarkLord-13/Machine-Learning-01.git
    ```

2. Navigate to the project directory:
    ```sh
    cd Machine-Learning-01
    ```

3. Install the required packages:
    ```sh
    pip install numpy pandas nltk scikit-learn
    ```

4. Download the dataset and place it in the appropriate directory.

5. Open the Jupyter Notebook `FakeNewsPrediction.ipynb` and run the cells to execute the project steps:
    ```sh
    jupyter notebook FakeNewsPrediction.ipynb
    ```

## Usage

1. **Data Collection and Preparation**:
    - Load the dataset containing news articles.
    - Handle missing values by filling them with empty strings.

2. **Data Preprocessing**:
    - Combine the author and title columns to create a new content column.
    - Perform stemming on the content column.
    - Convert the textual data to feature vectors using TF-IDF.

3. **Model Training**:
    - Split the data into training and testing sets.
    - Train a Logistic Regression model on the training data.

4. **Model Evaluation**:
    - Evaluate the model's accuracy on the training and testing data.
    - Print the accuracy scores.

5. **Prediction**:
    - Use the trained model to make predictions on new data.

## Results

The trained Logistic Regression model achieved an accuracy of approximately 98% on the training data and 97% on the test data. The model can be used to predict whether a news article is real or fake.

## License

This project is licensed under the MIT License.
