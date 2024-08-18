
# Movie Recommendation System

This project demonstrates a movie recommendation system built using TensorFlow and Pandas. The recommendation system is based on collaborative filtering and aims to suggest movies to users based on their previous ratings.

## Project Structure

The project consists of two main components:

1. **Jupyter Notebook**: Contains the implementation of the movie recommendation system, including data preprocessing, model building, and evaluation.
2. **Python Script (`utils.py`)**: Provides utility functions for data preprocessing, model weight management, and generating recommendations.

## Jupyter Notebook

The Jupyter Notebook walks through the following steps:

1. **Data Preparation**:
    - Importing and preprocessing the dataset.
    - Normalizing the ratings data.
    - Creating a matrix of user ratings.

2. **Model Preparation**:
    - Importing weights and biases from external files.
    - Using TensorFlow to create hidden layers and generate movie recommendations.

3. **Generating Recommendations**:
    - Recommending movies for individual users based on their ratings.
    - Generating recommendations for all users and storing the results in a DataFrame.

4. **Model Evaluation**:
    - Merging recommendations with the original dataset to compare predicted scores with actual ratings.
    - Calculating the Root Mean Squared Error (RMSE) to evaluate the model's performance.

## Python Script (`utils.py`)

The `utils.py` file contains the following utility functions:

1. **`get_data()`**: 
    - Imports and preprocesses the movie ratings data.
    - Removes duplicate entries.

2. **`normalize_data(df)`**:
    - Normalizes the ratings data by dividing all values by a normalization factor (5).

3. **`pivot_data()`**:
    - Pivots the DataFrame to create a matrix where rows represent users and columns represent movies.

4. **`get_normalized_data()`**:
    - Combines `pivot_data()` and `normalize_data()` to return a normalized ratings matrix.

5. **`weights()`**:
    - Imports the pre-trained model weights from a CSV file and converts them into a TensorFlow tensor.

6. **`hidden_bias()`** and **`visible_bias()`**:
    - Imports the hidden and visible biases from CSV files and converts them into TensorFlow tensors.

7. **`user_tensor(user_ratings)`**:
    - Converts user ratings into a TensorFlow tensor.

8. **`hidden_layer(v0, W, hb)`**:
    - Computes the hidden layer probabilities.

9. **`reconstructed_output(h0, W, vb)`**:
    - Computes the reconstructed output layer.

10. **`generate_recommendation(user_ratings, W, vb, hb)`**:
    - Generates movie recommendations for a user based on their ratings.

## How to Use

1. **Run the Jupyter Notebook**:
    - Follow the steps outlined in the notebook to preprocess the data, build the model, and generate recommendations.

2. **Use `utils.py`**:
    - Import the utility functions as needed to preprocess the data or generate recommendations programmatically.

## Dependencies

- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Scikit-learn (for RMSE calculation)

## Conclusion

This project provides a hands-on implementation of a collaborative filtering-based recommendation system. By utilizing the Jupyter Notebook and the `utils.py` script, you can explore and experiment with different aspects of the model to enhance its performance.
