# Bengali-Text-Emotion-Classifier

This project focuses on classifying emotions from any bengali text using various machine learning models, including fine-tuned BERT, Random Forest, SVM, Logistic Regression, and Naive Bayes. The project includes a Streamlit web application for easy interaction and visualization of the classification results.

## Overview

The primary goal of this project is to classify emotions into categories such as Sad, Happy, Surprise, anger, fear and disgust. The models have been trained and evaluated to provide accurate predictions based on the input text.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/suhanpahari/Bengali-Text-Emotion-Classifier.git
    cd Bengali-Text-Emotion-Classifier
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Models and Performance

Below is the table of performance metrics for each model evaluated on the validation dataset:

| Model                                | Train Accuracy | Test Accuracy | F1 Score (Test Data) | Precision (Test) Data) | Recall (Test Data) |
|--------------------------------------|----------------|---------------|----------------------|------------------------|---------------------|
| Custom BERT Model                    | 0.6430         | 0.4633        | 0.4592               | 0.4716                 | 0.4633              |
| Naive Bayes                          | 0.3175         | 0.27          | 0.27                 | 0.2503                 | 0.2376              |
| Random Forest                        | 0.9929         | 0.39          | 0.39                 | 0.2699                 | 0.2620              |
| SVM                                  | 0.7819         | 0.4533        | 0.3096               | 0.3338                 | 0.3239              |
| Logistic Regression                  | 0.5017         | 0.4067        | 0.4067               | 0.3926                 | 0.2751              |

## GUI Snapshot

![Screenshot 2024-07-28 021927](https://github.com/user-attachments/assets/b6d3c3de-e0f8-42d8-b690-9e1a75f9eeac)

<br>

![image](https://github.com/user-attachments/assets/4ac7b537-2b5b-4ab2-8739-c549d95acfd7)



## Project Information

### Fine-tuned BERT Model

The fine-tuned BERT model (my_bert_model.h5) has been trained on a dataset specifically designed for emotion classification. This model uses the BERT architecture to capture complex semantic relationships in the text. It is loaded with custom layers to handle the BERT encoding and used for predictions in the Streamlit application. This model is capable of understanding context and nuances in Bengali text.

### Random Forest Model

The Random Forest model utilizes a TF-IDF vectorizer to convert text data into numerical features before classification. This ensemble learning method combines multiple decision trees to improve classification accuracy and robustness. It is a reliable model for emotion classification, providing a straightforward approach to handle diverse text inputs.

### SVM Model

The Support Vector Machine (SVM) model, similar to the Random Forest model, uses a TF-IDF vectorizer for feature extraction. SVMs are effective in high-dimensional spaces and are known for their performance in classification tasks. This model separates different emotion classes by finding an optimal hyperplane.

### Logistic Regression Model

The Logistic Regression model uses a TF-IDF vectorizer to transform text data into feature vectors. This model is a popular choice for classification problems due to its simplicity and efficiency. It calculates probabilities to predict the emotion class of the input text.

### Naive Bayes Model

The Naive Bayes model, also based on a TF-IDF vectorizer, employs a probabilistic approach to emotion classification. This model assumes independence between features, making it computationally efficient. It is effective for text classification tasks where the independence assumption holds reasonably well.

### Streamlit Application

The Streamlit application provides a user-friendly interface for emotion classification. Users can input a sentence in Bengali and choose from different models, including SVM, Logistic Regression, Random Forest, BERT Fine-Tuned, and Naive Bayes. The application displays the predicted emotion along with the model's confidence score. It is designed for ease of use and accessibility, making emotion classification interactive and straightforward.

### Tokenization with mBERT

For tokenization, the project uses mBERT (Multilingual BERT) with the bert-base-multilingual-uncased variant. This choice enables handling multiple languages efficiently and improves the accuracy of the emotion classification models.


<h2 id="contact">Contact</h2>
  <p>For any inquiries or feedback, please contact:</p>
  <ul>
    <li><strong>Name:</strong> Soham Pahari</li>
    <li><strong>Education:</strong> B.tech CSE(Data Science) , UPES, Dehradun</li>
    <li><strong>Email:</strong> paharisoham@gmail.com / soham.109424@stu.upes.ac.in</li>
    <li><strong>GitHub:</strong> <a href="https://github.com/suhanpahari">sohamP</a></li>
  </ul>
  
  <h2 id="mentor">Mentor</h2>
  <p><strong>Dr. Sahinur Rahman Laskar</strong><br>
  Assistant Professor<br>
  School of Computer Science, UPES, Dehradun, India<br>
  Email: sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in<br>
  </p>
# emotion
# emotion
