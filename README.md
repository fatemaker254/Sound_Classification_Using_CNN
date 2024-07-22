# Steps

## Step1
Clone the repo

Create an environment using ```python -m venv venv```
Activate the Environment using  ```venv\Scripts\activate```

Install the requirements
```pip install -r requirements.txt```

## Step2
The dataset folder contains all the spectogram files grouped using elephant and non elephant 

If you want to convert any wav file to spectogram then use ```wav_spectogram.py``` file

## Step 3

The trained model is already pushed with the name ```elephant_sound_classifier.joblib``` 
You can train your own model using the ```train.py``` file with a different dataset

## Step 4

Use the ```predict.py``` file to predict if a particular spectogram is an elephant sound or not
