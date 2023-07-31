import pickle
import pandas as pd
from flask import Flask, render_template, request

# Assuming your trained model is saved in a file named 'model.pkl'
with open('model/rfa.pckl', 'rb') as file:
    model1 = pickle.load(file)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get input from the user
        metascore = float(request.form['metascore'])
        reviews = float(request.form['reviews'])
        likes = float(request.form['likes'])

        # Create a DataFrame with the user input
        input_data = pd.DataFrame({'Metascore': [metascore], 'Reviews': [reviews], 'Likes': [likes]})

        # Make predictions using the loaded model
        predictions = model1.predict(input_data)

        # Convert prediction to 'Success' or 'Flop'
        result = 'Success' if predictions[0] == 1 else 'Flop'

        return render_template('index.html', prediction=result)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True,port=7871)
