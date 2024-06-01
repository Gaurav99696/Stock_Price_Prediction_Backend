from flask import Flask, request, jsonify
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Ensure that the scaler is fitted with training data before using it
scaler = StandardScaler()

@app.route('/', methods=['POST'])
def home():
    try:
        model = tf.keras.models.load_model('my_model.h5')
        data = request.get_json()

        values_list = list(data.values())

        # Fit and transform the data (note: in real applications, use a pre-fitted scaler)
        norm_data = scaler.fit_transform([values_list])

        predict = model.predict(norm_data)

        zeros = np.zeros((predict.shape[0], 3))
        predictions_concat = np.concatenate((predict, zeros), axis=1)

        prediction = scaler.inverse_transform(predictions_concat)

        prediction_list = prediction[0].tolist()

        return jsonify(prediction_list[0])
    except Exception as e:
        return f"Error: {str(e)}", 400


if __name__ == '__main__':
    app.run()
