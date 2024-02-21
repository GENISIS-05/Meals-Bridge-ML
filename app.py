from flask import Flask, request
import numpy as np
from joblib import load
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained LSTM model
model = load_model('model.h5')

scaler = load('minmax_scaler.joblib')

@app.route('/', methods=['GET', 'POST'])
def predict_next_days_api():
    try:
        data = np.array([[0.346875],
       [0.375   ],
       [0.8125  ],
       [0.46875 ],
       [0.40625 ],
       [0.875   ],
       [0.371875],
       [0.46875 ],
       [0.55    ],
       [0.40625 ],
       [0.1875  ],
       [0.46875 ],
       [0.46875 ],
       [0.46875 ],
       [0.209375],
       [0.46875 ],
       [0.365625],
       [0.40625 ],
       [0.40625 ],
       [0.46875 ],
       [0.46875 ],
       [1.      ],
       [0.40625 ],
       [0.40625 ],
       [0.59375 ],
       [0.175   ],
       [0.175   ],
       [0.375   ],
       [0.7125  ],
       [0.46875 ],
       [0.40625 ],
       [0.28125 ],
       [0.375   ],
       [0.40625 ],
       [0.46875 ],
       [0.53125 ],
       [0.46875 ],
       [0.46875 ],
       [0.46875 ],
       [0.40625 ],
       [0.46875 ],
       [0.365625],
       [0.46875 ],
       [0.175   ],
       [0.46875 ],
       [0.175   ],
       [0.28125 ],
       [0.653125],
       [0.40625 ],
       [0.46875 ]])

        # Scale the input data
        scaled_data = scaler.transform(np.array(data).reshape(50, 1))
        num_days = int(request.args.get('num_days', default=21))
        
        custom_data_reshaped = np.reshape(scaled_data, (1, 50, 1))
        predictions = []

        # Generate predictions for the next 30 days
        for _ in range(num_days):
            next_day_prediction = model.predict(custom_data_reshaped)
            predictions.append(next_day_prediction[0, 0])

            # Update 'custom_data_reshaped' for the next iteration
            custom_data_reshaped = np.concatenate(
                (custom_data_reshaped[:, 1:, :], np.reshape(next_day_prediction, (1, 1, 1))),
                axis=1
            )
        predictions = scaler.inverse_transform([predictions])
        predictions = np.floor(predictions)
        return str(predictions)
    except Exception as e:
        error_message = str(e)
        return error_message, 400

if __name__ == '__main__':
    app.run(debug=True)