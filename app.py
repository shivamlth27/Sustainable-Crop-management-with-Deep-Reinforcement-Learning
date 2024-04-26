from flask import Flask, render_template, request
from fr import train_model
import matplotlib
from flask import Flask, request, jsonify


app = Flask(__name__)

#######################################################################################################################


from crop_model import predict
@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    if request.method == 'POST':
        data = request.get_json()
        prediction = predict(data)
        return jsonify({'prediction': prediction[0]})  # Assuming prediction is a single value

#####################################################################################################################################

# Set the backend to a non-GUI backend before importing pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt

crop_mapping = {'Maize': 1, 'Paddy': 0, 'Sugarcane': 2, 'Cotton': 3, 'Tobacco': 4, 'Barley': 5, 'Wheat': 6, 'Millets': 7}
soil_mapping = {'Sandy': 0, 'Loamy': 1, 'Black': 2, 'Red': 3, 'Clayey': 4}


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/crop_pred')
def crop_pred():
    return render_template('crop_pred.html')

@app.route('/yield_prediction')
def yield_prediction():
    return render_template('yield_pred.html')

@app.route('/crop_recommend')
def crop_recommendation():
    return render_template('crop_recommend.html')

@app.route('/rainfall_prediction')
def rainfall_prediction():
    return render_template('rain_pred.html')

@app.route('/fertilizer_recommend')
def fertilizer_recommend():
    return render_template('fertilizer_recommend.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        # Create a dictionary from the form data
        form_data = {
            'temperature': int(request.form.get('t')),
            'humidity': int(request.form.get('h')),
            'soil_moisture': int(request.form.get('soilMoisture')),
            'soil_type': soil_mapping[request.form.get('soil')],
            'crop_type': crop_mapping[request.form.get('crop')],
            'nitrogen': int(request.form.get('n')),
            'phosphorous': int(request.form.get('p')),
            'potassium': int(request.form.get('k'))
        }
        
        # Print the form data
        print(form_data)

        # Pass the form data to the train_model function
        result= train_model(form_data)
        print(result[0][0])
        # Return a response
        return jsonify({'prediction': result[0][0]})


if __name__ == '__main__':
    app.run(debug=True, port=5001)


