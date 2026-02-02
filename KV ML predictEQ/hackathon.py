from photonai.base import Hyperpipe, PipelineElement
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
matplotlib.use('Agg')

def load_and_preprocess_data():
    df = pd.read_csv('../HackPHS - Form Responses 1.csv')
    numeric_cols = ['hours_car', 'Miles_Driven', 'Household_members', 'Meat_week', 'Recyle_per_Pound', ' Trash_per_week']
    categorical_cols = ['Electric_Vehicle', 'Renewable_energy']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['Electric_Vehicle'] = df['Electric_Vehicle'].map({'Yes': 1, 'No': 0})
    df['Renewable_energy'] = df['Renewable_energy'].map({'Yes': 1, 'No': 0})

 
    df['isOver18'] = 1  # Default to over 18, or we can make it random or based on some logic

    return df

def train_model(df):
    X = df[['hours_car', 'Miles_Driven', 'Household_members', 'Electric_Vehicle', 'Meat_week', 'Renewable_energy', 'Recyle_per_Pound', ' Trash_per_week', 'isOver18']]
    emissions = df[' Emissions'].fillna(df[' Emissions'].mean())
    y = emissions

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67)

    my_pipe = Hyperpipe('carbon_emissions_pipe',
                        inner_cv=KFold(n_splits=3),
                        optimizer='grid_search',
                        metrics=['mean_absolute_error'],
                        best_config_metric='mean_absolute_error',
                        verbosity=0)

    my_pipe += PipelineElement('StandardScaler')
    my_pipe += PipelineElement('RandomForestRegressor', hyperparameters={'n_estimators': [50, 100, 200]})

    my_pipe.fit(X_train, y_train)

    best_estimator = my_pipe.optimum_pipe
    with open('photonai_model.pkl', 'wb') as f:
        pickle.dump(best_estimator, f)

    return best_estimator

df = load_and_preprocess_data()
loaded_model = train_model(df)

app = Flask(__name__)
CORS(app)

def process_input_data(data):
    data_for_model = {k: v for k, v in data.items() if k not in ['isOver18', 'phone_number']}
    if ' Trash per week' in data_for_model:
        data_for_model[' Trash_per_week'] = data_for_model.pop(' Trash per week')

    data_for_model['Electric_Vehicle'] = 1 if str(data_for_model.get('Electric_Vehicle', 'No')).lower() in ['yes', 'true', '1'] or data_for_model.get('Electric_Vehicle', False) is True else 0
    data_for_model['Renewable_energy'] = 1 if str(data_for_model.get('Renewable_energy', 'No')).lower() in ['yes', 'true', '1'] or data_for_model.get('Renewable_energy', False) is True else 0

    return data_for_model

def generate_suggestions(data_for_model):
    suggestions = []

    if data_for_model.get('hours_car', 0) > df['hours_car'].mean():
        suggestions.append("Consider reducing car usage by carpooling, biking, or using public transport to lower emissions.")
    if data_for_model.get('Miles_Driven', 0) > df['Miles_Driven'].mean():
        suggestions.append("Try to minimize driving by combining trips or using electric vehicles if possible.")
    if data_for_model.get('Meat_week', 0) > df['Meat_week'].mean():
        suggestions.append("Incorporate more plant-based meals to reduce your carbon footprint from meat consumption.")
    if data_for_model.get('Electric_Vehicle', 0) == 0:
        suggestions.append("Switching to an electric vehicle can significantly decrease your transportation emissions.")
    if data_for_model.get('Renewable_energy', 0) == 0:
        suggestions.append("Consider using renewable energy sources like solar panels for your home.")
    if data_for_model.get('Recyle_per_Pound', 0) < df['Recyle_per_Pound'].mean():
        suggestions.append("Increase recycling efforts to reduce waste and emissions from landfill.")
    if data_for_model.get(' Trash_per_week', 0) > df[' Trash_per_week'].mean():
        suggestions.append("Minimize waste by composting, reusing items, and buying products with less packaging.")

    return suggestions

def generate_comparison_graph(data_for_model):
    categorical_cols = ['Electric_Vehicle', 'Renewable_energy']
    feature_names = [k for k in data_for_model.keys() if k not in categorical_cols]
    user_values = [data_for_model[k] for k in feature_names]
    averages = [df[col].mean() for col in feature_names]

    feature_display_names = {
        'hours_car': 'Hours in Car',
        'Miles_Driven': 'Miles Driven',
        'Household_members': 'Household Members',
        'Meat_week': 'Meat per Week',
        'Recyle_per_Pound': 'Recycle per Pound',
        ' Trash_per_week': 'Trash per Week'
    }
    display_names = [feature_display_names.get(name, name) for name in feature_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(feature_names))
    ax.bar([i - 0.2 for i in x], user_values, width=0.4, label='Your Values', color='blue')
    ax.bar([i + 0.2 for i in x], averages, width=0.4, label='Average Values', color='orange')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right')
    ax.set_ylabel('Values')
    ax.set_title('Your Stats vs Average Stats')
    ax.legend()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)

    return img_base64



def send_imessage_if_needed(phone_number, predicted_emissions, average_emissions, is_above_average, suggestions):
    if phone_number:
        try:
            import platform
            if platform.system() == 'Darwin':
                try:
                    from imessage_kit import IMessageSDK
                    sdk = IMessageSDK()
                    suggestions_text = '\n'.join(suggestions) if suggestions else 'No specific suggestions available.'
                    message = f"Your Carbon Emission Results:\nPredicted: {predicted_emissions:.2f} kg\nAverage: {average_emissions} kg\nAbove Average: {'Yes' if is_above_average else 'No'}\n\nSuggestions:\n{suggestions_text}"
                    sdk.send(phone_number, message)
                    sdk.close()
                    print(f"Suggestions sent to {phone_number} via iMessage")
                except ImportError:
                    print("PhotonAI iMessage kit not available - install with: pip install @photon-ai/imessage-kit")
            else:
                print("iMessage not available on this system (requires macOS)")
        except Exception as e:
            print(f"Error sending iMessage: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        is_over_18 = data.get('isOver18', False)
        average_emissions = 90.0 if is_over_18 else 38.0

        data_for_model = process_input_data(data)

        feature_order = ['hours_car', 'Miles_Driven', 'Household_members', 'Electric_Vehicle', 'Meat_week', 'Renewable_energy', 'Recyle_per_Pound', ' Trash_per_week', 'isOver18']
        ordered_data = {feature: data_for_model.get(feature, 0) for feature in feature_order}
        ordered_data['isOver18'] = 1 if is_over_18 else 0
        input_data = pd.DataFrame([ordered_data])
        predicted_emissions = loaded_model.predict(input_data)[0]
        is_above_average = predicted_emissions > average_emissions

        suggestions = generate_suggestions(data_for_model)
        img_base64 = generate_comparison_graph(data_for_model)

        phone_number = data.get('phone_number', '').strip()
        send_imessage_if_needed(phone_number, predicted_emissions, average_emissions, is_above_average, suggestions)

        return jsonify({
            'predicted_emissions': float(predicted_emissions),
            'average_emissions': average_emissions,
            'is_above_average': bool(is_above_average),
            'graph': img_base64,
            'suggestions': suggestions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/send_imessage', methods=['POST'])
def send_imessage():
    try:
        import platform
        if platform.system() != 'Darwin':
            return jsonify({'message': 'iMessage integration not available on this system. Requires macOS.'})

        data = request.get_json()
        phone_number = data.get('phone_number', '')
        message = data.get('message', '')

        if not phone_number:
            return jsonify({'error': 'Phone number is required'}), 400

        try:
            from imessage_kit import IMessageSDK
            sdk = IMessageSDK()
            sdk.send(phone_number, message)
            sdk.close()
            return jsonify({'message': f'Suggestions sent to {phone_number} via iMessage!'})
        except ImportError:
            return jsonify({'message': 'PhotonAI iMessage kit not available. Install with: pip install @photon-ai/imessage-kit'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/open_rapidai', methods=['POST'])
def open_rapidai():
    try:
        data = request.get_json()
        context = data.get('context', '')
        advanced_suggestions = [
            "Based on your profile, consider offsetting emissions through carbon credits.",
            "Explore local community programs for sustainable living.",
            "Track your progress monthly to stay motivated.",
            "Consider joining carpooling networks to reduce individual vehicle usage.",
            "Investigate smart home energy monitoring systems for better efficiency tracking."
        ]
        enhanced_advice = f"Advanced AI Advice:\n{chr(10).join(advanced_suggestions)}\n\nOriginal Context: {context}"

        return jsonify({'message': 'RapidAI generated advanced personalized advice!', 'advice': enhanced_advice})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
