import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


app = Flask(__name__)


def create_and_save_model():
   
    data = np.array([
        [100, 50],  
        [120, 60],
        [80, 40],
        [150, 70],
        [90, 30],
        [110, 80]
    ])
    target = np.array([50, 60, 40, 70, 50, 65])  

    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

   
    with open('demand_prediction_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully.")


def load_model():
    try:
        with open('demand_prediction_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
       
        create_and_save_model()
        return load_model()


demand_model = load_model()


def dynamic_price(demand, competitor_price, inventory, base_price=100):
    demand_weight = 0.5
    competitor_weight = 0.3
    inventory_weight = 0.2
    demand_adjustment = demand_weight * (demand / 100) * base_price
    competitor_adjustment = competitor_weight * (competitor_price - base_price)
    inventory_adjustment = -inventory_weight * (inventory / 100) * base_price
    final_price = base_price + demand_adjustment + competitor_adjustment + inventory_adjustment
    return max(final_price, base_price * 0.5)


def predict_demand(competitor_price, inventory):
    features = [[competitor_price, inventory]]
    predicted_demand = demand_model.predict(features)
    return predicted_demand[0]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_dynamic_price', methods=['POST'])
def get_dynamic_price():
    data = request.get_json()
    competitor_price = data.get('competitor_price')
    inventory = data.get('inventory')
    base_price = data.get('base_price', 100)

   
    predicted_demand = predict_demand(competitor_price, inventory)

   
    price = dynamic_price(predicted_demand, competitor_price, inventory, base_price)

    return jsonify({
        "dynamic_price": round(price, 2),
        "predicted_demand": round(predicted_demand, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
