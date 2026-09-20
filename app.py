
import gradio as gr
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("used_car_random_forest.pkl")


# Function to predict price category
def predict_price(car_age, kilometers_driven):

    # Create input data
    input_data = pd.DataFrame({
        "Car_Age": [car_age],
        "Kilometers_Driven": [kilometers_driven]
    })

    # Predict the price category
    prediction = model.predict(input_data)[0]

    return str(prediction)


# Create the Gradio interface
app = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Car Age (Years)"),
        gr.Number(label="Kilometers Driven")
    ],
    outputs=gr.Textbox(label="Predicted Price Category"),
    title="Used Car Price Prediction",
    description="Random Forest Model"
)

# Launch the application
app.launch()
