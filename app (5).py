import gradio as gr
import joblib
import numpy as np

# The entire Gradio application code is placed in a string variable 'app_code'
app_code = """
import gradio as gr
import joblib
import numpy as np

# Load the saved model
model = joblib.load('heart_attack_model.pkl')

def predict_heart_attack(bmi, sleep_hours, physical_days, mental_days, high_risk_lifestyle):
    # The data contains 40 columns; we create an array of 40 zeros
    input_data = np.zeros((1, 40))

    # Fill features based on the most influential columns.
    # IMPORTANT: These indices must match the column order used during model training.
    input_data[0, 31] = bmi                  # BMI
    input_data[0, 7] = sleep_hours           # SleepHours
    input_data[0, 3] = physical_days         # PhysicalHealthDays
    input_data[0, 4] = mental_days           # MentalHealthDays
    input_data[0, 39] = int(high_risk_lifestyle) # High_Risk_Lifestyle

    # The rf_model was trained on unscaled features
    prediction = model.predict(input_data)[0]

    # Return prediction result
    if prediction == 1:
        return "⚠️ Warning: High Risk of Heart Attack! Please consult a physician."
    else:
        return "✅ Patient is at Low Risk. Healthy profile detected."

interface = gr.Interface(
    fn=predict_heart_attack,
    inputs=[
        gr.Slider(10, 60, value=25, label="Body Mass Index (BMI)"),
        gr.Slider(1, 12, value=7, label="Daily Sleep Hours"),
        gr.Slider(0, 30, value=2, label="Physical Health Days (Bad days in last month)"),
        gr.Slider(0, 30, value=2, label="Mental Health Days (Bad days in last month)"),
        gr.Radio(["0", "1"], value="0", label="High Risk Lifestyle Index (1=Yes, 0=No)")
    ],
    outputs=gr.Textbox(label="Heart Attack Prediction Result"),
    title="Cardiovascular Health & Heart Attack Prediction System",
    description="Predicting Heart Attack Risk using Random Forest Classifier (Dataset 2022)."
)

if __name__ == "__main__":
    interface.launch()
"""

with open("app.py", "w") as f:
    f.write(app_code.strip())
print("✅ app.py has been created successfully!")