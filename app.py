"""Gradio application for predicting medical insurance costs."""

from pathlib import Path

import gradio as gr
import joblib
import pandas as pd


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "insurance_model.pkl"
FEATURES_PATH = APP_DIR / "feature_columns.pkl"

EXPECTED_FEATURES = [
    "age",
    "bmi",
    "children",
    "sex_male",
    "smoker_yes",
    "region_northwest",
    "region_southeast",
    "region_southwest",
]


def load_artifacts():
    """Load and validate the model artifacts exported from the training notebook."""
    missing = [
        path.name for path in (MODEL_PATH, FEATURES_PATH) if not path.is_file()
    ]
    if missing:
        return None, None, (
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Export them from Colab and place them beside app.py."
        )

    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_features = list(joblib.load(FEATURES_PATH))
    except Exception as exc:  # Keep the UI available and show a useful deployment error.
        return None, None, f"Could not load the saved model files: {exc}"

    if loaded_features != EXPECTED_FEATURES:
        return None, None, (
            "Feature mismatch. feature_columns.pkl must contain, in order: "
            + ", ".join(EXPECTED_FEATURES)
        )

    if not hasattr(loaded_model, "predict"):
        return None, None, "The loaded model does not provide a predict() method."

    return loaded_model, loaded_features, None


model, feature_columns, startup_error = load_artifacts()


def predict_insurance(age, bmi, children, gender, smoker, region):
    """Encode user input exactly as it was encoded during model training."""
    if startup_error:
        raise gr.Error(startup_error)

    if age is None or bmi is None or children is None:
        raise gr.Error("Please complete all numerical fields.")

    age = int(age)
    bmi = float(bmi)
    children = int(children)

    if not 18 <= age <= 100:
        raise gr.Error("Age must be between 18 and 100.")
    if not 10 <= bmi <= 70:
        raise gr.Error("BMI must be between 10 and 70.")
    if not 0 <= children <= 10:
        raise gr.Error("Number of children must be between 0 and 10.")

    row = {column: 0.0 for column in feature_columns}
    row["age"] = age
    row["bmi"] = bmi
    row["children"] = children
    row["sex_male"] = 1.0 if gender == "Male" else 0.0
    row["smoker_yes"] = 1.0 if smoker == "Yes" else 0.0

    region_columns = {
        "Northwest": "region_northwest",
        "Southeast": "region_southeast",
        "Southwest": "region_southwest",
    }
    # Northeast is the baseline category, so every region dummy remains zero.
    if region in region_columns:
        row[region_columns[region]] = 1.0

    input_data = pd.DataFrame([row], columns=feature_columns)
    prediction = float(model.predict(input_data)[0])

    return f"Estimated Insurance Cost: ${prediction:,.2f}"


status_message = (
    "✅ Model files loaded successfully."
    if startup_error is None
    else f"⚠️ **Setup needed:** {startup_error}"
)

with gr.Blocks(title="Medical Insurance Cost Predictor") as app:
    gr.Markdown(
        "# Medical Insurance Cost Predictor\n"
        "Enter the applicant's details to estimate annual medical insurance charges."
    )
    gr.Markdown(status_message)

    with gr.Row():
        with gr.Column():
            age_input = gr.Number(label="Age", value=30, minimum=18, maximum=100, precision=0)
            bmi_input = gr.Number(label="BMI", value=25.0, minimum=10, maximum=70)
            children_input = gr.Number(
                label="Number of Children", value=0, minimum=0, maximum=10, precision=0
            )
        with gr.Column():
            gender_input = gr.Radio(["Female", "Male"], label="Gender", value="Female")
            smoker_input = gr.Radio(["No", "Yes"], label="Smoker", value="No")
            region_input = gr.Dropdown(
                ["Northeast", "Northwest", "Southeast", "Southwest"],
                label="Region",
                value="Northeast",
            )

    predict_button = gr.Button("Predict Insurance Cost", variant="primary")
    result_output = gr.Textbox(label="Prediction Result", interactive=False)

    predict_button.click(
        fn=predict_insurance,
        inputs=[
            age_input,
            bmi_input,
            children_input,
            gender_input,
            smoker_input,
            region_input,
        ],
        outputs=result_output,
    )

    gr.Examples(
        examples=[
            [25, 22.5, 0, "Female", "No", "Northeast"],
            [45, 31.2, 2, "Male", "Yes", "Southeast"],
        ],
        inputs=[
            age_input,
            bmi_input,
            children_input,
            gender_input,
            smoker_input,
            region_input,
        ],
    )

    gr.Markdown(
        "*Educational demonstration only. This prediction is not financial or medical advice.*"
    )


if __name__ == "__main__":
    app.launch(theme=gr.themes.Soft())
