import lime.lime_tabular
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Preload step: Initialize model, scaler, and explainer
# Load training data for the LIME explainer
file_path_train = "mitbih_train.csv"
df_train = pd.read_csv(file_path_train)
df_train = df_train.dropna()
X_train = df_train.iloc[:, :-1].values  # Features: 187 time points

# Load pre-fitted scaler (assumes 'scaler.pkl' was saved during training)
scaler = joblib.load('scaler.pkl')

# Scale training data for the explainer
X_train_scaled = scaler.transform(X_train)

# Load the pre-trained XGBoost model
model = joblib.load('XGBoost_softprob_for_LIME.pkl')

# Set up the LIME explainer
class_names = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=[str(i) for i in range(187)],  # Feature names as strings: "0" to "186"
    class_names=class_names,
    mode='classification'
)

def get_lime_explanations_for_instance(x_instance, model=model, scaler=scaler, explainer=explainer):
    """
    Compute LIME explanations for a single test instance from the MIT-BIH dataset.
    
    Parameters:
    -----------
    x_instance : array-like, shape (1, 187) or (187,)
        Single test instance with 187 time points (unscaled).
    model : xgb.Booster, optional (default=preloaded model)
        Preloaded XGBoost model trained with 'multi:softprob'.
    scaler : sklearn.preprocessing.StandardScaler, optional (default=preloaded scaler)
        Pre-fitted StandardScaler for feature scaling.
    explainer : lime.lime_tabular.LimeTabularExplainer, optional (default=preloaded explainer)
        Preloaded LIME explainer configured with scaled training data.
    
    Returns:
    --------
    tuple :
        - scaled_instance : np.ndarray, shape (1, 187)
            The scaled input instance.
        - lime_values : np.ndarray, shape (5, 187)
            LIME explanation values, where 5 is the number of classes and 187 is the number of time points.
        - predicted_class : int
            Predicted class index (0 to 4).
    """
    # Define the prediction function for LIME
    def predict_proba(X):
        dmatrix = xgb.DMatrix(X)
        return model.predict(dmatrix)  # Returns probabilities for 5 classes

    # Convert input to NumPy array and ensure correct shape
    x_instance = np.array(x_instance)
    if x_instance.shape == (187,):
        x_instance = x_instance.reshape(1, -1)  # Reshape to (1, 187)
    if x_instance.shape != (1, 187):
        raise ValueError(f"Expected x_instance shape (1, 187) or (187,), got {x_instance.shape}")

    # Scale the input instance
    x_instance_scaled = scaler.transform(x_instance)  # Shape: (1, 187)

    # Predict the class
    pred_probs = predict_proba(x_instance_scaled)  # Shape: (1, 5)
    predicted_class = np.argmax(pred_probs[0])    # Integer: 0 to 4

    print("Lime explaining")
    # Compute LIME explanation
    exp = explainer.explain_instance(
        x_instance_scaled[0],  # Shape: (187,)
        predict_proba,
        num_features=187,      # Include all features
        top_labels=5           # Explain all 5 classes
    )

    print("LIME explained")
    # Extract LIME values into an array of shape (5, 187)
    lime_values = np.zeros((5, 187))
    for label, feature_weights in exp.local_exp.items():
        for feature, weight in feature_weights:
            lime_values[label, feature] = weight  # label is class index, feature is time point

    return x_instance_scaled, lime_values, predicted_class

# Example usage
if __name__ == "__main__":
    # Load test data for demonstration
    file_path_test = "mitbih_test.csv"
    df_test = pd.read_csv(file_path_test)
    df_test = df_test.dropna()
    x_instance = df_test.iloc[20000, :-1].values  # First test instance, unscaled

    # Get LIME explanations
    scaled_instance, lime_vals, pred_class = get_lime_explanations_for_instance(x_instance)
    
    # Print results
    print("Scaled instance shape:", scaled_instance.shape)  # Expected: (1, 187)
    print("LIME values shape:", lime_vals.shape)            # Expected: (5, 187)
    print("Predicted class:", pred_class)                   # Expected: int (0-4)