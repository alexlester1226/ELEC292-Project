import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib  # for loading the saved model


# Load your trained logistic regression model
# model = joblib.load("logistic_model.pkl")  # Replace with your actual path
# scaler = joblib.load("scaler.pkl")         # If you saved the scaler too

# Replace this with your actual feature extraction function
def extract_features_from_csv(df):
    window = df['Absolute acceleration (m/s^2)'].rolling(window=500).mean().dropna()
    features = [
        window.max(),
        window.min(),
        window.mean(),
        window.median(),
        window.std(),
        window.max() - window.min(),
        window.skew(),
        window.kurtosis(),
        np.sum(np.abs(window)),
        np.count_nonzero(np.diff(np.sign(window)))
    ]
    return np.array([features])


# === Tkinter GUI ===
def run_gui():
    def load_and_classify():
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            features = extract_features_from_csv(df)
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)

            label_result.config(text=f"Predicted: {'Jumping' if prediction[0] == 1 else 'Walking'}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to process file:\n{e}")

    # GUI window
    root = tk.Tk()
    root.title("Activity Classifier")
    root.geometry("400x200")

    title = tk.Label(root, text="Walking vs Jumping Classifier", font=("Helvetica", 14))
    title.pack(pady=20)

    btn_load = tk.Button(root, text="Load CSV File", command=load_and_classify)
    btn_load.pack(pady=10)

    global label_result
    label_result = tk.Label(root, text="", font=("Helvetica", 12))
    label_result.pack(pady=10)

    root.mainloop()


# Run the app
if __name__ == "__main__":
    run_gui()
