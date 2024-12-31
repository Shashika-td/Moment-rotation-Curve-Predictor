# Moment-Rotation Curve Predictor

This is a Tkinter-based GUI application that predicts moment-rotation curves for beam-to-column bolted connections using a pre-trained XGBoost regression model. The application also provides SHAP-based explanations to show the impact of input parameters on the predictions.

---

## Requirements

### Python Version:
- Python 3.12 or higher

### Libraries:

The following libraries are required to run the Python script. Each library can be installed by running the following commands in the command prompt:

- `joblib`: Install using
  ```bash
  pip install joblib
  ```
- `numpy`: Install using
  ```bash
  pip install numpy
  ```
- `matplotlib`: Install using
  ```bash
  pip install matplotlib
  ```
- `Pillow`: Install using
  ```bash
  pip install pillow
  ```
- `shap`: Install using
  ```bash
  pip install shap
  ```

---

## Setup Instructions

   - Download the primary script `Moment_Rotation_Predictor.py`.
   - Download the pre-trained model file `XGB_Model.pkl`.
   - Ensure both files are located in the same directory.
   - Verify that all required libraries are installed as specified in the requirements section.
   - Run the Application:
   ```bash
   python Moment_Rotation_Predictor.py
   ```

---

## Note:

Please ensure that the input values for each parameter fall within the specified range of applications mentioned in the GUI. Inputs outside those ranges may lead to an error message.

---

## Contact

For any questions, please contact me via email at: [shashikagtdharmawansha@gmail.com](mailto:shashikagtdharmawansha@gmail.com)

