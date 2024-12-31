import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, Canvas, Entry, Button
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import shap

# Load the saved XGB model
XGB_Reg = joblib.load('XGB_Model.pkl')

# Function to predict y for given rotation
def predict_moments(inputs):
    rotations = np.arange(0, 0.121, 0.004)
    predictions = []
    for rotation in rotations:
        input_data = np.array([[*inputs, rotation]])
        prediction = XGB_Reg.predict(input_data)
        predictions.append(prediction[0])
    return rotations, predictions

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Moment-Rotation Curve Predictor")
        self.root.geometry("900x730")
        self.root.configure(bg="#DBDBDB")

        self.create_widgets()

    def create_widgets(self):
        canvas = Canvas(
            self.root,
            bg="#DBDBDB",
            height=730,
            width=900,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        canvas.place(x=0, y=0)


        self.create_rectangles(canvas)
        self.create_text_elements(canvas)
        self.create_entries(canvas)
        self.create_buttons(canvas)

        # Create canvas for plots
        self.plot_canvas = Canvas(self.root, bg="white", width=400, height=200)
        self.plot_canvas.place(x=30, y=480)
        self.shap_canvas = Canvas(self.root, bg="white", width=400, height=200)
        self.shap_canvas.place(x=470, y=480)

        # Load and display the image
        self.load_and_display_image(canvas)

    def create_text_elements(self, canvas):
        texts = [
            (28.0, 17.0, "Define Input Parameters", "Inter SemiBold", 16),
            (483.0, 17.0, "Range of Application", "Inter SemiBold", 16),
            (37.0, 57.0, "Overall width (b)", "Inter", 14),
            (419.0, 55.0, "mm", "Inter", 14),
            (502.0, 51.0, "100 < b (mm) < 200", "Inter", 14),
            (502.0, 98.0, "200 < d (mm) < 300", "Inter", 14),
            (502.0, 141.0, "70 < Sv (mm) < 150", "Inter", 14),
            (502.0, 184.0, "50 < Sh (mm) < 150", "Inter", 14),
            (502.0, 227.0, "10 < tw (mm) < 14", "Inter", 14),
            (502.0, 270.0, "10 < tf (mm) < 14", "Inter", 14),
            (502.0, 313.0, "8 < tp (mm) < 14", "Inter", 14),
            (502.0, 355.0, "16 < db (mm) < 24", "Inter", 14),
            (37.0, 98.0, "Overall depth (d)", "Inter", 14),
            (419.0, 98.0, "mm", "Inter", 14),
            (37.0, 141.0, "Bolt vertical spacing (Sv)", "Inter", 14),
            (419.0, 141.0, "mm", "Inter", 14),
            (37.0, 184.0, "Bolt horizontal spacing (Sh)", "Inter", 14),
            (419.0, 184.0, "mm", "Inter", 14),
            (37.0, 227.0, "Web thickness (tw)", "Inter", 14),
            (419.0, 227.0, "mm", "Inter", 14),
            (37.0, 270.0, "Flange thickness (tf)", "Inter", 14),
            (419.0, 270.0, "mm", "Inter", 14),
            (37.0, 313.0, "Endplate thickness (tp)", "Inter", 14),
            (419.0, 313.0, "mm", "Inter", 14),
            (37.0, 355.0, "Bolt diameter (db)", "Inter", 14),
            (419.0, 355.0, "mm", "Inter", 14),
            (658.0, 300.0, "Note: Dimensions of the column and beam are\nidentical", "Inter", 11),
            (30.0, 455.0, "Predicted moment-rotation curve", "Inter SemiBold", 16),
            (470.0, 455.0, "Impact of geometric parameters", "Inter SemiBold", 16),
            (470.0, 700.0, "Note: The red color indicates that the parameter positively contributes to the prediction ", "Inter", 11)
        ]

        for x, y, text, font_family, font_size in texts:
            canvas.create_text(
                x,
                y,
                anchor="nw",
                text=text,
                fill="#000000",
                font=(font_family, font_size * -1)
            )

    def create_entries(self, canvas):
        self.entry_widgets = []
        self.input_limits = [
            (100, 300),  # Overall depth (d)
            (100, 300),  # Overall width (b)
            (70, 150),   # Bolt vertical spacing (Sv)
            (50, 150),  # Bolt horizontal spacing (Sh)
            (10, 14),   # Web thickness (tw)
            (10, 14),   # Flange thickness (tf)
            (8, 14),    # Endplate thickness (tp)
            (16, 24)   # Bolt diameter (db)
               
        ]
        entries = [
            (228.0, 47.0),  # Overall depth (d)
            (228.0, 90.0),  # Overall width (b)
            (228.0, 133.0),    # Endplate thickness (tp)
            (228.0, 176.0),   # Bolt diameter (db)
            (228.0, 219.0),   # Flange thickness (tf)
            (228.0, 262.0),   # Web thickness (tw)
            (228.0, 305.0),  # Bolt horizontal spacing (Sh)
            (228.0, 348.0)   # Bolt vertical spacing (Sv)
        ]
        for (x, y), (min_val, max_val) in zip(entries, self.input_limits):
            entry = Entry(
                bd=0,
                bg="#FCF8F8",
                fg="#000716",
                highlightthickness=0
            )
            entry.place(x=x, y=y, width=185.0, height=30.0)
            self.entry_widgets.append(entry)
            entry.bind("<FocusOut>", lambda event, e=entry, min_v=min_val, max_v=max_val: self.validate_entry(e, min_v, max_v))

    def create_buttons(self, canvas):
        buttons = [
            (25.0, 410.0, "Predict", self.predict_and_plot),
            (167.0, 410.0, "Clear", self.clear_entries),
            (309.0, 410.0, "Close", lambda: self.root.quit())
        ]
        for x, y, text, command in buttons:
            button = Button(
                text=text,
                borderwidth=0,
                highlightthickness=0,
                command=command,
                relief="flat",
                width=12,
                height=2
            )
            button.place(x=x, y=y)

    def create_rectangles(self, canvas):
        rectangles = [
            (20.0, 15.0, 450.0, 385.0),  # Rectangle for "Define Input Parameters" and input fields
            (470.0, 15.0, 890.0, 385.0)  # Rectangle for "Range of Application" and range labels
        ]
        for x1, y1, x2, y2 in rectangles:
            canvas.create_rectangle(x1, y1, x2, y2, fill="", outline="#535353", width=2)

    def load_and_display_image(self, canvas):
        try:
            # Load the image
            image = Image.open("ima.png")
            original_width, original_height = image.size

            # Define the maximum size
            max_width, max_height = 230, 230

            # Calculate the ratio to maintain the aspect ratio
            ratio = min(max_width / original_width, max_height / original_height)

            # Calculate the new dimensions
            new_width = int(original_width * ratio)
            new_height = int(original_height * ratio)

            # Resize the image while maintaining the aspect ratio
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)

            # Display the image in the Tkinter window
            image_label = ttk.Label(self.root, image=photo)
            image_label.image = photo  # Keep a reference to avoid garbage collection
            image_label.place(x=680, y=60)

        except Exception as e:
            messagebox.showerror("Image Error", f"Error loading image: {e}")

    def validate_input(self, value, min_val, max_val):
        if value == "":
            return False  # Empty input is invalid
        try:
            value = float(value)
            return min_val <= value <= max_val
        except ValueError:
            return False

    def validate_entry(self, entry_widget, min_val, max_val):
        value = entry_widget.get()
        if not self.validate_input(value, min_val, max_val):
            messagebox.showerror("Input Error", f"Input for {entry_widget} is out of range ({min_val}-{max_val}).")
            entry_widget.focus_set()

    def clear_entries(self):
        for entry in self.entry_widgets:
            entry.delete(0, 'end')
            entry.config(bg="#FCF8F8")

        # Clear plots
        if self.plot_canvas:
            for widget in self.plot_canvas.winfo_children():
                widget.destroy()
        if self.shap_canvas:
            for widget in self.shap_canvas.winfo_children():
                widget.destroy()

    def get_input_values(self):
        inputs = []
        for i, entry in enumerate(self.entry_widgets):
            value = entry.get()
            if self.validate_input(value, *self.input_limits[i]):
                inputs.append(float(value))
            else:
                raise ValueError(f"Input for {self.input_labels[i]} is out of range.")
        return inputs

    def predict_and_plot(self):
        try:
            inputs = self.get_input_values()
            rotations, predictions = predict_moments(inputs)
            self.plot_curve(rotations, predictions)
            self.explain_with_shap(inputs)
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def plot_curve(self, rotations, predictions):
        fig, ax = plt.subplots(figsize=(4, 2.15))
        ax.plot(rotations, predictions, color='red')  # Set line color to red
        ax.set_xlabel("Rotation (rad)", fontname='Arial', fontsize=8)  # Set x-axis label font
        ax.set_ylabel("Moment (kNm)", fontname='Arial', fontsize=8)  # Set y-axis label font
        #ax.set_title("Rotation vs Moment", fontname='Arial', fontsize=10)  # Set title font

        # Set limits for x and y axes to ensure (0,0) is visible
        ax.set_xlim(left=0)  # x-axis starts at 0
        ax.set_ylim(bottom=0)  # y-axis starts at 0

        fig.tight_layout()

        # Clear previous plot and create new plot in the canvas
        for widget in self.plot_canvas.winfo_children():
            widget.destroy()
        self.plot_canvas.update_idletasks()
        plot_canvas = FigureCanvasTkAgg(fig, master=self.plot_canvas)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack()
        plt.close(fig)

    def explain_with_shap(self, inputs):
        rotations = np.arange(0, 0.121, 0.004)
        shap_values_list = []

        for rotation in rotations:
            input_data = np.array([[*inputs, rotation]])
            explainer = shap.TreeExplainer(XGB_Reg)
            shap_values = explainer.shap_values(input_data)
            shap_values_list.append(shap_values)

        # Average the SHAP values across all rotations
        avg_shap_values = np.mean(shap_values_list, axis=0)
        avg_shap_values = avg_shap_values[:, :-1]  # Remove the rotation feature

        # Plot the averaged SHAP values
        feature_names = ['Overall width (b)', 'Overall depth (d)', 'Bolt vertical spacing (Sv)',
                         'Bolt horizontal spacing (Sh)', 'Web thickness (tw)',
                         'Flange thickness (tf)', 'Endplate thickness (tp)',
                         'Bolt diameter (db)']
        avg_shap_values = avg_shap_values[0]
        fig, ax = plt.subplots(figsize=(4, 2.15))
        ax.barh(range(len(avg_shap_values)), avg_shap_values, color=['b' if x < 0 else 'r' for x in avg_shap_values])
        ax.set_yticks(range(len(avg_shap_values)))
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_xlabel('SHAP value (Impact on prediction)', fontsize=8)
        #ax.set_title('Impact of geometric features', fontsize=10)
        plt.tight_layout()

        # Clear previous SHAP plot and create new plot in the canvas
        for widget in self.shap_canvas.winfo_children():
            widget.destroy()
        self.plot_canvas.update_idletasks()
        shap_canvas = FigureCanvasTkAgg(fig, master=self.shap_canvas)
        shap_canvas.draw()
        shap_canvas.get_tk_widget().pack()
        plt.close(fig)

# Create Tkinter window
root = tk.Tk()
app = PredictionApp(root)
root.mainloop()