import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

# Define the drag coefficient models
def drag_coefficient_model_morsi(Re, a1, a2, a3):
    return a1 + (a2 / Re) + (a3 / Re**2)

def drag_coefficient_model_terfous(Re, a1, a2, a3, a4, a5):
    return a1 + (a2 / Re) + (a3 / Re**2) + (a4 / Re**0.1) + (a5 / Re**0.2)

def calculate_rmse(observed, predicted):
    return np.sqrt(np.mean((observed - predicted) ** 2))
    #return np.sqrt(((observed - predicted) ** 2).mean())

def fit_drag_coefficient(data, model_type, curve_color, points_color):
    # Extract C_d and Re from the data
    Cd_data = data['C_d']
    Re_data = data['Re']

    # Select model and initial guess
    if model_type == 'Morsi and Alexander (1972)':
        model = drag_coefficient_model_morsi
        initial_guess = [1.0, 1.0, 1.0]  # Initial guess for [a1, a2, a3]
    elif model_type == 'Terfous et al. (2013)':
        model = drag_coefficient_model_terfous
        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]  # Initial guess for [a1, a2, a3, a4]
    
    # Perform curve fitting
    popt, _ = curve_fit(model, Re_data, Cd_data, p0=initial_guess)

    # Extract the fitted parameters
    fitted_params = popt

    # Generate values for plotting the fitted curve
    Re_fit = np.linspace(min(Re_data), max(Re_data), 500)
    if model_type == 'Morsi and Alexander (1972)':
        Cd_fit = model(Re_fit, *fitted_params)
        Cd_pred = model(Re_data, *fitted_params)
    elif model_type == 'Terfous et al. (2013)':
        Cd_fit = model(Re_fit, *fitted_params)
        Cd_pred = model(Re_data, *fitted_params)

    # Calculate RMSE
    rmse = calculate_rmse(Cd_data, Cd_pred)

    # Plot the original data and the fitted curve
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(Re_data, Cd_data, label='Data', color=points_color, s=50, linewidth=1.0, edgecolor='black')
    ax.plot(Re_fit, Cd_fit, color=curve_color, label='Fitted Curve', linewidth=2.0)
    ax.set_xlabel('Reynolds number (Re)', fontsize=14)
    ax.set_ylabel('Drag Coefficient ($C_d$)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)

    st.pyplot(fig)

    return fitted_params, rmse

def main():
    st.title('Drag Coefficient Fitting')

    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader('Data from CSV file:')
        st.write(data)

        # Get all possible color names
        color_names = list(mcolors.CSS4_COLORS.keys())
        
        # Set default colors
        default_curve_color = 'red'
        default_points_color = 'black'

        # Select the model
        model_type = st.selectbox('Select drag coefficient model:', 
                                  ['Morsi and Alexander (1972)', 'Terfous et al. (2013)'])

        # Select colors
        curve_color = st.selectbox('Select curve color:', color_names, index=color_names.index(default_curve_color))
        points_color = st.selectbox('Select points color:', color_names, index=color_names.index(default_points_color))

        fitted_params, rmse = fit_drag_coefficient(data, model_type, curve_color, points_color)

        st.subheader('Fitted Coefficients:')
        if model_type == 'Morsi and Alexander (1972)':
            st.write(f"a1 = {fitted_params[0]:.4f}")
            st.write(f"a2 = {fitted_params[1]:.4f}")
            st.write(f"a3 = {fitted_params[2]:.4f}")
        elif model_type == 'Terfous et al. (2013)':
            st.write(f"a1 = {fitted_params[0]:.4f}")
            st.write(f"a2 = {fitted_params[1]:.4f}")
            st.write(f"a3 = {fitted_params[2]:.4f}")
            st.write(f"a4 = {fitted_params[3]:.4f}")
            st.write(f"a5 = {fitted_params[4]:.4f}")

        st.subheader('Fitting Error:')
        st.write(f"Root Mean Square Error (RMSE) = {rmse:.4f}")

if __name__ == '__main__':
    main()

