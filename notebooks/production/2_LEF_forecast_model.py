# Calculating RNBS
# Lindsay Fitzpatrick
# ljob@umich.edu
# 08/19/2024

# This script: 
# 1. Uses the forecast data from CFS and runs it through the trained models to produce
# an ensemble of RNBS forecasts for each of the Great Lakes. 
# 2. Saves the forecast values as a CSV
# 3. Creates a timeseries plot that is saves as a PNG.

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
import calendar

# Directory to downloaded/processed CFS data
dir = f'C:/Users/fitzpatrick/Desktop/Data/'

# Read in the CSVs
pcp_data = pd.read_csv(dir+f'CFS_PCP_forecasts_Sums_CMS.csv',sep=',')
tmp_data = pd.read_csv(dir+f'CFS_TMP_forecasts_Avgs_K.csv',sep=',')
evap_data = pd.read_csv(dir+f'CFS_EVAP_forecasts_Sums_CMS.csv',sep=',')

# Open the trained model(s)
# For the MVP we are only going to use the GP model since it has the best performance
GP_model = joblib.load('GP_trained_model.joblib')

X = pd.DataFrame({
    'su_pcp': pcp_data['sup_lake'],
    'er_pcp': pcp_data['eri_lake'],
    'on_pcp': pcp_data['ont_lake'],
    'mh_pcp': pcp_data['mic_lake']+pcp_data['hur_lake'],
    'su_evap': evap_data['sup_lake'],
    'er_evap': evap_data['eri_lake'],
    'on_evap': evap_data['ont_lake'],
    'mh_evap': evap_data['mic_lake']+evap_data['hur_lake'],
    'su_tmp': tmp_data['sup_lake'],
    'er_tmp': tmp_data['eri_lake'],
    'on_tmp': tmp_data['ont_lake'],
    'mh_tmp': (tmp_data['mic_lake']+tmp_data['hur_lake'])/2
})
print(X)

# Standardize the data
x_scaler = joblib.load('x_scaler.joblib')
y_scaler = joblib.load('y_scaler.joblib')
X_scaled = x_scaler.transform(X)

# Predict RNBS using GP
y_pred_scaled = GP_model.predict(X_scaled)

y_pred = y_scaler.inverse_transform(y_pred_scaled) # unscale the predictions
df_y_pred = pd.DataFrame(y_pred, columns=['sup', 'eri', 'ont', 'mih'])
print(df_y_pred)

df_y_pred['month'] = pcp_data['forecast_month'].astype(int)
df_y_pred['year'] = pcp_data['forecast_year'].astype(int)

current_month = datetime.now().month
current_year = datetime.now().year

filtered_y_pred = df_y_pred[
    (df_y_pred['year'] > current_year) |
    ((df_y_pred['year'] == current_year) & (df_y_pred['month'] >= current_month))
]
filtered_y_pred.to_csv(dir+f'RNBS_forecasts.csv',sep=',',index=False)

print(filtered_y_pred)

def mean_min_max(df,lake):
    mean = df.groupby(['year', 'month'])[lake].mean().reset_index()
    min = df.groupby(['year', 'month'])[lake].min().reset_index()
    max = df.groupby(['year', 'month'])[lake].max().reset_index()

    return mean, min, max

mean_su, min_su, max_su = mean_min_max(filtered_y_pred,'sup')
mean_er, min_er, max_er = mean_min_max(filtered_y_pred,'eri')
mean_on, min_on, max_on = mean_min_max(filtered_y_pred,'ont')
mean_mh, min_mh, max_mh = mean_min_max(filtered_y_pred,'mih')

def plot_rnbs_forecast(x_values, data_dict):
    """
    Plots RNBS forecasts for different lakes.

    Parameters:
    - x_values: array-like, the x-axis values (e.g., months)
    - data_dict: dictionary containing data for each lake. The keys are lake names, and the values are tuples containing
                 (mean, min, max) for that lake.

    The dictionary should be formatted as:
    {
        'Lake Superior': (mean_su, min_su, max_su),
        'Lake Erie': (mean_er, min_er, max_er),
        'Lake Ontario': (mean_on, min_on, max_on),
        'Lake Mich-Huron': (mean_mh, min_mh, max_mh)
    }
    """
    # Create a 4x1 grid of subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # Titles for the subplots
    titles = [
        'Lake Superior 9-month RNBS Forecast',
        'Lake Erie 9-month RNBS Forecast',
        'Lake Ontario 9-month RNBS Forecast',
        'Lake Mich-Huron 9-month RNBS Forecast'
    ]
    
    # Plot each dataset
    for i, (lake_name, (mean, min_val, max_val)) in enumerate(data_dict.items()):
        # Ensure min_val and max_val are scalars

        axs[i].plot(x_values, mean, color='red', linestyle='-', linewidth=1.5, label='Mean')
        axs[i].axhline(y=0, color='black', linestyle='-', linewidth=1.2)
        axs[i].fill_between(x_values, min_val, max_val, color='gray', alpha=0.2)
        axs[i].set_ylabel('RNBS [cms]')
        axs[i].set_title(titles[i])
        axs[i].set_xlim(0, 9)  # Setting x-axis limits from 1 to 10
        axs[i].set_ylim(min_val.min()-500, max_val.max()+500)
        axs[i].grid(True, linestyle='--', alpha=0.6)
        
        if i == 3:  # Set labels for the last subplot
            axs[i].set_xticklabels([calendar.month_abbr[df_y_pred['month'][0]], calendar.month_abbr[df_y_pred['month'][1]],
                                    calendar.month_abbr[df_y_pred['month'][2]], calendar.month_abbr[df_y_pred['month'][3]],
                                    calendar.month_abbr[df_y_pred['month'][4]], calendar.month_abbr[df_y_pred['month'][5]],
                                    calendar.month_abbr[df_y_pred['month'][6]], calendar.month_abbr[df_y_pred['month'][7]],
                                    calendar.month_abbr[df_y_pred['month'][8]], calendar.month_abbr[df_y_pred['month'][9]]])
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(dir+f'RNBS_forecasts.png', bbox_inches='tight')
    plt.show()

# Example usage
x_values = np.arange(len(mean_su['sup']))  # x-axis values
data_dict = {
    'Lake Superior': (mean_su['sup'], min_su['sup'], max_su['sup']),
    'Lake Erie': (mean_er['eri'], min_er['eri'], max_er['eri']),
    'Lake Ontario': (mean_on['ont'], min_on['ont'], max_on['ont']),
    'Lake Mich-Huron': (mean_mh['mih'], min_mh['mih'], max_mh['mih'])
}

plot_rnbs_forecast(x_values, data_dict)

