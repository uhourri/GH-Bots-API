import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from statsmodels.tsa.arima.model import ARIMA

print("hello")

# Read data
activities = pd.read_parquet('../data-raw/activities.parquet')

# Grouping data by contributor and date, and counting up the activities by day
activities_by_day = (
    activities
    .assign(date=pd.to_datetime(activities['date']).dt.date)
    .groupby(['contributor', 'date'])
    .activity
    .count()
    .reset_index(name='n_activities')
)

# Get the data of the top contributor 'sourcegraph-bot' to test time series decomposition method
temp = (
    activities_by_day[activities_by_day['contributor'] == 'sourcegraph-bot']
    .drop(['contributor'], axis=1)
    .reset_index(drop=True)
    .set_index('date', drop=True)
    .asfreq('D')
)
temp.index.name = None


# Group by 'contributor'
contributors = activities_by_day.groupby('contributor')

results_df = pd.DataFrame(columns=['contributor', 'sum_n_activities', 'median_n_activities'])

for name, contributor in contributors:
    data = contributor.copy()

    temp = (
        contributor.copy()
        .drop(['contributor'], axis=1)
        .reset_index(drop=True)
        .set_index('date', drop=True)
        .asfreq('D')
    )
    temp.index.name = None


    # Append the results to the DataFrame
    results_df = results_df.append({
        'contributor': name,
        'sum_n_activities': data['n_activities'].sum(),
        'median_n_activities': data['n_activities'].sum()
    }, ignore_index=True)

    results_df.to_csv('contributors_activities_summary.csv', index=False)