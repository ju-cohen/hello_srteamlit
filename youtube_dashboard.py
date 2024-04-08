import os
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# Style funciton:
#   - we only want to apply the styling to numerical columns: the try will go to except if v is a text column
#   - 'props' will contain the parameters to pass to the style

def style_negative(v, props=''):
    """ Style negative values in dataframe"""
    try: 
        return props if v < 0 else None
    except:
        pass
    
def style_positive(v, props=''):
    """Style positive values in dataframe"""
    try: 
        return props if v > 0 else None
    except:
        pass    
    
def audience_simple(country):
    """Show top represented countries"""
    if country == 'US':
        return 'USA'
    elif country == 'IN':
        return 'India'
    else:
        return 'Other'


@st.cache_data
def load_data():
    folder = r"datasets\\Youtube"
    metrics_by_video = "Aggregated_Metrics_By_Video.csv"
    metrics_by_country_sub = "Aggregated_Metrics_By_Country_And_Subscriber_Status.csv"
    comments_data = "All_Comments_final.csv"
    perf_over_time = "Video_Performance_Over_Time.csv"
    # load data
    df_agg = pd.read_csv(os.path.join(folder, metrics_by_video)).iloc[1:,:]  #except first row (header)
    df_agg_sub = pd.read_csv(os.path.join(folder, metrics_by_country_sub))
    df_comments = pd.read_csv(os.path.join(folder, comments_data))
    df_time = pd.read_csv(os.path.join(folder, perf_over_time))

    #some feature engineering
    # renaming columns (ascii issues, that could be dealt wiht more systematically)
    df_agg.columns = ['Video','Video title','Video publish time','Comments added',
                    'Shares','Dislikes','Likes','Subscribers lost',
                    'Subscribers gained','RPM(USD)','CPM(USD)','Average % viewed',
                    'Average view duration','Views','Watch time (hours)','Subscribers',
                    'Your estimated revenue (USD)','Impressions','Impressions ctr(%)']
    df_agg['Video publish time'] = pd.to_datetime(df_agg['Video publish time'], format='%b %d, %Y')  # from text to datetime, for easier sorting/processing
    df_agg['Average view duration'] = df_agg['Average view duration'].apply(lambda x: datetime.strptime(x,'%H:%M:%S'))
    df_agg['Avg_duration_sec'] = df_agg['Average view duration'].apply(lambda x: x.second + x.minute*60 + x.hour*3600)  # seconds is continuous, so much more relevant
    df_agg['Engagement_ratio'] =  (df_agg['Comments added'] + df_agg['Shares'] +df_agg['Dislikes'] + df_agg['Likes']) /df_agg.Views
    df_agg['Views / sub gained'] = df_agg['Views'] / df_agg['Subscribers gained']
    df_agg.sort_values('Video publish time', ascending = False, inplace = True)
    df_time['Date'] = pd.to_datetime(df_time['Date'].str.replace('Sept', 'Sep'))

    return df_agg, df_agg_sub, df_comments, df_time


df_agg, df_agg_sub, df_comments, df_time = load_data()


#engineer data to extract analytics
# first, a differential between the median of stats and the most recent 12 months
df_agg_diff =df_agg.copy()
metric_date_12mo = df_agg_diff['Video publish time'].max() - pd.DateOffset(months=12)
median_agg = df_agg_diff[df_agg_diff['Video publish time'] >= metric_date_12mo].median(numeric_only=True)

# percent difference to the median
numeric_cols = np.array((df_agg_diff.dtypes == 'float64') | (df_agg_diff.dtypes == 'int64'))
df_agg_diff.iloc[:,numeric_cols] = (df_agg_diff.iloc[:,numeric_cols] - median_agg).div(median_agg)

#merge daily data with publish data to get delta 
df_time_diff = pd.merge(df_time, df_agg.loc[:,['Video','Video publish time']], left_on ='External Video ID', right_on = 'Video')
df_time_diff['days_published'] = (df_time_diff['Date'] - df_time_diff['Video publish time']).dt.days

# get last 12 months of data rather than all data 
date_12mo = df_agg['Video publish time'].max() - pd.DateOffset(months =12)
df_time_diff_yr = df_time_diff[df_time_diff['Video publish time'] >= date_12mo]

# get daily view data (first 30 days), median & percentiles 
views_days = pd.pivot_table(df_time_diff_yr,index= 'days_published',values ='Views', aggfunc = [np.mean,np.median,lambda x: np.percentile(x, 80),lambda x: np.percentile(x, 20)]).reset_index()
views_days.columns = ['days_published','mean_views','median_views','80pct_views','20pct_views']
views_days = views_days[views_days['days_published'].between(0,30)]  # keep the first 30 days of every video
views_cumulative = views_days.loc[:,['days_published','median_views','80pct_views','20pct_views']] 
views_cumulative.loc[:,['median_views','80pct_views','20pct_views']] = views_cumulative.loc[:,['median_views','80pct_views','20pct_views']].cumsum()

#build dashboard
add_sidebar = st.sidebar.selectbox('Aggregate or Individual Video', ('Aggregate Metrics','Individual Video Analysis'))
# In the sidebar, we added 2 types of features: aggregate the videos, or individual video analysis

# Final GUI
if add_sidebar == "Aggregate Metrics":
    # Selection of 10 metrics to show (+ Video publish time for date filtering)
    df_agg_metrics = df_agg[['Video publish time','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]
    metric_date_6mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=6)
    metric_date_12mo = df_agg_metrics['Video publish time'].max() - pd.DateOffset(months=12)
    metric_medians6mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_6mo].median(numeric_only=True)
    metric_medians12mo = df_agg_metrics[df_agg_metrics['Video publish time'] >= metric_date_12mo].median(numeric_only=True)

    # Single metric
    #st.metric('Views', metric_medians6mo['Views'], 500)  # args = 'label', 'value', delta

    # Multiple metrics, put in columns programatically
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    count = 0
    for i in metric_medians6mo.index:
        with columns[count]:
            delta = (metric_medians6mo[i] - metric_medians12mo[i])/metric_medians12mo[i]
            st.metric(label=i, value=round(metric_medians6mo[i],1), delta="{:.2%}".format(delta))
            count += 1
            if count >= 5:
                count = 0

    #get date information / trim to relevant data 
    df_agg_diff['Publish_date'] = df_agg_diff['Video publish time'].apply(lambda x: x.date())  #keep only date of the full datetime
    df_agg_diff_final = df_agg_diff.loc[:,['Video title','Publish_date','Views','Likes','Subscribers','Shares','Comments added','RPM(USD)','Average % viewed',
                             'Avg_duration_sec', 'Engagement_ratio','Views / sub gained']]
    
    # format individual columns
    df_agg_numeric_lst = df_agg_diff_final.median(numeric_only=True).index.tolist()   # list of numerical columns
    df_to_pct = {}
    for i in df_agg_numeric_lst:
        df_to_pct[i] = '{:.1%}'.format  # 1 decimal only, percentage
    
    # /!\ semi-colon for the props
    st.dataframe(df_agg_diff_final.style.hide().map(style_negative, props='color:red;').map(style_positive, props='color:green;').format(df_to_pct))
    # the format() at the end: for each of the numerical columns (keys of the df_to_pct dict), give the formatting of 1 decimal %
elif add_sidebar == "Individual Video Analysis":
    videos = tuple(df_agg['Video title'])
    st.write("Individual Video Performance")
    video_select = st.selectbox('Pick a Video:', videos)

    # Select data from the selected video only
    agg_filtered = df_agg[df_agg['Video title'] == video_select]
    agg_sub_filtered = df_agg_sub[df_agg_sub['Video Title'] == video_select]  # second CSV, with subscribers info
    agg_sub_filtered['Country'] = agg_sub_filtered['Country Code'].apply(audience_simple)  # split into USA, India or Other
    agg_sub_filtered.sort_values('Is Subscribed', inplace=True)  # used so that the first value shown on the graph is always the same (True or False)
    
    # Plotly express for a bar chart
    # pass the column names
    fig = px.bar(agg_sub_filtered, x ='Views', y='Is Subscribed', color ='Country', orientation ='h')
    #order axis 
    st.plotly_chart(fig)
    
    agg_time_filtered = df_time_diff[df_time_diff['Video Title'] == video_select]
    first_30 = agg_time_filtered[agg_time_filtered['days_published'].between(0,30)]
    first_30 = first_30.sort_values('days_published')
    
    # Graph Object from plotly
    fig2 = go.Figure()
    # To create lines: scatter plot, but connecting the points as a line
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['20pct_views'],
                    mode='lines',
                    name='20th percentile', line=dict(color='purple', dash ='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['median_views'],
                        mode='lines',
                        name='50th percentile', line=dict(color='black', dash ='dash')))
    fig2.add_trace(go.Scatter(x=views_cumulative['days_published'], y=views_cumulative['80pct_views'],
                        mode='lines', 
                        name='80th percentile', line=dict(color='royalblue', dash ='dash')))
    
    # How does this video perform wrt the others
    fig2.add_trace(go.Scatter(x=first_30['days_published'], y=first_30['Views'].cumsum(),
                        mode='lines', 
                        name='Current Video' ,line=dict(color='firebrick',width=8)))
    
    fig2.update_layout(title='View comparison first 30 days',
                       xaxis_title='Days Since Published',
                       yaxis_title='Cumulative views')
    
    st.plotly_chart(fig2)

