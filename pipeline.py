import os
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import SelectKBest, chi2
from datetime import date
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import dataloader as dl

os.system('cls')

df = dl.read_excel_file()
# df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)

# print(df)
# print(df.columns)

def add_customer_age(df):
    current_year = date.today().year
    df_year_birth = df['Year_Birth']
    df['Age'] = current_year - df['Year_Birth']
    return df

# =========================
# Income Analysis
# =========================
CORR_SALARY = 0
if CORR_SALARY == 1:
    df_no_nan = df[df['Income'].notna()].copy()

    # 1
    df_marital_income = df_no_nan.groupby('Marital_Status')['Income'].sum()
    print(df_marital_income)

    # 2
    encode_edu = {
        'Basic' : 1,
        'Graduation' : 2,
        'Master' : 3,
        'PhD' : 4,
        '2n Cycle': 5
    }
    df_no_nan['Edu_level'] = df_no_nan['Education'].map(encode_edu)
    df_no_nan['Age*Edu'] = df_no_nan['Age'] * df_no_nan['Edu_level']
    df_no_nan['Home'] = df_no_nan['Kidhome'] + df_no_nan['Teenhome']
    df_corr = df_no_nan[['Age', 'Edu_level', 'Age*Edu', 'Home', 'Income']]
    correlation_matrix = df_corr.corr()
    print(f"\nCorrelation Matrix ALL:")
    print(correlation_matrix)

    # 3
    marital_list = df_no_nan['Marital_Status'].unique().tolist()
    # ['Single', 'Together', 'Married', 'Divorced', 'Widow', 'Alone', 'Absurd', 'YOLO']
    for marital in marital_list:
        # Filter only current marital group
        df_copy = df_no_nan[df_no_nan['Marital_Status'] == marital].copy()
        # Create new columns inside df_copy only
        df_copy['Edu_level'] = df_copy['Education'].map(encode_edu)
        df_copy['Age*Edu'] = df_copy['Age'] * df_copy['Edu_level']
        df_copy['Home'] = df_copy['Kidhome'] + df_copy['Teenhome']
        # Pick columns for correlation
        df_corr = df_copy[['Age', 'Edu_level', 'Age*Edu', 'Home', 'Income']]
        correlation_matrix = df_corr.corr()
        print(f"\nCorrelation Matrix <{marital}>:")
        print(correlation_matrix)
    
    # impute by method 1, as each marital staus has a big gap, and overall no corr between age/edu to income

def impute_missing_salary(df):
    df_no_nan = df[df['Income'].notna()].copy()
    impute_income_dict = (df_no_nan.groupby('Marital_Status')['Income'].mean()).round(0).astype(int).to_dict()
    # print(impute_income_dict)
    df['Income'] = df['Income'].fillna(df['Marital_Status'].map(impute_income_dict))

    rows_with_any_null = df[df.isnull().any(axis=1)]
    if len(rows_with_any_null) > 0:
        print("WARNING! Null data exists!")
    return df

# =========================
# Demographics
# =========================
CUSTOMER_AGE = 0
if CUSTOMER_AGE == 1:
    values = df['Age']
    plt.hist(values)
    plt.title('Customer Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Number of Customer')
    # plt.show()
    plt.savefig(f"chart\\Histogram Customer Age.png", dpi=300, bbox_inches='tight')
    plt.clf()


def assign_income_level(row):
    if row["Income"] < row["p40"]:
        return "T3"
    elif row["Income"] < row["p80"]:
        return "T2"
    else:
        return "T1"
CUSTOMER_AGE_INCOME = 0
if CUSTOMER_AGE_INCOME == 1:
    df_before_80 = df[df['Age'] <= 80]
    percentiles = df_before_80.groupby('Age')['Income'].quantile([0.4, 0.8]).unstack()
    percentiles.columns = ['p40', 'p80']
    df_before_80 = df_before_80.merge(percentiles, on="Age", how="left")
    df_before_80["Income_Level"] = df_before_80.apply(assign_income_level, axis=1)
    # Count number of customers per age per Income_Level
    age_income_counts = df_before_80.groupby(['Age', 'Income_Level']).size().unstack(fill_value=0)
    # Ensure the order of stacking
    age_income_counts = age_income_counts[['T3', 'T2', 'T1']]  # T3=blue, T2=orange, T1=green
    # Columns order in bars: T3, T2, T1
    colors = ['blue', 'orange', 'green']
    bar_order = ['T3', 'T2', 'T1']
    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(12,6))
    age_income_counts.plot(kind='bar', stacked=True, color=colors, ax=ax)
    # Manually create legend in desired order: T1, T2, T3
    legend_labels = ['T1', 'T2', 'T3']  # Desired order
    legend_colors = ['green', 'orange', 'blue']  # Match the stacked bar colors
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(legend_colors, legend_labels)]
    plt.legend(handles=handles, title='Income Level')
    plt.title('Customer Income Level Distribution by Age')
    plt.xlabel('Age')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"chart\\Bar Customer Income Level Age In.png", dpi=300, bbox_inches='tight')
    plt.clf()


    df_after_80 = df[df['Age'] > 80]
    percentiles = df_after_80.groupby('Age')['Income'].quantile([0.4, 0.8]).unstack()
    percentiles.columns = ['p40', 'p80']
    df_after_80 = df_after_80.merge(percentiles, on="Age", how="left")
    df_after_80["Income_Level"] = df_after_80.apply(assign_income_level, axis=1)
    # Count number of customers per age per Income_Level
    age_income_counts = df_after_80.groupby(['Age', 'Income_Level']).size().unstack(fill_value=0)
    # Ensure the order of stacking
    age_income_counts = age_income_counts[['T3', 'T2', 'T1']]  # T3=blue, T2=orange, T1=green
    # Columns order in bars: T3, T2, T1
    colors = ['blue', 'orange', 'green']
    bar_order = ['T3', 'T2', 'T1']
    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(12,6))
    age_income_counts.plot(kind='bar', stacked=True, color=colors, ax=ax)
    # Manually create legend in desired order: T1, T2, T3
    legend_labels = ['T1', 'T2', 'T3']  # Desired order
    legend_colors = ['green', 'orange', 'blue']  # Match the stacked bar colors
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(legend_colors, legend_labels)]
    plt.legend(handles=handles, title='Income Level')
    plt.title('Customer Income Level Distribution by Age')
    plt.xlabel('Age')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"chart\\Bar Customer Income Level Age Out.png", dpi=300, bbox_inches='tight')
    plt.clf()


# =========================
# Purchasing
# =========================
CUSTOMER_PURCHASING = 0
if CUSTOMER_PURCHASING == 1:
    list_of_mnt = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']

    for col in list_of_mnt:
        df_before_80 = df[df['Age'] <= 80]
        age_mnt_sum = df_before_80.groupby('Age')[col].sum()
        values = age_mnt_sum
        norm = plt.Normalize(age_mnt_sum.min(), age_mnt_sum.max())  # normalize to 0-1
        colors = cm.Blues(norm(age_mnt_sum.values))  # choose a colormap (Blues, Reds, etc.)

        plt.bar(age_mnt_sum.index, age_mnt_sum.values, color=colors)
        plt.title(f'Customer Purchase {col} Distribution')
        plt.xlabel('Age')
        plt.ylabel(f'Amount spent ($)')
        # plt.show()
        plt.savefig(f"chart\\Bar Customer Preference {col} Age In.png", dpi=300, bbox_inches='tight')
        plt.clf()
    
    for col in list_of_mnt:
        df_after_80 = df[df['Age'] > 80]
        age_mnt_sum = df_after_80.groupby('Age')[col].sum()
        values = age_mnt_sum
        norm = plt.Normalize(age_mnt_sum.min(), age_mnt_sum.max())  # normalize to 0-1
        colors = cm.Blues(norm(age_mnt_sum.values))  # choose a colormap (Blues, Reds, etc.)

        plt.bar(age_mnt_sum.index, age_mnt_sum.values, color=colors)
        plt.title(f'Customer Purchase {col} Distribution')
        plt.xlabel('Age')
        plt.ylabel(f'Amount spent ($)')
        # plt.show()
        plt.savefig(f"chart\\Histogram Customer {col} Age Out.png", dpi=300, bbox_inches='tight')
        plt.clf()

# =========================
# Membership & Recency
# =========================
MEMBER_YEAR = 0
if MEMBER_YEAR == 1:
    df_member_year = df
    # Extract year and quarter
    df_member_year['Year'] = df_member_year['Dt_Customer'].dt.year
    df_member_year['Quarter'] = df_member_year['Dt_Customer'].dt.quarter  # 1,2,3,4
    # Group by year and quarter
    df_qtr = df_member_year.groupby(['Year', 'Quarter']).size().unstack(fill_value=0)  # Each column = quarter
    # Plot
    df_qtr.plot(kind='bar', figsize=(10,6))
    plt.title('Customer Registration by Year and Quarter')
    plt.xlabel('Year')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=0)
    plt.legend(title='Quarter', labels=['Q1', 'Q2', 'Q3', 'Q4'])
    plt.tight_layout()
    plt.savefig("chart\\Bar Customer Quarterly Registration.png", dpi=300)
    plt.clf()
    plt.close()

    df_member_year = df
    # Extract year and quarter
    df_member_year['Year'] = df_member_year['Dt_Customer'].dt.year
    df_member_year['Quarter'] = df_member_year['Dt_Customer'].dt.quarter  # 1,2,3,4
    # Keep relevant columns
    list_member_activity_num = [
        'NumDealsPurchases', 'NumWebPurchases', 
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'
    ]
    df_member_activity = df_member_year[list_member_activity_num + ['Year', 'Quarter']]
    # Group by Year and Quarter
    df_qtr = df_member_activity.groupby(['Year', 'Quarter']).sum()
    # Reset index for easier plotting
    df_qtr = df_qtr.reset_index()
    # Create a new column combining Year and Quarter for x-axis
    df_qtr['Year_Quarter'] = df_qtr['Year'].astype(str) + '-Q' + df_qtr['Quarter'].astype(str)
    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(14,7))
    # Plot each activity metric stacked
    bottom = None
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # custom colors
    for i, col in enumerate(list_member_activity_num):
        if bottom is None:
            ax.bar(df_qtr['Year_Quarter'], df_qtr[col], label=col, color=colors[i])
            bottom = df_qtr[col]
        else:
            ax.bar(df_qtr['Year_Quarter'], df_qtr[col], bottom=bottom, label=col, color=colors[i])
            bottom += df_qtr[col]
    # Labels and formatting
    plt.title('Customer Activity by Year and Quarter', fontsize=16)
    plt.xlabel('Year-Quarter', fontsize=12)
    plt.ylabel('Number of Activities', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title='Activity Type')
    plt.tight_layout()
    # plt.show()
    plt.savefig("chart\\Bar Customer Quarterly Activity.png", dpi=300)
    plt.clf()
    plt.close()

# =========================
# Membership & Recency
# =========================
MEMBER_RECENCY = 0
if MEMBER_RECENCY == 1:

    df_before_80 = df[df['Age'] <= 80]
    age_recent_avg = df_before_80.groupby('Age')['Recency'].mean()
    values = age_recent_avg
    norm = plt.Normalize(age_recent_avg.min(), age_recent_avg.max())  # normalize to 0-1
    colors = cm.Blues(norm(age_recent_avg.values))  # choose a colormap (Blues, Reds, etc.)
    plt.bar(age_recent_avg.index, age_recent_avg.values, color=colors)
    plt.title(f'Customer Average Recency Distribution')
    plt.xlabel('Age')
    plt.ylabel(f'Average Recency (Days Since Last Purchase)')
    plt.xticks(ticks=range(min(age_recent_avg.index), max(age_recent_avg.index)+1, 5))
    for x, y in zip(age_recent_avg.index, age_recent_avg.values):
        plt.text(x, y + 0.5,                # slightly above bar
                f'{y:.0f}',                 # no decimal
                ha='center', va='bottom',
                fontsize=8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"chart\\Bar Customer Average Recency Age In.png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


    # Group recency by age
    age_groups = df_before_80.groupby('Age')['Recency']
    # List of ages (sorted)
    ages = sorted(age_groups.groups.keys())
    # List of recency lists
    recency_data = [age_groups.get_group(age).values for age in ages]
    # Compute mean recency per age
    mean_recency = np.array([np.mean(vals) for vals in recency_data])
    # Normalize means to range 0–1 (for colormap)
    norm = (mean_recency - mean_recency.min()) / (mean_recency.max() - mean_recency.min() + 1e-9)
    # Use blue colormap
    cmap = plt.cm.Blues
    colors = [cmap(v) for v in norm]
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    # Create boxplot
    box = ax.boxplot(recency_data, patch_artist=True)
    # Apply gradient colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    # Labels
    ax.set_title("Recency Distribution by Age (Gradient by Average Recency)", fontsize=16)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Recency (Days Since Last Purchase)", fontsize=12)
    # X-axis ticks (show every 5 years)
    ax.set_xticks(range(1, len(ages) + 1))
    ax.set_xticklabels(ages, rotation=0)
    ax.set_xticks(ax.get_xticks()[::5])  # show every 5th label
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"chart\\Box-Plot Customer Average Recency Age In.png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


    df_after_80 = df[df['Age'] > 80]
    age_recent_avg = df_after_80.groupby('Age')['Recency'].mean()
    values = age_recent_avg
    norm = plt.Normalize(age_recent_avg.min(), age_recent_avg.max())  # normalize to 0-1
    colors = cm.Blues(norm(age_recent_avg.values))  # choose a colormap (Blues, Reds, etc.)
    plt.bar(age_recent_avg.index, age_recent_avg.values, color=colors)
    plt.title(f'Customer Average Recency Distribution')
    plt.xlabel('Age')
    plt.ylabel(f'Average Recency (Days Since Last Purchase)')
    plt.xticks(ticks=range(min(age_recent_avg.index), max(age_recent_avg.index)+1, 5))
    for x, y in zip(age_recent_avg.index, age_recent_avg.values):
        plt.text(x, y + 0.5,                # slightly above bar
                f'{y:.0f}',                 # no decimal
                ha='center', va='bottom',
                fontsize=8)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"chart\\Bar Customer Average Recency Age Out.png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


    # Group recency by age
    age_groups = df_after_80.groupby('Age')['Recency']
    # List of ages (sorted)
    ages = sorted(age_groups.groups.keys())
    # List of recency lists
    recency_data = [age_groups.get_group(age).values for age in ages]
    # Compute mean recency per age
    mean_recency = np.array([np.mean(vals) for vals in recency_data])
    # Normalize means to range 0–1 (for colormap)
    norm = (mean_recency - mean_recency.min()) / (mean_recency.max() - mean_recency.min() + 1e-9)
    # Use blue colormap
    cmap = plt.cm.Blues
    colors = [cmap(v) for v in norm]
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))
    # Create boxplot
    box = ax.boxplot(recency_data, patch_artist=True)
    # Apply gradient colors
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    # Labels
    ax.set_title("Recency Distribution by Age (Gradient by Average Recency)", fontsize=16)
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Recency (Days Since Last Purchase)", fontsize=12)
    # X-axis ticks (show every 5 years)
    ax.set_xticks(range(1, len(ages) + 1))
    ax.set_xticklabels(ages, rotation=0)
    ax.set_xticks(ax.get_xticks()[::5])  # show every 5th label
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"chart\\Box-Plot Customer Average Recency Age Out.png", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()


# =========================
# Segmentation
# =========================
CUSTOMER_PREFERENCE = 0
if CUSTOMER_PREFERENCE == 1:
    spend_cols = [
    'MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds'
    ]
    df_spend = df[spend_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_spend)

    # Option 1
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster_k'] = kmeans.fit_predict(X_scaled)

    # Option 2
    hc = AgglomerativeClustering(n_clusters=4, metric='euclidean', linkage='ward')
    df['Cluster_hc'] = hc.fit_predict(X_scaled)

    # Option 3
    dbscan = DBSCAN(eps=2.0, min_samples=5)
    df['Cluster_db'] = dbscan.fit_predict(X_scaled)

    cluster_profile_k = df.groupby('Cluster_k')[spend_cols].mean()
    print(cluster_profile_k)

    cluster_profile_hc = df.groupby('Cluster_hc')[spend_cols].mean()
    print(cluster_profile_hc)

    cluster_profile_db = df.groupby('Cluster_db')[spend_cols].mean()
    print(cluster_profile_db)

    # # Option 1, cluster 0 (Radar)
    # cluster = 3  # example
    # values = cluster_profile_k.loc[cluster].values
    # labels = spend_cols
    # angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    # values = np.concatenate((values, [values[0]]))
    # angles += angles[:1]
    # plt.figure(figsize=(6,6))
    # plt.polar(angles, values)
    # plt.fill(angles, values, alpha=0.3)
    # plt.title(f"Cluster {cluster} Spending Profile")
    # plt.show()

    
    # Option 4
    df_ratio = df_spend.div(df_spend.sum(axis=1), axis=0).fillna(0)
    X_scaled = StandardScaler().fit_transform(df_ratio)
    # Then cluster using K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['Cluster_kr'] = kmeans.fit_predict(X_scaled)
    cluster_profile_kr = df.groupby('Cluster_kr')[spend_cols].mean()
    print(cluster_profile_kr)


if __name__ == "__main__":
    print("All good, End 0")