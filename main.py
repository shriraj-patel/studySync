import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder




def create_attendance_dataframe(): 
    file_path2 = "Mendeley/Students_Performance_data_set.xlsx"
    df_attendance = pd.read_excel(file_path2)
    df_attendance = df_attendance[['Average attendance on class', 'What is your current CGPA?']]
    df_attendance.loc[df_attendance['Average attendance on class'] == '94-98', 'Average attendance on class'] = '94'
    df_attendance['Average attendance on class'] = df_attendance['Average attendance on class'].astype(int)
    df_attendance['What is your current CGPA?'] = df_attendance['What is your current CGPA?'].astype(float)
    df_attendance['Attendance'] = df_attendance['Average attendance on class'] / 10
    df_attendance['Attendance'] = df_attendance['Attendance'].astype(int)
    df_attendance['GPA'] = df_attendance['What is your current CGPA?'].astype(int)
    df_attendance.loc[df_attendance['Attendance'] < 5, 'Attendance'] = 4
    df_attendance.loc[df_attendance['Attendance'] == 10, 'Attendance'] = 9
    df_attendance['Count'] = df_attendance.groupby(['Attendance', 'GPA'], observed=True).transform('size')
    df_attendance['Attendance Sum'] = df_attendance.groupby(['Attendance'], observed=True).transform('size')
    df_attendance['%'] = 100 * df_attendance['Count'] / df_attendance['Attendance Sum']
    df_attendance = df_attendance.sort_values(by=['Attendance', 'GPA'])
    return df_attendance


def create_attendance_visualization(df): 
    plt.xticks(ticks=range(6), labels=['0-50', '50-60', '60-70', '70-80', '80-90', '90-100'])
    plt.xlabel('Attendance (%)')
    plt.ylabel('Percentage of Students (%)')
    plt.title('As attendance increases, the average GPA increases substantially')
    ax = sns.barplot(data=df, x='Attendance', y='%', hue='GPA', palette='viridis', order=[4, 5, 6, 7, 8, 9])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=['0-0.99', '1-1.99', '2-2.99', '3-3.99', '4'], title='GPA', loc='right', bbox_to_anchor=(1.22, 0.5))
    plt.show()


def load_grade_data(file_path="Harvard/Student Data set.xlsx"):
    return pd.read_excel(file_path)

def preprocess_data(df):
    print("Initial rows:", len(df))

    grade_mapping = {
        'A': 12, 'A-': 11, 'B+': 10, 'B': 9, 'B-': 8,
        'C+': 7, 'C': 6, 'C-': 5, 'D+': 4, 'D': 3, 'D-': 2, 'E': 1
    }
    for col in ['F1Grade', 'F2Grade', 'F3Grade']:
        df[col] = df[col].map(grade_mapping)

    df = df.dropna(subset=['F1Grade', 'F2Grade', 'F3Grade'], how='all').copy()
    for col in ['F1Grade', 'F2Grade', 'F3Grade']:
        df[col] = df[col].fillna(df[col].median())

    df['AvgGrade'] = df[['F1Grade', 'F2Grade', 'F3Grade']].mean(axis=1)
    df['AvgGrade'] = df['AvgGrade'].fillna(df['AvgGrade'].median())

    df['ADrugs_cleaned'] = df['ADrugs'].astype(str).str.lower().replace({
        'n0': 'no', 'no': 'no', 'N0': 'no', 'NO': 'no',
        'yes': 'yes', 'YES': 'yes'
    })
    df['ADrugs_binary'] = df['ADrugs_cleaned'].map({'yes': 1, 'no': 0})
    df = df[df['ADrugs_binary'].notna()].copy()

    numeric_cols = ['Absence', 'Age', 'FEducation', 'MEducation']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = ['Gender', 'PEmployed', 'Residence']
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.lower()
        df[col] = df[col].fillna('unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    print("Final cleaned rows:", len(df))
    return df

def plot_enhanced_roc_curve(df):
    features = df[['F1Grade', 'F2Grade', 'F3Grade', 'AvgGrade', 'Absence',
                   'Gender', 'Age', 'PEmployed', 'FEducation', 'MEducation',
                   'Residence']]
    target = df['ADrugs_binary']

    df_model = pd.concat([features, target], axis=1).dropna()
    features = df_model[features.columns]
    target = df_model['ADrugs_binary']

    # print("Rows available for training:", len(features))
    if features.empty:
        print("No data to train on after cleaning. Saving empty ROC.")
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title('ROC Curve (No Data Available)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.savefig('roc_curve_enhanced.png')
        plt.close()
        return

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n AUC Score: {auc:.3f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve - Predicting Drug Use\n'
              'Features: F1Grade, F2Grade, F3Grade, AvgGrade, Absence, Gender, Age, '
              'PEmployed, FEducation, MEducation, Residence')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig('roc_curve_enhanced.png')
    plt.show()

def plot_grade_distributions(df):
    plt.figure(figsize=(8, 6))
    adrugs_means = df.groupby('ADrugs_cleaned')['AvgGrade'].mean()
    adrugs_counts = df.groupby('ADrugs_cleaned')['AvgGrade'].count()

    plt.bar(adrugs_means.index, adrugs_means.values, color=['blue', 'orange'])
    plt.title('Average Grade by Actual Drug Use')
    plt.xlabel('Drug Use (ADrugs)')
    plt.ylabel('Average Grade')

    for i, v in enumerate(adrugs_means.values):
        plt.text(i, v + 0.1, f'n={adrugs_counts.iloc[i]}', ha='center')

    plt.savefig('adrugs_bar_plot.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    df.boxplot(column='AvgGrade', by='ADrugs_cleaned')
    plt.title('Grade Distribution by Actual Drug Use')
    plt.suptitle('')
    plt.xlabel('Drug Use (ADrugs)')
    plt.ylabel('Average Grade')
    plt.savefig('adrugs_box_plot.png')
    plt.show()






def create_sleep_dataframe():
    """
    Loads and cleans the Sleep sheet from the Excel file.
    Returns a DataFrame with the 'Hours of sleep' and 'Current GPA' columns.
    """
    # Update the file path to include the FGCU folder and correct filename
    file_path = "FGCU/Survey data_Student Health Behavior and Academic Success.xlsx"
    
    # Load the Sleep sheet
    df = pd.read_excel(file_path, sheet_name="Sleep")
    
    # Clean the "Hours of sleep" column:
    # Extract the first group of digits (e.g., "10 or more hours" becomes "10")
    df["Hours of sleep"] = df["Hours of sleep"].astype(str).str.extract(r'(\d+)')[0]
    df["Hours of sleep"] = pd.to_numeric(df["Hours of sleep"], errors='coerce')
    
    # Ensure Current GPA is numeric
    df["Current GPA"] = pd.to_numeric(df["Current GPA"], errors='coerce')
    
    # Drop rows with missing values in either column
    df = df.dropna(subset=["Hours of sleep", "Current GPA"])
    
    return df

def create_sleep_visualization(df):
    """
    Creates a hexbin heatmap of Hours of Sleep vs. Current GPA.
    """
    plt.figure(figsize=(8,6))
    hb = plt.hexbin(df["Hours of sleep"], df["Current GPA"], gridsize=25, cmap='viridis', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel("Hours of Sleep")
    plt.ylabel("Current GPA")
    plt.title("Heatmap: Hours of Sleep vs. Current GPA")
    plt.show()

def load_social_data(file_path="Mendeley/Students_Performance_data_set.xlsx"):
    """
    Loads the Excel file containing GPA and social media usage data.
    Returns the raw DataFrame.
    """
    return pd.read_excel(file_path)

def preprocess_social_data(df):
    """
    Cleans and bins GPA and social media usage data for visualization.

    Returns:
        pd.DataFrame: cleaned DataFrame with 'GPA Range' and 'SocialMediaTime'
    """
    gpa_col = 'What is your current CGPA?'
    sm_col = 'How many hour do you spent daily in social media?'

    # Drop rows with missing values
    df = df.dropna(subset=[gpa_col, sm_col])
    df[gpa_col] = pd.to_numeric(df[gpa_col], errors='coerce')
    df[sm_col] = pd.to_numeric(df[sm_col], errors='coerce')
    df = df.dropna(subset=[gpa_col, sm_col])

    # Bin GPA into categories
    gpa_bins = [0, 1, 2, 3, 4]
    gpa_labels = ['0–0.99', '1–1.99', '2–2.99', '3–3.99']
    df['GPA Range'] = pd.cut(df[gpa_col], bins=gpa_bins, labels=gpa_labels, include_lowest=True)

    # Bin social media usage into ranges
    sm_bins = [-1, 1, 3, 5, 10, 24]
    sm_labels = ['<1 hr', '1–3 hrs', '3–5 hrs', '5–10 hrs', '10+ hrs']
    df['SocialMediaTime'] = pd.cut(df[sm_col], bins=sm_bins, labels=sm_labels)

    return df


def plot_gpa_by_social_media_time(file_path="Mendeley/Students_Performance_data_set.xlsx"):
    """
    Creates a clustered bar chart showing GPA distribution by time spent on social media daily.
    """

    df = pd.read_excel(file_path)

    # Define column names
    gpa_col = 'What is your current CGPA?'
    sm_col = 'How many hour do you spent daily in social media?'

    # Clean data
    df = df.dropna(subset=[gpa_col, sm_col])
    df[gpa_col] = pd.to_numeric(df[gpa_col], errors='coerce')
    df[sm_col] = pd.to_numeric(df[sm_col], errors='coerce')
    df = df.dropna(subset=[gpa_col, sm_col])

    # Bin GPA
    bins = [0, 1, 2, 3, 4]
    labels = ['0–0.99', '1–1.99', '2–2.99', '3–3.99']
    df['GPA Range'] = pd.cut(df[gpa_col], bins=bins, labels=labels, include_lowest=True)

    # Bin social media hours
    df['SocialMediaTime'] = pd.cut(df[sm_col], bins=[-1, 1, 3, 5, 10, 24],
                                   labels=['<1 hr', '1–3 hrs', '3–5 hrs', '5–10 hrs', '10+ hrs'])

    # Prepare grouped data
    grouped = df.groupby(['SocialMediaTime', 'GPA Range'], observed=False).size().reset_index(name='Count')

    # Pivot the table so GPA ranges become columns (for line plotting)
    pivot_df = grouped.pivot(index='SocialMediaTime', columns='GPA Range', values='Count').fillna(0)

    # Plot
    plt.figure(figsize=(10, 6))
    for gpa_range in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[gpa_range], marker='o', label=gpa_range)

    plt.title('GPA Distribution by Time Spent on Social Media')
    plt.xlabel('Daily Social Media Usage')
    plt.ylabel('Number of Students')
    plt.legend(title='GPA Range')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



   
