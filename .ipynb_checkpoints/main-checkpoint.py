import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve




def create_attendance_dataframe(): 
    file_path2 = "Mendeley/Students_Performance_data_set.xlsx"
    df_attendance = pd.read_excel(file_path2)
    df_attendance = df_attendance[['Average attendance on class', 'What is your current CGPA?']]
    df_attendance = df_attendance[df_attendance['Average attendance on class'] != '94-98']
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
    plt.title('Relationship between attendance % and GPA')
    ax = sns.barplot(data=df, x='Attendance', y='%', hue='GPA', palette='viridis', order=[4, 5, 6, 7, 8, 9])
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=['0-0.99', '1-1.99', '2-2.99', '3-3.99', '4'], title='GPA', loc='right', bbox_to_anchor=(1.22, 0.5))
    plt.show()


def load_grade_data(file_path="Harvard/Student Data set.xlsx"):
    df = pd.read_excel(file_path)
    return df

def prepare_grade_data(df):
    new_df = df[['F1Grade', 'F2Grade', 'F3Grade', 'ADrugs']].copy()

    grade_mapping = {
        'A': 12, 'A-': 11, 'B+': 10, 'B': 9, 'B-': 8, 
        'C+': 7, 'C': 6, 'C-': 5, 'D+': 4, 'D': 3, 'D-': 2, 'E': 1
    }
    for col in ['F1Grade', 'F2Grade', 'F3Grade']:
        new_df[col] = new_df[col].map(grade_mapping)

    new_df['AvgGrade'] = new_df[['F1Grade', 'F2Grade', 'F3Grade']].mean(axis=1)

    new_df['ADrugs_cleaned'] = new_df['ADrugs'].astype(str).str.lower().replace({
        'n0': 'no', 'no': 'no', 'N0': 'no', 'NO': 'no', 'yes': 'yes', 'YES': 'yes'
    })
    new_df['ADrugs_binary'] = new_df['ADrugs_cleaned'].map({'yes': 1, 'no': 0})

    new_df.dropna(subset=['F1Grade', 'F2Grade', 'F3Grade', 'AvgGrade', 'ADrugs_binary'], inplace=True)

    correlation = new_df[['ADrugs_binary', 'AvgGrade']].corr()
    print("Correlation between Drug Use and Average Grade:\n", correlation)

    return new_df

def plot_roc_curve(df):
    features = df[['F1Grade', 'F2Grade', 'F3Grade', 'AvgGrade']]
    target = df['ADrugs_binary']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title('ROC Curve - Predicting Drug Use from Grades')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig('roc_curve.png')
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

