# code.py
from main import (
    load_grade_data,
    preprocess_data,
    plot_grade_distributions,
    plot_enhanced_roc_curve,
    plot_gpa_by_social_media_time

    
)

# Load and preprocess data
df = load_grade_data()
prepared_df = preprocess_data(df)

# Plot enhanced ROC curve
plot_enhanced_roc_curve(prepared_df)

plot_grade_distributions(prepared_df)

plot_gpa_by_social_media_time()
