# code.py
from main import (
    load_grade_data,
    prepare_grade_data,
    plot_roc_curve,
    plot_grade_distributions
)


df = load_grade_data()
prepared_df = prepare_grade_data(df)
plot_roc_curve(prepared_df)
plot_grade_distributions(prepared_df)
