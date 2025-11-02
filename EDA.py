import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load dataset
df = pd.read_csv('./heart_disease_uci.csv')

# Interactive correlation heatmap
def plot_interactive_correlation(df):
    corr = df.corr()
    fig = px.imshow(corr, 
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title='Feature Correlation Matrix')
    fig.show()
    return fig

# Patient demographics by disease
def analyze_demographics(df):
    fig = go.Figure()
    
    for disease in [0, 1]:
        subset = df[df['target'] == disease]
        fig.add_trace(go.Box(
            y=subset['age'],
            name=f'Heart Disease: {disease}',
            boxmean='sd'
        ))
    
    fig.update_layout(title='Age Distribution by Heart Disease Status')
    fig.show()

# Risk factor analysis
def risk_factor_analysis(df):
    risk_factors = ['age', 'chol', 'trestbps', 'thalach']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Risk Factor Distribution by Heart Disease', fontsize=16)
    
    for idx, factor in enumerate(risk_factors):
        ax = axes[idx//2, idx%2]
        df.boxplot(column=factor, by='target', ax=ax)
        ax.set_title(factor.upper())
    
    plt.tight_layout()
    return fig