import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle as pkl
import seaborn as sns
sns.set(style='whitegrid')

def main_results():
    # Create the DataFrame
    data = {
        'Method': ['Linear Regression', 'LightGBM', 'ElasticNet', 'Informer', 'Reformer', 'Transformer'],
        'Binance Coin': [-0.00171779, 0.0194612, 0.00174964, -0.00513873, -0.0068208, -0.00568632],
        'Bitcoin': [0.0160228, 0.0237192, 0.0191107, 0.000928189, 0.00182861, 0.000378054],
        'Cardano': [0.0289549, 0.0282305, 0.00230169, 0.00965425, 0.00766372, 0.00716961],
        'Weighted': [0.0178922, 0.0265247, 0.00599041, 0.000145813, 0.000311872, 0.000044829]
    }
    df = pd.DataFrame(data)

    # Set the index to 'Method' for easier plotting
    df.set_index('Method', inplace=True)

    # Plotting
    ax = df.plot.bar(rot=0, figsize=(14, 8))
    ax.set_title('Performance Metrics of Various Methods across Cryptocurrencies')
    ax.set_ylabel('Scores')
    ax.set_xlabel('Method')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("../results/main_results.png")
    # plt.show()

def line_charts():
    data_file = "exprs/lightgbm_simple_strategy_nfold6_emb86400/preds_targets.pkl"
    data = pkl.load(open(data_file, 'rb'))
    preds, targets = data["preds"].flatten(), data["targets"].flatten()
    # Assuming 'actual_values' and 'predicted_values' are lists or arrays of your data
    preds, targets = preds[:1000], targets[:1000]
    print(len(preds), len(targets))
    dates = pd.date_range(start="2021-09-21", periods=len(targets), freq='min')
    # print(dates, targets, preds)
    df = pd.DataFrame({
        'Date': dates,
        'Actual': targets,
        'Predicted': preds
    })

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'].to_numpy(), df['Actual'].to_numpy(), label='Actual', color='blue', linestyle='-')
    plt.plot(df['Date'].to_numpy(), df['Predicted'].to_numpy(), label='Predicted', color='red', linestyle='--')
    plt.title('Comparison of Actual and Predicted Return of Bitcoin')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/line_charts.png")

def feat_importance():
    model_file = "exprs/lightgbm_simple_strategy_nfold6_emb86400/model1.txt"
    model = lgb.Booster(model_file=model_file)
    # Assuming 'model' is your trained LightGBM model
    feature_importances = model.feature_importance(importance_type='gain')
    
    feature_names = ['SMA_60', 'EMA', 'ADX_60',
       'BOP', 'ADXR', 'APO', 'CCI', 'CMO',
       'PPO', 'MOM', 'LCM_60',
       'LRM_60', 'ROC', 'ROCP', 'ROCR',
       'ATR']

    df_features = pd.DataFrame({
        'Features': feature_names,
        'Importance': feature_importances
    })
    df_features.sort_values(by='Importance', ascending=False, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Features', data=df_features.sort_values(by='Importance', ascending=False), palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)
    plt.savefig("results/feat_importance1.png")

def feat_plot(model_i):
    model_file = f"exprs/lightgbm_simple_strategy_nfold6_emb86400/model{model_i}.txt"
    model = lgb.Booster(model_file=model_file)
    # Assuming 'model' is your trained LightGBM model
    feature_importances = model.feature_importance(importance_type='gain')
    
    feature_names = ['SMA_60', 'EMA', 'ADX_60',
       'BOP', 'ADXR', 'APO', 'CCI', 'CMO',
       'PPO', 'MOM', 'LCM_60',
       'LRM_60', 'ROC', 'ROCP', 'ROCR',
       'ATR']

    df_features = pd.DataFrame({
        'Features': feature_names,
        'Importance': feature_importances
    })
    df_features.sort_values(by='Importance', ascending=False, inplace=True)

    sns.barplot(x='Importance', y='Features', data=df_features.sort_values(by='Importance', ascending=False), palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45)

def multi_feat_importances(): 
    data_path = "data/" 
    weight_df = pd.read_csv(data_path + "asset_details.csv")
    id2names = {str(weight_df.loc[ind, "Asset_ID"]):weight_df.loc[ind, "Asset_Name"] for ind in weight_df.index} 
    # Set the Seaborn style for the plots
    sns.set_style('whitegrid')

    # Initialize the figure
    plt.figure(figsize=(12, 12))

    # Create a 3x3 grid of subplots
    asset_ids = [i for i in range(4)] + [i + 5 for i in range(5)]
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        
        feat_plot(asset_ids[i-1])
        # If this is the subplot we want to highlight (for example, subplot #5)
        
        plt.title(id2names[str(asset_ids[i-1])])

    # Adjust the layout
    plt.tight_layout()
    plt.savefig("results/feat_importance.png")

def feature_strategy(): 
    # Initialize the data using a dictionary
    data = {
        'Strategy': ['ReturnStrategy', 'CleanStrategy', 'CleanStrategy2'],
        'Binance Coin': [0.0194612, 0.0117673, 0.0152109],
        'Bitcoin': [0.0237192, 0.0224812, 0.024704],
        'Cardano': [0.0282305, 0.0244421, 0.0277643],
        'Weighted': [0.0265247, 0.0187089, 0.0225287]
    }

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    # Draw a bar plot
    sns.barplot(data=df.melt(id_vars=["Strategy"], var_name="Cryptocurrency", value_name="Performance"),
                x='Strategy', y='Performance', hue='Cryptocurrency', palette='viridis')

    # Add some customization
    plt.title('Performance Metrics of Different Strategies Across Cryptocurrencies', fontsize=16)
    plt.xlabel('Strategy', fontsize=14)
    plt.ylabel('Performance', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Cryptocurrency', title_fontsize='13', fontsize='12')

    # Show the plot
    plt.tight_layout()

    plt.savefig("results/feature_strategy.png")

def deep_results():
        
    # Create a DataFrame from the provided data.
    data = {
        'Method': ['iInformer', 'iReformer', 'iTransformer'],
        'Binance Coin': [-0.00513873, -0.0068208, -0.00568632],
        'Bitcoin': [0.000928189, 0.00182861, 0.000378054],
        'Cardano': [0.00965425, 0.00766372, 0.00716961],
        'Weighted': [0.000145813, 0.000311872, 4.42829e-05]
    }

    df = pd.DataFrame(data)
    df.set_index('Method', inplace=True)

    # Plotting the bar plot.
    ax = df.plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title('Performance Metrics of Crypto Strategies by Method')
    plt.xlabel('Method')
    plt.ylabel('Performance')
    plt.axhline(0, color='gray', linewidth=0.8)  # Add a line at zero for reference
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Cryptocurrency')
    plt.tight_layout()

    plt.savefig("results/deep_results.png")

# line_charts()
# feat_importance()
# multi_feat_importances()
# feature_strategy()
deep_results()