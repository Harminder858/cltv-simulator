import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
import seaborn as sns
from scipy.stats import pareto, nbinom, gamma

def generate_sample_data(size=1000, freq_mean=2, freq_var=1, recency_max=365, T_max=730):
    np.random.seed(42)
    frequency = np.random.negative_binomial(freq_mean, 1/(1+freq_var), size=size)
    T = np.random.uniform(1, T_max, size=size)
    recency = np.random.uniform(0, np.minimum(recency_max, T))
    recency[frequency == 0] = 0
    monetary_value = np.random.gamma(100, 1, size=size)
    return pd.DataFrame({
        'customer_id': range(1, size + 1),
        'frequency': frequency,
        'recency': recency,
        'T': T,
        'monetary_value': monetary_value
    })

def calculate_cltv(bgf, t, customer_data):
    predicted_purchases = bgf.predict(t, customer_data['frequency'], customer_data['recency'], customer_data['T'])
    return predicted_purchases * customer_data['monetary_value']

def plot_distributions(r, alpha, s, beta):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.linspace(pareto.ppf(0.01, beta), pareto.ppf(0.99, beta), 100)
    ax1.plot(x, pareto.pdf(x, beta), 'r-', lw=2, alpha=0.6, label='Pareto PDF')
    ax1.set_title('Pareto Distribution')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    
    x = np.arange(0, 50)
    ax2.plot(x, nbinom.pmf(x, r, alpha/(alpha+1)), 'b-', lw=2, alpha=0.6, label='NBD PMF')
    ax2.set_title('Negative Binomial Distribution')
    ax2.set_xlabel('Number of Purchases')
    ax2.set_ylabel('Probability')
    
    x = np.linspace(gamma.ppf(0.01, s, scale=1/beta), gamma.ppf(0.99, s, scale=1/beta), 100)
    ax3.plot(x, gamma.pdf(x, s, scale=1/beta), 'g-', lw=2, alpha=0.6, label='Gamma PDF')
    ax3.set_title('Gamma Distribution')
    ax3.set_xlabel('Lifetime')
    ax3.set_ylabel('Density')
    
    plt.tight_layout()
    return fig

def custom_frequency_recency_matrix(bgf, max_frequency=10, max_recency=365):
    """Create a custom frequency/recency matrix."""
    freq_range = range(max_frequency)
    recency_range = range(0, max_recency, 30)  # Every 30 days
    
    Z = np.zeros((len(freq_range), len(recency_range)))
    for i, freq in enumerate(freq_range):
        for j, recency in enumerate(recency_range):
            Z[i, j] = bgf.conditional_expected_number_of_purchases_up_to_time(
                30, freq, recency, recency
            )
    
    return Z, freq_range, recency_range

def plot_frequency_recency_matrix(bgf):
    Z, freq_range, recency_range = custom_frequency_recency_matrix(bgf)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    pcm = ax.imshow(Z, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(len(recency_range))[::2])
    ax.set_xticklabels([f"{r}" for r in recency_range][::2])
    ax.set_yticks(range(len(freq_range)))
    ax.set_yticklabels(freq_range)
    plt.colorbar(pcm, label='Expected Purchases in 30 days')
    ax.set_title('Frequency-Recency Matrix')
    return fig

def main():
    st.title(' Interactive CLTV Pareto/NBD Simulator')
    
    st.write("""
    Welcome to the  CLTV Pareto/NBD Simulator. This tool helps you understand customer lifetime value using the Pareto/NBD model.
    
    How to use this tool:
    1. Use the sidebar to generate sample data or adjust model parameters.
    2. Explore each tab to understand different aspects of the CLTV analysis.
    3. Experiment with different settings to see how they affect the results.
    
    Takeaways from each tab:
    - Data Overview: Understand the basic characteristics of your customer base.
    - Model Parameters: See how different parameters affect the underlying distributions.
    - Customer Segmentation: Visualize how customers are grouped based on their behavior.
    - CLTV Prediction: Predict and analyze customer lifetime value.
    - Individual Customer: Dive deep into individual customer analysis.
    """)
    
    # Sidebar for data generation parameters
    st.sidebar.header("Data Generation Parameters")
    sample_size = st.sidebar.slider("Sample Size", 100, 10000, 1000)
    freq_mean = st.sidebar.slider("Frequency Mean", 0.1, 10.0, 2.0)
    freq_var = st.sidebar.slider("Frequency Variance", 0.1, 10.0, 1.0)
    recency_max = st.sidebar.slider("Max Recency (days)", 1, 1000, 365)
    T_max = st.sidebar.slider("Max T (days)", recency_max, 2000, 730)

    if st.sidebar.button("Generate New Data"):
        st.session_state.data = generate_sample_data(sample_size, freq_mean, freq_var, recency_max, T_max)

    if 'data' not in st.session_state:
        st.session_state.data = generate_sample_data(sample_size, freq_mean, freq_var, recency_max, T_max)

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", 
        "Model Parameters", 
        "Customer Segmentation", 
        "CLTV Prediction", 
        "Individual Customer"
    ])

    # Tab 1: Data Overview
    with tab1:
        st.header("Data Overview")
        st.write("""
        This tab provides a snapshot of your customer data. Understanding these basic metrics helps 
        set the foundation for more advanced analyses.
        
        - Frequency: Number of repeat purchases
        - Recency: Time since last purchase
        - T: Customer's age (time since first purchase)
        - Monetary Value: Average transaction value
        """)
        st.dataframe(st.session_state.data.head(10))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Customers", len(st.session_state.data))
            st.metric("Avg. Frequency", f"{st.session_state.data['frequency'].mean():.2f}")
        with col2:
            st.metric("Avg. Recency (days)", f"{st.session_state.data['recency'].mean():.2f}")
            st.metric("Avg. T (days)", f"{st.session_state.data['T'].mean():.2f}")

    # Tab 2: Model Parameters
    with tab2:
        st.header("Model Parameters")
        st.write("""
        The Pareto/NBD model uses four key parameters to describe customer behavior:
        
        - r: Represents the purchasing process (higher values indicate more frequent purchases)
        - α (alpha): Represents the dropout process (higher values indicate quicker dropout)
        - s: Shape parameter for the gamma distribution of transaction rates
        - β (beta): Scale parameter for the gamma distribution of dropout rates
        
        Adjust these parameters to see how they affect the underlying distributions.
        """)
        col1, col2 = st.columns(2)
        with col1:
            r = st.slider("r (purchase frequency)", 0.1, 5.0, 1.0, 0.1)
            alpha = st.slider("α (time until dropout)", 0.1, 10.0, 1.0, 0.1)
        with col2:
            s = st.slider("s (transaction variability)", 0.1, 5.0, 1.0, 0.1)
            beta = st.slider("β (dropout variability)", 0.1, 10.0, 1.0, 0.1)

        st.pyplot(plot_distributions(r, alpha, s, beta))
        st.write("""
        These graphs show how the model parameters affect the underlying distributions:
        
        - Pareto Distribution: Represents the heterogeneity in customer transaction rates
        - Negative Binomial Distribution: Models the number of purchases made by a customer
        - Gamma Distribution: Represents the heterogeneity in customer lifetimes
        
        Experimenting with these parameters helps understand how different customer behaviors 
        are modeled in the Pareto/NBD framework.
        """)

    # Tab 3: Customer Segmentation
# Tab 3: Customer Segmentation
    with tab3:
        st.header("Customer Segmentation")
        st.write("""
        This visualization helps you understand how customers are segmented based on their 
        purchasing behavior. The heatmap shows the expected number of transactions in the next 30 days
        for customers with different combinations of frequency (past purchases) and recency (days since last purchase).
        
        - Warmer colors (yellow to red) indicate higher expected future purchases
        - Lighter colors (white to light yellow) indicate lower expected future purchases
        
        This segmentation can help in targeting marketing efforts and understanding 
        customer value across different behavioral groups.
        """)
        bgf = BetaGeoFitter(penalizer_coef=0.01)
        bgf.fit(st.session_state.data['frequency'], st.session_state.data['recency'], st.session_state.data['T'])
        fig = plot_frequency_recency_matrix(bgf)
        st.pyplot(fig)

        st.write("""
        Interpreting the Frequency-Recency Matrix:

        1. High Frequency, Low Recency (Top-Right):
           - These are your best customers. They buy often and have bought recently.
           - Action: Focus on retention, exclusive offers, and VIP programs.

        2. High Frequency, High Recency (Top-Left):
           - Frequent buyers who haven't purchased recently. They might be at risk of churning.
           - Action: Re-engagement campaigns, personalized offers based on past purchases.

        3. Low Frequency, Low Recency (Bottom-Right):
           - New customers or occasional buyers who have purchased recently.
           - Action: Encourage repeat purchases, introduce loyalty programs.

        4. Low Frequency, High Recency (Bottom-Left):
           - These customers are at the highest risk of churn.
           - Action: Reactivation campaigns, surveys to understand their needs better.

        5. Gradient from White to Red:
           - The intensity of color represents the expected number of purchases in the next 30 days.
           - Darker red areas indicate customer segments with the highest expected activity.

        Using This Information:
        - Segmentation: Group customers based on their position in this matrix for targeted marketing.
        - Resource Allocation: Focus more resources on customers in the warmer color regions.
        - Campaign Design: Tailor your marketing messages based on the frequency and recency of customer purchases.
        - Customer Journey: Track how customers move across this matrix over time to understand lifecycle stages.

        Remember, this matrix is based on historical data and model predictions. Regular updates and 
        validation against actual customer behavior are crucial for maintaining its accuracy and relevance.
        """)

        # Add an interactive element for exploring specific points
        st.subheader("Explore Specific Customer Segments")
        col1, col2 = st.columns(2)
        with col1:
            freq = st.slider("Frequency", 0, 10, 5)
        with col2:
            recency = st.slider("Recency (days)", 0, 365, 30)
        
        expected_purchases = bgf.conditional_expected_number_of_purchases_up_to_time(
            30, freq, recency, recency
        )
        st.write(f"For a customer with {freq} past purchases and last purchase {recency} days ago:")
        st.write(f"Expected purchases in next 30 days: {expected_purchases:.2f}")

        st.write("""
        Use the sliders above to explore how different combinations of frequency and recency 
        affect the expected number of purchases. This can help you understand the model's 
        predictions for specific customer segments and inform your marketing strategies.
        """)
    # Tab 4: CLTV Prediction
    with tab4:
        st.header("CLTV Prediction")
        st.write("""
        Customer Lifetime Value (CLTV) prediction helps estimate the future value of customers. 
        This is crucial for making informed decisions about customer acquisition, retention, 
        and marketing strategies.
        
        Adjust the prediction time horizon to see how it affects CLTV predictions.
        """)
        t = st.slider('Prediction time horizon (days)', 1, 365, 30)
        cltv = calculate_cltv(bgf, t, st.session_state.data)
        st.session_state.data['predicted_cltv'] = cltv

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(cltv, kde=True, ax=ax)
            ax.set_xlabel('CLTV')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
            st.write("""
            This histogram shows the distribution of predicted CLTVs across your customer base. 
            A right-skewed distribution is common, indicating a small number of high-value customers.
            """)
        with col2:
            st.metric("Mean CLTV", f"${cltv.mean():.2f}")
            st.metric("Median CLTV", f"${cltv.median():.2f}")
            st.metric("Max CLTV", f"${cltv.max():.2f}")
            st.write("""
            These metrics provide insights into your overall customer value:
            - Mean CLTV: Average expected value per customer
            - Median CLTV: Typical customer value (less affected by outliers)
            - Max CLTV: Highest predicted customer value
            """)

        st.subheader("Top 10 Customers by Predicted CLTV")
        st.dataframe(st.session_state.data.nlargest(10, 'predicted_cltv')[['customer_id', 'frequency', 'recency', 'T', 'monetary_value', 'predicted_cltv']])
        st.write("""
        This table shows your top 10 customers based on predicted CLTV. These are your most 
        valuable customers, and understanding their characteristics can help in developing 
        strategies to nurture similar high-value relationships.
        """)

    # Tab 5: Individual Customer
    with tab5:
        st.header("Explore Individual Customer")
        st.write("""
        This section allows you to dive deep into individual customer data. By examining 
        specific customers, you can better understand how the model interprets different 
        purchasing behaviors.
        """)
        customer_id = st.number_input("Enter Customer ID", min_value=1, max_value=len(st.session_state.data), value=1)
        customer = st.session_state.data[st.session_state.data['customer_id'] == customer_id].iloc[0]
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Frequency", f"{customer['frequency']}")
        col2.metric("Recency (days)", f"{customer['recency']:.2f}")
        col3.metric("T (days)", f"{customer['T']:.2f}")
        
        col4, col5 = st.columns(2)
        col4.metric("Monetary Value", f"${customer['monetary_value']:.2f}")
        col5.metric("Predicted CLTV", f"${customer['predicted_cltv']:.2f}")

        st.subheader("Customer's Position in Distributions")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Pareto distribution
        x = np.linspace(pareto.ppf(0.01, beta), pareto.ppf(0.99, beta), 100)
        ax1.plot(x, pareto.pdf(x, beta), 'r-', lw=2, alpha=0.6, label='Pareto PDF')
        ax1.axvline(customer['monetary_value'], color='g', linestyle='--', label='Customer')
        ax1.set_title('Pareto Distribution (Monetary Value)')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Negative Binomial distribution
        x = np.arange(0, 50)
        ax2.plot(x, nbinom.pmf(x, r, alpha/(alpha+1)), 'b-', lw=2, alpha=0.6, label='NBD PMF')
        ax2.axvline(customer['frequency'], color='g', linestyle='--', label='Customer')
        ax2.set_title('Negative Binomial Distribution (Frequency)')
        ax2.set_xlabel('Number of Purchases')
        ax2.set_ylabel('Probability')
        ax2.legend()
        
        # Gamma distribution
        x = np.linspace(gamma.ppf(0.01, s, scale=1/beta), gamma.ppf(0.99, s, scale=1/beta), 100)
        ax3.plot(x, gamma.pdf(x, s, scale=1/beta), 'g-', lw=2, alpha=0.6, label='Gamma PDF')
        ax3.axvline(customer['T'], color='r', linestyle='--', label='Customer')
        ax3.set_title('Gamma Distribution (Customer Age)')
        ax3.set_xlabel('Lifetime')
        ax3.set_ylabel('Density')
        ax3.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        st.write("""
        These graphs show where this specific customer falls within the model's distributions:

        1. Pareto Distribution (Monetary Value):
           - This graph shows the distribution of monetary values across all customers.
           - The green dashed line represents this customer's monetary value.
           - If the line is towards the right, the customer has higher than average transaction values.

        2. Negative Binomial Distribution (Frequency):
           - This graph shows the distribution of purchase frequencies.
           - The green dashed line represents this customer's purchase frequency.
           - A line further to the right indicates a more frequent buyer compared to the average.

        3. Gamma Distribution (Customer Age):
           - This graph represents the distribution of customer lifetimes (T).
           - The red dashed line shows this customer's age (time since first purchase).
           - A line further to the right suggests a longer-standing customer relationship.

        Interpreting these visualizations:
        - Monetary Value: Indicates the customer's spending level compared to others.
        - Frequency: Shows how often the customer makes purchases relative to the average.
        - Customer Age: Reflects the longevity of the customer relationship.

        Business Implications:
        - High Monetary Value + High Frequency: These are your most valuable customers. Consider VIP programs or special attention to retain them.
        - High Monetary Value + Low Frequency: There's potential to increase purchase frequency through targeted promotions.
        - Low Monetary Value + High Frequency: Focus on upselling or cross-selling to increase transaction values.
        - Low Monetary Value + Low Frequency: These customers might need reactivation campaigns or might not be the best fit for your products/services.
        - Customer Age: Longer-standing customers often have higher loyalty. For newer customers with high values in other metrics, consider loyalty programs to increase retention.
        """)

        st.subheader("Customer's Future Value Prediction")
        future_horizons = [30, 90, 180, 365]
        future_values = [calculate_cltv(bgf, t, customer.to_frame().T).iloc[0] for t in future_horizons]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(future_horizons, future_values, marker='o')
        ax.set_xlabel('Time Horizon (Days)')
        ax.set_ylabel('Predicted CLTV ($)')
        ax.set_title('Predicted Future Value')
        st.pyplot(fig)
        
        st.write("""
        This graph projects the customer's predicted lifetime value over different time horizons:
        - A steep upward trend suggests a high-value customer with strong growth potential.
        - A flatter line might indicate a stable but slower-growing customer value.
        - Any dips or plateaus could signal potential churn risks.

        Using this Information:
        1. Short-term Planning: Focus on the 30-90 day predictions for immediate marketing actions or short-term sales forecasts.
        2. Long-term Strategy: Use the 180-365 day predictions for long-term customer relationship management and overall business planning.
        3. Churn Prevention: If the CLTV growth rate slows significantly in longer horizons, consider implementing retention strategies.
        4. Resource Allocation: Prioritize resources and personalized attention based on the projected value growth of different customers.

        Remember, these are predictions based on historical data and model assumptions. Regular model updates and validation against actual results are crucial for maintaining accuracy and reliability in your CLTV predictions.
        """)

        st.subheader("Customer's Position in Frequency-Recency Matrix")
        fig = plot_frequency_recency_matrix(bgf)
        ax = fig.gca()
        ax.plot(customer['recency'] // 30, customer['frequency'], 'ro', markersize=10, label='Customer')
        ax.legend()
        st.pyplot(fig)
        
        st.write("""
        This plot shows where the customer sits in the Frequency-Recency Matrix:
        - The red dot represents the current customer's position.
        - The color of the area where the dot is located indicates the expected future purchases for this customer.
        
        Interpreting the customer's position:
        - Top-right (high frequency, high recency): These are your best customers. They buy often and have bought recently.
        - Top-left (high frequency, low recency): Frequent buyers who haven't purchased recently. They might need reactivation.
        - Bottom-right (low frequency, high recency): New customers or occasional buyers who have purchased recently.
        - Bottom-left (low frequency, low recency): These customers are at the highest risk of churn.

        Business Actions based on Position:
        - For customers in warmer areas (red/orange), focus on retention and increasing their value further.
        - For customers in cooler areas (blue/green), consider reactivation campaigns or special offers to increase engagement.
        - Use this matrix to segment your customer base and tailor your marketing strategies accordingly.
        """)

if __name__ == "__main__":
    main()