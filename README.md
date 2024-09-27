# CLTV Pareto/NBD Simulator

## Overview

This interactive Customer Lifetime Value (CLTV) Pareto/NBD Simulator is a Streamlit-based web application designed to help users understand and explore the concepts of customer lifetime value using the Pareto/NBD model. It provides a user-friendly interface for generating sample data, adjusting model parameters, and visualizing various aspects of CLTV analysis.

## Features

- **Data Generation**: Create sample customer data with adjustable parameters.
- **Model Parameter Exploration**: Visualize how changes in Pareto/NBD model parameters affect underlying distributions.
- **Customer Segmentation**: Explore customer segments using a frequency-recency matrix.
- **CLTV Prediction**: Calculate and analyze customer lifetime value predictions.
- **Individual Customer Analysis**: Dive deep into individual customer data and predictions.

## How to Use

1. **Data Overview**: Examine basic characteristics of the generated customer base.
2. **Model Parameters**: Adjust Pareto/NBD model parameters and observe their effects on distributions.
3. **Customer Segmentation**: Use the frequency-recency matrix to understand customer segments.
4. **CLTV Prediction**: Analyze CLTV predictions and distribution across customers.
5. **Individual Customer**: Explore detailed information about specific customers.

## Installation

To run this simulator locally:

1. Clone this repository:
   ```
   git clone https://github.com/your-username/cltv-simulator.git
   ```
2. Navigate to the project directory:
   ```
   cd cltv-simulator
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```
   streamlit run cltv_simulator.py
   ```

## Deployment

This app is deployed on Streamlit Sharing. You can access the live version at: 

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Lifetimes

See `requirements.txt` for specific version information.

## Contributing

Contributions to improve the simulator are welcome. Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This simulator uses the Pareto/NBD model implementation from the [Lifetimes](https://github.com/CamDavidsonPilon/lifetimes) library.
- Inspired by various CLTV analysis techniques and best practices in customer analytics.
