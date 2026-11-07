# Customer360 Data Directory

This directory stores datasets used by Customer360.

## Supported Datasets

1. **Telco Customer Churn Dataset (`telco_customer_churn.csv`)**
   - Source: Based on the IBM Telco Customer Churn dataset.
   - Purpose: Customer-level subscription features and historical churn indicators for supervised machine learning and K-Means clustering.
   - Key Fields: `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Contract`, `InternetService`, `PaymentMethod`, `Churn`.

2. **Retail Transactions Dataset (`retail_transactions.csv`)**
   - Purpose: Transaction-level order history for RFM (Recency, Frequency, Monetary) segmentation.
   - Key Fields: `TransactionID`, `CustomerID`, `TransactionDate`, `Quantity`, `UnitPrice`, `TotalAmount`.

3. **User Uploaded Datasets**
   - CSV datasets uploaded through the Data Upload page in the application.

