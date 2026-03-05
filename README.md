# JP Morgan Quantitative Research Virtual Experience

This repository contains my work from the JP Morgan Quantitative Research virtual experience on Forage. The project focuses on applying quantitative methods and Python programming to problems commonly encountered in financial markets and risk management.

The tasks involve analysing commodity price data, valuing a storage contract, modelling credit risk, and constructing a rating system from borrower credit scores.

## Task 1 – Natural Gas Price Analysis

Historical monthly natural gas prices were analysed to identify trends and patterns in the data. A price estimation function was implemented to estimate gas prices for any given date using interpolation within the observed data and extrapolation outside the dataset.

This model allows prices to be estimated for arbitrary future or past dates, which can be used as an input for pricing financial contracts.

## Task 2 – Natural Gas Storage Contract Pricing

Using the estimated gas price function, a pricing model was developed for a natural gas storage contract. The model simulates the cash flows generated when gas is injected into storage and later withdrawn and sold.

The contract value is determined by considering:

- the dates of gas injection and withdrawal  
- gas purchase and sale prices  
- injection and withdrawal rates  
- maximum storage capacity  
- storage costs and transaction costs  

The model returns the value of the storage contract by calculating total revenue from sales minus purchasing and operational costs.

## Task 3 – Credit Risk Model

A logistic regression model was built to estimate the probability of default (PD) for loan borrowers using borrower characteristics such as:

- income  
- FICO score  
- total outstanding debt  
- number of credit lines  
- years of employment  

Additional financial ratios, including debt-to-income and payment-to-income, were constructed to improve predictive power.

The model outputs the probability that a borrower will default on a loan.

## Task 4 – Credit Rating Quantization

A credit rating system was created by mapping FICO scores into discrete rating buckets. The goal was to determine bucket boundaries that best summarise the data.

The optimisation approach maximises the log-likelihood of observed defaults within each bucket, producing rating groups that reflect both borrower density and default behaviour.

The resulting system assigns ratings such that lower rating numbers correspond to stronger credit quality.

## Expected Loss Calculation

Using the probability of default from the model, the expected loss on a loan can be estimated using:

Expected Loss = Probability of Default × Loss Given Default × Exposure

A recovery rate of 10% was assumed, meaning the loss given default is 90% of the loan amount.

## Technologies Used

- Python  
- pandas  
- numpy  
- matplotlib  
- scikit-learn  

## Files

gas_model.py  
Natural gas price modelling and storage contract valuation.

loan_default_model.py  
Logistic regression model for estimating probability of default and expected loss.

fico_bucket_quantization.py  
Credit rating bucket construction using FICO score quantization.

Nat_Gas.csv  
Historical natural gas price dataset.

Task 3 and 4_Loan_Data.csv  
Borrower dataset used for credit risk modelling and rating construction.
