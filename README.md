# NPA Risk Prediction

> **Domain:** Banking

## Overview

Retail lenders face rising non-performing assets and pressure to grow responsibly. Existing underwriting relies on static rules and manual checks, missing early warning signs in credit histories and repayment behaviour. Decisions use lagging indicators, allowing high-risk borrowers to pass screening while creditworthy applicants get declined. Fragmented data across bureau files, applications, bank statements, and collections limits comprehensive risk views. Without predictive default probability at origination and across the loan lifecycle, institutions bear higher credit losses, larger capital provisions, and expensive collections. Customer experience suffers through inconsistent pricing and slow turnaround times, limiting disbursement growth.

## Approach

- Ran discovery with risk, underwriting, and collections teams to align default definitions, target horizon (e.g., 90+ DPD in 12 months), compliant variable use
- Unified datasets: bureau history, delinquencies, income/obligations, application attributes, bank-statement features into a governed feature store
- Engineered features (DTI, utilisation, enquiry velocity, payment consistency, vintage risk); benchmarked logistic regression and tree-based ensembles
- Addressed class imbalance with stratified folds and cost-sensitive learning; validated via AUC, KS, lift charts and back-testing on vintages
- Converted probabilities into scorecard with risk tiers, cut-offs, policy overlays for approve/decline, limit, and risk-based pricing
- Deployed scoring API; established monitoring (PSI, drift, calibration), weekly refresh, governance for periodic re-training

## Skills & Technologies

- Credit Risk Modelling
- Feature Engineering
- Logistic Regression
- Gradient Boosting
- Imbalanced Learning
- Cross-Validation
- Scorecard Development
- Model Monitoring
