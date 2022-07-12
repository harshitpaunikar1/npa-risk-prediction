# NPA Risk Prediction Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## Model development lifecycle

```mermaid
flowchart TD
    N1["Step 1\nRan discovery with risk, underwriting, and collections teams to align default defi"]
    N2["Step 2\nUnified datasets: bureau history, delinquencies, income/obligations, application a"]
    N1 --> N2
    N3["Step 3\nEngineered features (DTI, utilisation, enquiry velocity, payment consistency, vint"]
    N2 --> N3
    N4["Step 4\nAddressed class imbalance with stratified folds and cost-sensitive learning; valid"]
    N3 --> N4
    N5["Step 5\nConverted probabilities into scorecard with risk tiers, cut-offs, policy overlays "]
    N4 --> N5
```

## ROC/AUC and KS curves

```mermaid
flowchart LR
    N1["Inputs\nImages or camera frames entering the inference workflow"]
    N2["Decision Layer\nROC/AUC and KS curves"]
    N1 --> N2
    N3["User Surface\nAPI-facing integration surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nOperating cost per workflow"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
