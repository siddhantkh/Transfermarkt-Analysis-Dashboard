# Football Players Dashboard

Interactive dashboard for exploratory analysis of football player data from Transfermarkt.

## Overview

This Streamlit-based dashboard provides visualization and analysis of 18,192+ professional football players, focusing on market value, age, club association, and clustering insights.

## Features

- **Player Metrics**: Market value, age distribution, position analysis
- **Interactive Filters**: Filter by position and nationality
- **Club Comparison**: Compare multiple clubs side-by-side
- **Player Clustering**: KMeans and DBSCAN clustering analysis
- **Visual Analytics**: Histograms, box plots, scatter plots, pie charts, and bar charts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run dashboard.py
```

Ensure `players.csv` is in the same directory as `dashboard.py`.

## Dataset

- **Source**: Transfermarkt (via Kaggle)
- **Players**: 18,192 (after preprocessing)
- **Attributes**: Age, height, market value, position, club, nationality, etc.

## Technologies

- Streamlit
- Plotly
- Pandas
- Scikit-learn (KMeans, DBSCAN)

## Authors

Siddhant Khadka & Nikki Kayastha (Kathmandu University)
