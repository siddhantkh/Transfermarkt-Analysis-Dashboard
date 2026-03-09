import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("players.csv")
    df['date_of_birth'] = pd.to_datetime(df['date_of_birth'], errors='coerce')
    df['Age'] = datetime.now().year - df['date_of_birth'].dt.year
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
positions = st.sidebar.multiselect("Select Position", options=df['position'].unique(), default=df['position'].unique())
# clubs = st.sidebar.multiselect("Select Club", options=df['current_club_name'].unique(), default=df['current_club_name'].unique())
countries = st.sidebar.multiselect("Select Nationality", options=df['country_of_citizenship'].dropna().unique(), default=df['country_of_citizenship'].dropna().unique())

# Filtered Data
filtered_df = df[
    (df['position'].isin(positions)) &
    # (df['current_club_name'].isin(clubs)) &
    (df['country_of_citizenship'].isin(countries))
]

# Tabs
main_tab, cluster_tab, club_tab = st.tabs(["Dashboard", "Player Clustering", "Club vs Club"])

with main_tab:
    # Title
    st.title("⚽ Football Players Dashboard")
    st.markdown("Interactive dashboard showing player data with Plotly and Streamlit.")

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Players", len(filtered_df))
    col2.metric("Avg Market Value (€)", f"{filtered_df['market_value_in_eur'].mean():,.0f}")
    col3.metric("Avg Age", f"{filtered_df['Age'].mean():.1f}")

    # Market Value Distribution
    left, right = st.columns(2)
    with left:
        st.subheader("Market Value Distribution")
        fig1 = px.histogram(filtered_df, x="market_value_in_eur", nbins=50, title= "Histogram")
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        st.subheader("Market Value by Position")
        fig2 = px.box(filtered_df, x="position", y="market_value_in_eur", points="all", title="Box Plot")
        st.plotly_chart(fig2, use_container_width=True)

    # Age vs Market Value
    st.subheader("Age vs Market Value")
    fig3 = px.scatter(filtered_df, x="Age", y="market_value_in_eur", color="position", title="Scatter Plot")
    st.plotly_chart(fig3, use_container_width=True)

    # Foot Preference and Grouped Bar
    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Foot Preference Distribution")
        fig4 = px.pie(filtered_df, names="foot", title="Pie Chart")
        st.plotly_chart(fig4, use_container_width=True)

    with right2:
        st.subheader("Avg Market Value by Position and Foot")
        grouped = filtered_df.groupby(['position', 'foot'])['market_value_in_eur'].mean().reset_index()
        fig6 = px.bar(grouped, x="position", y="market_value_in_eur", color="foot", barmode="group", title= "Grouped Bar Chart")
        st.plotly_chart(fig6, use_container_width=True)

    # Age Distribution
    st.subheader("Age Distribution of Players")
    fig7 = px.histogram(filtered_df, x="Age", nbins=40, title="Histogram")
    st.plotly_chart(fig7, use_container_width=True)

    # Most Valuable Defender
    st.subheader("Most Valuable Defenders")
    defenders = filtered_df[df['position'] == "Defender"].sort_values(by="market_value_in_eur", ascending=False).head(10)
    st.table(defenders[['name', 'position', 'market_value_in_eur', 'current_club_name']])

    # Most Valuable Goalkeepers
    st.subheader("Most Valuable Goalkeepers")
    keepers = filtered_df[df['position'] == "Goalkeeper"].sort_values(by="market_value_in_eur", ascending=False).head(10)
    st.table(keepers[['name', 'position', 'market_value_in_eur', 'current_club_name']])

    # Most Valuable XI (Based on Sub-Position with 2 CBs)
    st.subheader("Most Valuable XI")
    xi_positions = [
        ('Goalkeeper', 1),
        ('Right-Back', 1),
        ('Left-Back', 1),
        ('Centre-Back', 2),
        ('Defensive Midfield', 1),
        ('Central Midfield', 1),
        ('Attacking Midfield', 1),
        ('Left Winger', 1),
        ('Right Winger', 1),
        ('Centre-Forward', 1),
    ]

    xi_players = pd.DataFrame()
    for sub_pos, count in xi_positions:
        selected = filtered_df[df['sub_position'] == sub_pos].sort_values(by='market_value_in_eur', ascending=False).head(count)
        xi_players = pd.concat([xi_players, selected])

    fig8 = px.bar(xi_players, x="name", y="market_value_in_eur", color="sub_position", title="Top 11 Most Valuable Players by Sub-Position (Bar Chart)")
    st.plotly_chart(fig8, use_container_width=True)

    # Most Valuable Clubs
    st.subheader("Top 10 Most Valuable Clubs")
    club_value = filtered_df.groupby('current_club_name')['market_value_in_eur'].sum().reset_index().sort_values(by='market_value_in_eur', ascending=False).head(10)
    st.table(club_value)

    # Table Preview
    st.subheader("Top Players by Market Value")
    st.dataframe(filtered_df[['name', 'position', 'Age', 'market_value_in_eur', 'current_club_name']].sort_values(by="market_value_in_eur", ascending=False).head(20))

with cluster_tab:
    st.title("🔍 Player Clustering")
    st.markdown("Visualize players using different clustering methods")

    features = ['Age', 'height_in_cm', 'market_value_in_eur']
    cluster_data = df[features].dropna()
    scaler = StandardScaler()
    cluster_scaled = scaler.fit_transform(cluster_data)

    clustering_method = st.selectbox("Select Clustering Method", ["KMeans", "DBSCAN"])

    if clustering_method == "KMeans":
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        cluster_data['Cluster'] = kmeans.fit_predict(cluster_scaled)
    elif clustering_method == "DBSCAN":
        eps_val = st.slider("DBSCAN Epsilon (eps)", 0.1, 5.0, 1.0)
        min_samples_val = st.slider("Minimum Samples", 2, 10, 5)
        db = DBSCAN(eps=eps_val, min_samples=min_samples_val)
        cluster_data['Cluster'] = db.fit_predict(cluster_scaled)

    fig_cluster = px.scatter(
        cluster_data,
        x='Age',
        y='market_value_in_eur',
        color=cluster_data['Cluster'].astype(str),
        height = 700,
        title=f"Player Clustering by {clustering_method} (Scatter Plot)"
    )
    st.plotly_chart(fig_cluster, use_container_width=True)
    
with club_tab:
    st.title("🏟️ Club vs Club Comparison")
    club_list = df['current_club_name'].dropna().unique()
    selected_clubs = st.multiselect("Select Clubs to Compare", club_list, default=list(club_list[:2]))

    if len(selected_clubs) >= 2:
        st.title("Comparison by Market Value Distribution")
        compare_df = df[df['current_club_name'].isin(selected_clubs)]
        fig_comp = px.box(compare_df, x='current_club_name', y='market_value_in_eur', color='current_club_name', points='all', title="Box Plot", height=700)
        fig_comp.update_layout(yaxis=dict(dtick=20000000)) 
        st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("Avg Market Value and Age of the Selected Clubs")
        agg = compare_df.groupby('current_club_name').agg({'market_value_in_eur': 'mean', 'Age': 'mean'}).reset_index()
        st.dataframe(agg.rename(columns={'market_value_in_eur': 'Avg Market Value (€)', 'Age': 'Avg Age'}))

        st.subheader("Best Player in Each Selected Club")
        top_players = compare_df.sort_values(by='market_value_in_eur', ascending=False).groupby('current_club_name').head(1)
        st.table(top_players[['current_club_name', 'name', 'position', 'market_value_in_eur', 'Age']].reset_index(drop=True))

    else:
        st.warning("Please select at least two clubs for comparison.")