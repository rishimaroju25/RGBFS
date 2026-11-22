from pathlib import Path
import pandas as pd
import streamlit as st

try:
    from dashboard.compat import data_file, full_width
except ImportError:  # loaded by file path, or another "dashboard" package is installed
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("rgbfs_compat", Path(__file__).resolve().parents[1] / "compat.py")
    _compat = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_compat)
    full_width = _compat.full_width
    data_file = _compat.data_file
import plotly.express as px

ROOT = Path(__file__).resolve().parents[2]

@st.cache_data(ttl=300)
def load_data(root):
    base = Path(root) / 'data' / 'processed'
    metrics = pd.read_parquet(data_file(root, 'features/clusters_market_adjusted_20d.parquet')).reset_index()
    metrics = metrics.rename(columns={metrics.columns[0]: 'date'})
    members = pd.read_parquet(data_file(root, 'features/cluster_memberships_market_adjusted_20d.parquet'))
    for frame in (metrics, members):
        frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
    mapping_path = base / 'sector_mapping.csv'
    if not mapping_path.exists():  # shared copy in the repo (scripts/build_sector_mapping.py)
        mapping_path = Path(root) / 'docs' / 'sector_mapping.csv'
    mapping = pd.read_csv(mapping_path) if mapping_path.exists() else None
    stress_path = data_file(root, 'stress_labels_daily.parquet')
    stress = pd.read_parquet(stress_path) if stress_path.exists() else None
    if stress is not None:
        stress.index = pd.to_datetime(stress.index).normalize()
    return metrics.sort_values('date'), members, mapping, stress


def render_clusters_panel(root=ROOT):
    st.header('Market clustering analysis')
    st.caption('Market-adjusted network snapshots sampled approximately every 20 trading days.')
    try:
        metrics, members, mapping, stress = load_data(str(root))
    except (OSError, ValueError, KeyError) as exc:
        st.error(f'Unable to load clustering data: {exc}')
        return
    if metrics.empty or members.empty:
        st.warning('Clustering data is empty.')
        return
    dates = metrics['date'].drop_duplicates().sort_values().tolist()
    bounds = st.date_input('Historical chart range', (dates[0].date(), dates[-1].date()), min_value=dates[0].date(), max_value=dates[-1].date(), key='cluster_range')
    if len(bounds) != 2:
        st.info('Select both dates.')
        return
    history = metrics[metrics.date.between(*map(pd.Timestamp, bounds))].copy()
    if history.empty:
        st.warning('No snapshots in this range.')
        return
    options = history.date.drop_duplicates().tolist()
    selected = st.selectbox('Community snapshot', options, index=len(options)-1, format_func=lambda d: d.strftime('%Y-%m-%d'), key='cluster_snapshot')
    row = history.loc[history.date.eq(selected)].iloc[0]
    for col, (label, key, fmt) in zip(st.columns(4), [('Communities','n_communities','.0f'),('Modularity','modularity','.4f'),('Active nodes','n_nodes','.0f'),('Density','density','.2%')]):
        col.metric(label, format(row[key], fmt) if pd.notna(row[key]) else 'N/A')
    for col, (label, key, fmt) in zip(st.columns(4), [('Largest share','largest_share','.2%'),('Stability','stability','.3f'),('Sector ARI','sector_ari','.3f'),('Edges','n_edges',',.0f')]):
        col.metric(label, format(row[key], fmt) if pd.notna(row[key]) else 'N/A')
    st.subheader('Historical trends')
    labels = {'n_communities':'Community count','modularity':'Modularity','stability':'Community stability','largest_share':'Largest community share','sector_ari':'Sector alignment (ARI)','density':'Network density','n_nodes':'Active nodes','n_edges':'Edges','n_stocks':'Eligible stocks'}
    metric = st.selectbox('Metric', list(labels), format_func=labels.get, key='cluster_metric')
    fig = px.line(history, x='date', y=metric, markers=True, labels={metric:labels[metric], 'date':'Snapshot date'})
    if metric in ('density','largest_share'):
        fig.update_yaxes(tickformat='.0%')
    if stress is not None and 'stress' in stress:
        history['stress'] = history.date.map(stress['stress'])
        if st.checkbox('Highlight retrospective stress snapshots', value=True, key='cluster_stress'):
            flagged = history[history.stress.eq(1)]
            fig.add_scatter(x=flagged.date, y=flagged[metric], mode='markers', name='Stress snapshot', marker=dict(color='red', size=9))
        st.caption('Stress labels are retrospective; missing labels are unknown, not normal.')
    full_width(st.plotly_chart, fig)
    st.caption('Community IDs are snapshot-specific. Stability compares partitions across dates; ARI measures agreement with sector labels.')
    snapshot = members.loc[members.date.eq(selected)].copy()
    if snapshot.empty:
        st.warning('No membership records for this snapshot.')
        return
    snapshot['community'] = snapshot.community.astype(str)
    sizes = snapshot.groupby('community').ticker.nunique().reset_index(name='stocks').sort_values('stocks', ascending=False)
    community_order = sorted(sizes.community.tolist(), key=lambda x: int(x))
    st.subheader('Community size distribution')
    bar = px.bar(sizes, x='community', y='stocks', color='community',
                 category_orders={'community': community_order},
                 labels={'stocks':'Stocks','community':'Community'})
    bar.update_xaxes(type='category', categoryorder='array',
                     categoryarray=community_order, tickmode='array',
                     tickvals=community_order, ticktext=community_order)
    full_width(st.plotly_chart, bar)
    if mapping is not None and {'ticker','sector'}.issubset(mapping.columns):
        snapshot = snapshot.merge(mapping[['ticker','sector'] + (['name'] if 'name' in mapping else [])].drop_duplicates('ticker'), on='ticker', how='left')
        coverage = snapshot.sector.notna().mean()
        snapshot['sector'] = snapshot.sector.fillna('Unmatched').str.replace('_', ' ', regex=False).str.title()
        if 'name' in snapshot:
            snapshot['name'] = snapshot['name'].fillna('Unknown')
        st.subheader('Sector composition')
        st.caption(f'Sector coverage: {coverage:.1%}. Mapping is a limited historical reference, not verified point-in-time classifications. Unmatched stocks are retained.')
        counts = pd.crosstab(snapshot.community, snapshot.sector).reindex(community_order, fill_value=0)
        mode = st.radio('Heatmap values', ['Stock count','Within-community percentage'], horizontal=True, key='cluster_heatmap_mode')
        values = counts if mode == 'Stock count' else counts.div(counts.sum(axis=1), axis=0)*100
        heatmap = px.imshow(values.to_numpy(), x=values.columns.tolist(),
                            y=['Community ' + c for c in values.index],
                            aspect='auto', text_auto='.0f' if mode == 'Stock count' else '.1f',
                            labels={'x':'Sector','y':'Community','color':mode},
                            color_continuous_scale='Blues')
        heatmap.update_yaxes(type='category', autorange='reversed',
                             tickmode='array', tickvals=['Community ' + c for c in values.index])
        heatmap.update_layout(height=max(450, 32 * len(values) + 180))
        if mode != 'Stock count':
            heatmap.update_traces(texttemplate='%{z:.1f}%')
            heatmap.update_coloraxes(colorbar_ticksuffix='%')
        full_width(st.plotly_chart, heatmap)
    else:
        st.info('Optional sector mapping unavailable; sector composition omitted.')
    st.subheader('Stock and community explorer')
    community = st.selectbox('Community filter', ['All'] + community_order, key='cluster_group')
    query = st.text_input('Search stock ticker', key='cluster_ticker').strip().upper()
    visible = snapshot if community == 'All' else snapshot[snapshot.community.eq(community)]
    if query:
        visible = visible[visible.ticker.astype(str).str.upper().str.contains(query, regex=False)]
    visible = visible.copy()
    visible['date'] = visible['date'].dt.strftime('%Y-%m-%d')
    full_width(st.dataframe, visible, hide_index=True)
    st.download_button('Download filtered membership CSV', visible.to_csv(index=False), 'cluster_memberships.csv', 'text/csv', key='cluster_download_members')
    with st.expander('Historical metrics table'):
        history_display = history.copy()
        history_display['date'] = history_display['date'].dt.strftime('%Y-%m-%d')
        full_width(st.dataframe, history_display, hide_index=True)
        st.download_button('Download historical metrics CSV', history.to_csv(index=False), 'cluster_metrics.csv', 'text/csv', key='cluster_download_metrics')
