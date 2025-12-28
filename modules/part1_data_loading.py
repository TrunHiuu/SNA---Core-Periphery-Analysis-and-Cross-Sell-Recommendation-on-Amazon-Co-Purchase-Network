"""
PART 1: DATA LOADING & PREPROCESSING
Mục đích:
- Nạp dữ liệu đồ thị và cộng đồng
- Chọn ra Top N danh mục sản phẩm (Category) đầu tiên (đủ lớn) để phân tích chuyên sâu
- Áp dụng thuật toán Random Walk Sampling (RWS) để lấy mẫu đồ thị con
"""

import os
import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CÁC HÀM ĐỌC DỮ LIỆU
# ==========================================

def read_ungraph(path):
    """Đọc đồ thị từ file cạnh"""
    G = nx.Graph()
    if not os.path.exists(path):
        st.warning("Không tìm thấy file cạnh. Tạo đồ thị mẫu để demo.")
        G.add_edges_from([(i, i+1) for i in range(500)] + [(0, 50), (0, 10), (10, 20), (5, 150)])
        return G
    with open(path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            try:
                a, b = line.strip().split()
                G.add_edge(int(a), int(b))
            except:
                continue
    return G


def read_communities(path):
    """Đọc các cộng đồng từ file"""
    communities = []
    if not os.path.exists(path):
        st.warning("Không tìm thấy file cộng đồng.")
        return []
    with open(path, "rt") as f:
        for line in f:
            nodes = [int(x) for x in line.strip().split() if x]
            if nodes:
                communities.append(nodes)
    return communities


@st.cache_resource
def load_data(data_dir="./data/"):
    """Nạp dữ liệu đồ thị và cộng đồng"""
    edge_path = os.path.join(data_dir, "com-amazon.ungraph.txt")
    cmty_path = os.path.join(data_dir, "com-amazon.all.dedup.cmty.txt")
    
    G_full = read_ungraph(edge_path)
    G_full.remove_edges_from(nx.selfloop_edges(G_full))
    communities = read_communities(cmty_path)
    
    return G_full, communities


@st.cache_data
def analyze_categories(_G_full, communities, n_categories: int):
    """Phân tích Top N danh mục đầu tiên (theo đúng thứ tự file).

    - Lọc các community có kích thước > 1000.
    - Lấy tối đa N community ĐẦU TIÊN trong danh sách `valid_communities`.
    - Sau đó mới sắp xếp theo số node để hiển thị bảng/charts.

    `Cat_ID` là index gốc trong `valid_communities`, nên dùng được với
    `get_category_graph` (giữ đúng behaviour ban đầu).
    """

    cat_stats = []
    valid_communities = [c for c in communities if len(c) > 1000]

    # Chỉ phân tích tối đa N community đầu tiên
    for i, nodes in enumerate(valid_communities[:n_categories]):
        subg = _G_full.subgraph(nodes)
        n = subg.number_of_nodes()
        e = subg.number_of_edges()
        density = nx.density(subg)
        max_deg = max([d for n, d in subg.degree()]) if n > 0 else 0

        cat_stats.append({
            'Cat_ID': i,
            'Nodes': n,
            'Edges': e,
            'Density': density,
            'Max_Degree': max_deg
        })

    df_cat = pd.DataFrame(cat_stats).sort_values('Nodes', ascending=False)
    return df_cat, valid_communities


@st.cache_resource
def get_category_graph(_G_full, communities, cat_idx=0):
    """Lấy đồ thị của danh mục được chọn"""
    valid_communities = [c for c in communities if len(c) > 1000]
    if cat_idx < len(valid_communities):
        selected_nodes = valid_communities[cat_idx]
        G_category = _G_full.subgraph(selected_nodes).copy()
        return G_category
    return _G_full.copy()


# ==========================================
# HÀM RENDER STREAMLIT
# ==========================================

def render_part1(G_full, communities):
    """Render giao diện Part 1: Data Loading & Preprocessing"""
    
    # ==================== CUSTOM CSS ====================
    st.markdown("""
    <style>
    .part-header {
        background: linear-gradient(135deg, #232F3E 0%, #37475A 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
    }
    .part-title {
        color: #FF9900;
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .part-desc {
        color: #FFFFFF;
        font-size: 0.95rem;
        opacity: 0.9;
    }
    .info-card {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .info-value {
        color: #FF9900;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .info-label {
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    .section-title {
        color: #232F3E;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-left: 0.8rem;
        border-left: 4px solid #FF9900;
    }
    .highlight-box {
        background: linear-gradient(90deg, #FFF8E7 0%, #FFFFFF 100%);
        border-left: 4px solid #FF9900;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .stats-card {
        background: #F7F8F8;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #E0E0E0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class="part-header">
        <div class="part-title">Data Loading & Preprocessing</div>
        <div class="part-desc">
            Nạp dữ liệu đồ thị, chọn danh mục sản phẩm và đánh giá chất lượng lấy mẫu
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== THÔNG TIN ĐỒ THỊ ====================
    st.markdown('<div class="section-title">Thông tin đồ thị đầy đủ</div>', unsafe_allow_html=True)
    
    num_nodes = G_full.number_of_nodes()
    num_edges = G_full.number_of_edges()
    num_communities = len(communities)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{num_nodes:,}</div>
            <div class="info-label">Số Node</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{num_edges:,}</div>
            <div class="info-label">Số Cạnh</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{num_communities:,}</div>
            <div class="info-label">Số Cộng đồng</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== PHÂN TÍCH DANH MỤC ====================
    st.markdown('<div class="section-title">Phân tích các danh mục đầu tiên</div>', unsafe_allow_html=True)

    # Người dùng chọn số lượng danh mục đầu tiên sẽ được phân tích
    col_top, col_empty = st.columns([1, 3])
    with col_top:
        top_n = st.selectbox(
            "Số lượng Top Categories:",
            options=[5, 10, 15, 20],
            index=1  # Mặc định top 10
        )

    # Phân tích đúng Top N cộng đồng đầu tiên (đủ lớn)
    with st.spinner("Đang phân tích danh mục..."):
        df_cat, valid_communities = analyze_categories(G_full, communities, n_categories=top_n)

    if not df_cat.empty:
        st.markdown(f"""
        <div class="highlight-box">
            Hiển thị <strong>Top {top_n} danh mục đầu tiên đủ lớn</strong> trong tổng số {len(communities):,} cộng đồng
        </div>
        """, unsafe_allow_html=True)

        # df_cat đã chỉ chứa tối đa top_n danh mục đầu tiên, chỉ cần reset index
        df_top = df_cat.reset_index(drop=True)
        
        # Hiển thị bảng với scroll nếu top_n > 10
        # Height cố định 400px cho scroll, tự động cho <= 10 hàng
        if top_n > 10:
            st.markdown("""
            <style>
            .stDataFrame > div { max-height: 400px; overflow-y: auto !important; }
            </style>
            """, unsafe_allow_html=True)
            table_height = 400
        else:
            table_height = 35 + 35 * top_n  # header + rows
        
        st.dataframe(
            df_top.style.format({
                'Nodes': '{:,.0f}',
                'Edges': '{:,.0f}',
                'Density': '{:.5f}',
                'Max_Degree': '{:.0f}'
            }),
            use_container_width=True,
            hide_index=True,
            height=table_height
        )
        
        # ==================== ANIMATED BAR CHART ====================
        # Sắp xếp theo số nodes giảm dần cho chart
        df_sorted = df_top.sort_values('Nodes', ascending=False)
        
        # Tạo vertical bar chart với Plotly
        fig_bar = go.Figure()
        
        # Thêm bar chart dọc
        fig_bar.add_trace(go.Bar(
            x=[f"Cat {i}" for i in df_sorted['Cat_ID']],
            y=df_sorted['Nodes'].tolist(),
            marker=dict(
                color=df_sorted['Nodes'].tolist(),
                colorscale=[[0, '#37475A'], [0.5, '#FF9900'], [1, '#FF6600']],
                line=dict(color='#232F3E', width=1)
            ),
            text=[f"{int(n):,}" for n in df_sorted['Nodes']],
            textposition='outside',
            textfont=dict(size=10, color='#232F3E'),
            hovertemplate='<b>Category %{x}</b><br>Nodes: %{y:,.0f}<extra></extra>'
        ))
        
        fig_bar.update_layout(
            title=dict(
                text=f'<b>Top {top_n} Categories theo số lượng Nodes</b>',
                font=dict(size=16, color='#232F3E'),
                x=0.5
            ),
            xaxis=dict(
                title=dict(text='Category', font=dict(size=12, color='#666')),
                tickfont=dict(size=10),
                tickangle=-45
            ),
            yaxis=dict(
                title=dict(text='Số Nodes', font=dict(size=12, color='#666')),
                tickfont=dict(size=10),
                gridcolor='#E8E8E8',
                showgrid=True
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=60, r=20, t=60, b=80),
            height=450,
            showlegend=False
        )
        
        fig_bar.update_traces(
            marker_line_width=1.5,
            opacity=0.9
        )
        
        # Hiển thị chart - sử dụng scrollbar khi nhiều categories
        if top_n > 10:
            # Container scrollable cho chart
            st.markdown("""
            <style>
            [data-testid="stPlotlyChart"] > div {
                overflow-x: auto !important;
                overflow-y: hidden;
            }
            </style>
            """, unsafe_allow_html=True)
            fig_bar.update_layout(
                width=max(800, top_n * 60),  # Tối thiểu 800px, mỗi bar ~60px
                height=480
            )
            st.plotly_chart(fig_bar, use_container_width=False, config={'displayModeBar': True, 'scrollZoom': True})
        else:
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
        # ==================== CHỌN DANH MỤC ĐỂ PHÂN TÍCH ====================
        st.markdown('<div class="section-title">Chọn danh mục để phân tích</div>', unsafe_allow_html=True)
        
        # cat_options đã được sắp xếp theo Nodes giảm dần (từ df_top)
        # Nên cat_options[0] là category có số node cao nhất
        cat_options = df_top['Cat_ID'].tolist()
        cat_nodes_dict = dict(zip(df_top['Cat_ID'], df_top['Nodes']))
        
        # Khởi tạo session_state nếu chưa có hoặc giá trị không hợp lệ
        if 'selected_cat_idx' not in st.session_state or st.session_state['selected_cat_idx'] not in cat_options:
            # Mặc định chọn category có nodes cao nhất (đầu tiên trong list đã sort)
            st.session_state['selected_cat_idx'] = cat_options[0]
        
        # Tìm index của category đã chọn trong list
        current_selection = st.session_state['selected_cat_idx']
        default_index = cat_options.index(current_selection) if current_selection in cat_options else 0
        
        
        # Callback khi dropdown thay đổi
        def on_category_change():
            selected = st.session_state.part1_category_selector
            st.session_state['selected_cat_idx'] = selected
        
        selected_idx = st.selectbox(
            "Chọn danh mục:",
            options=cat_options,
            index=default_index,
            format_func=lambda x: f"Danh mục {x} ({cat_nodes_dict.get(x, 0):,.0f} nodes)",
            label_visibility="collapsed",
            key="part1_category_selector",
            on_change=on_category_change
        )
        
        # Đảm bảo session_state được cập nhật (cho lần chạy đầu tiên)
        st.session_state['selected_cat_idx'] = selected_idx
        
        # Lấy đồ thị của category đã chọn
        with st.spinner("Đang tải đồ thị danh mục..."):
            G_category = get_category_graph(G_full, communities, selected_idx)
        
        # Lưu G_category vào session_state
        st.session_state['G_category'] = G_category
        
        st.success(f"Đã chọn danh mục {selected_idx}: {G_category.number_of_nodes():,} node, {G_category.number_of_edges():,} cạnh")
        st.info(f"Đồ thị này sẽ được sử dụng để phân tích ở các Part tiếp theo")
        
        # ==================== ĐÁNH GIÁ CHẤT LƯỢNG ====================
        st.markdown('<div class="section-title">Đánh giá chất lượng lấy mẫu</div>', unsafe_allow_html=True)
        
        with st.spinner("Đang vẽ biểu đồ phân phối bậc..."):
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Phân phối bậc gốc
            deg_orig = [d for n, d in G_full.degree()]
            axes[0].hist(deg_orig, bins=50, density=True, alpha=0.7, color='#232F3E', edgecolor='white')
            axes[0].set_title("Phân phối bậc - Đồ thị gốc", fontweight='bold', color='#232F3E')
            axes[0].set_yscale('log')
            axes[0].set_xlabel("Bậc", fontsize=10)
            axes[0].set_ylabel("Mật độ (log)", fontsize=10)
            axes[0].spines['top'].set_visible(False)
            axes[0].spines['right'].set_visible(False)
            
            # Phân phối bậc đồ thị con
            deg_samp = [d for n, d in G_category.degree()]
            axes[1].hist(deg_samp, bins=50, density=True, alpha=0.7, color='#FF9900', edgecolor='white')
            axes[1].set_title("Phân phối bậc - Đồ thị con", fontweight='bold', color='#232F3E')
            axes[1].set_yscale('log')
            axes[1].set_xlabel("Bậc", fontsize=10)
            axes[1].set_ylabel("Mật độ (log)", fontsize=10)
            axes[1].spines['top'].set_visible(False)
            axes[1].spines['right'].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # So sánh thống kê
        st.markdown('<div class="section-title">So sánh thống kê</div>', unsafe_allow_html=True)
        
        with st.spinner("Đang tính toán thống kê..."):
            avg_deg_orig = np.mean(deg_orig)
            avg_clust_orig = nx.average_clustering(G_full)
            avg_deg_samp = np.mean(deg_samp)
            avg_clust_samp = nx.average_clustering(G_category)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="stats-card">
                    <div style="font-weight:600; color:#232F3E; margin-bottom:1rem; font-size:1.1rem;">Đồ thị gốc</div>
                    <div style="margin-bottom:0.8rem;">
                        <div style="color:#666; font-size:0.85rem;">Bậc trung bình</div>
                        <div class="info-value" style="font-size:1.5rem;">{avg_deg_orig:.4f}</div>
                    </div>
                    <div>
                        <div style="color:#666; font-size:0.85rem;">Hệ số phân cụm</div>
                        <div class="info-value" style="font-size:1.5rem;">{avg_clust_orig:.4f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stats-card">
                    <div style="font-weight:600; color:#FF9900; margin-bottom:1rem; font-size:1.1rem;">Đồ thị con (đã chọn)</div>
                    <div style="margin-bottom:0.8rem;">
                        <div style="color:#666; font-size:0.85rem;">Bậc trung bình</div>
                        <div class="info-value" style="font-size:1.5rem;">{avg_deg_samp:.4f}</div>
                    </div>
                    <div>
                        <div style="color:#666; font-size:0.85rem;">Hệ số phân cụm</div>
                        <div class="info-value" style="font-size:1.5rem;">{avg_clust_samp:.4f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("Không tìm thấy danh mục đủ lớn để phân tích.")
    
    return st.session_state.get('G_category', G_full)
