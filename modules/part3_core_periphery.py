"""
PART 3: CORE-PERIPHERY STRUCTURE ANALYSIS
Mục đích:
- K-core Decomposition: Thực hiện thuật toán phân rã K-core
- Node Classification: Phân loại sản phẩm thành Core Zone, Inner Periphery, Outer Periphery
- Visualization: Trực quan hóa đồ thị với màu sắc phân biệt
- Density Matrix: Tính toán ma trận mật độ giữa các nhóm
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter


@st.cache_data
def compute_core_periphery(_G):
    """Tính toán K-core decomposition và phân loại node"""
    core_nums = nx.core_number(_G)
    max_k = max(core_nums.values())
    
    node_classes = {}
    for n, k in core_nums.items():
        if k == max_k:
            node_classes[n] = 'Core Zone'
        elif k >= max_k * 0.7:
            node_classes[n] = 'Inner Periphery'
        else:
            node_classes[n] = 'Outer Periphery'
    
    return core_nums, max_k, node_classes


@st.cache_data
def compute_density_matrix(_G, node_classes):
    """Tính toán ma trận mật độ giữa các nhóm"""
    class_labels = ['Core Zone', 'Inner Periphery', 'Outer Periphery']
    node_counts_per_class = Counter(node_classes.values())
    
    actual_edges_matrix = pd.DataFrame(0, index=class_labels, columns=class_labels, dtype=int)
    potential_edges_matrix = pd.DataFrame(0, index=class_labels, columns=class_labels, dtype=int)
    
    # Đếm số cạnh thực tế
    for u, v in _G.edges():
        class_u = node_classes.get(u, 'Unknown')
        class_v = node_classes.get(v, 'Unknown')
        
        if class_u in class_labels and class_v in class_labels:
            actual_edges_matrix.loc[class_u, class_v] += 1
            if class_u != class_v:
                actual_edges_matrix.loc[class_v, class_u] += 1
    
    # Tính số cạnh tiềm năng
    for i in class_labels:
        for j in class_labels:
            count_i = node_counts_per_class.get(i, 0)
            count_j = node_counts_per_class.get(j, 0)
            
            if i == j:
                potential_edges_matrix.loc[i, j] = count_i * (count_i - 1) // 2
            else:
                potential_edges_matrix.loc[i, j] = count_i * count_j
    
    # Ma trận mật độ
    density_matrix = actual_edges_matrix / potential_edges_matrix
    density_matrix = density_matrix.fillna(0)
    
    return density_matrix, actual_edges_matrix, node_counts_per_class


def render_part3(G):
    """Render giao diện Part 3: Core-Periphery Structure Analysis"""
    
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
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
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
    .interpretation-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #FFFFFF 100%);
        border-left: 4px solid #00b894;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .core-zone { color: #d62728; font-weight: 600; }
    .inner-periphery { color: #ff7f0e; font-weight: 600; }
    .outer-periphery { color: #1f77b4; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class="part-header">
        <div class="part-title">Core-Periphery Structure Analysis</div>
        <div class="part-desc">
            Phân tích cấu trúc Lõi-Ngoại vi của mạng lưới sử dụng thuật toán K-core Decomposition
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== THÔNG TIN ĐỒ THỊ ĐANG PHÂN TÍCH ====================
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    selected_cat = st.session_state.get('selected_cat_idx', None)
    
    if selected_cat is not None:
        st.success(f"Đang phân tích: Subgraph của Danh mục {selected_cat} ({num_nodes:,} nodes, {num_edges:,} cạnh)")
    else:
        st.info(f"Đang phân tích: Đồ thị gốc ({num_nodes:,} nodes, {num_edges:,} cạnh). Vào Part 1 để chọn danh mục cụ thể.")
    
    # ==================== MỤC ĐÍCH ====================
    st.markdown("""
    <div class="highlight-box">
        <strong>Mục đích:</strong> Phân tích cấu trúc Lõi-Ngoại vi để xác định các sản phẩm trung tâm (Anchor Items) 
        và sản phẩm ngách trong mạng lưới đồng mua hàng Amazon
    </div>
    """, unsafe_allow_html=True)
    
    # Tính toán K-core
    with st.spinner("Đang phân tích cấu trúc K-core..."):
        core_nums, max_k, node_classes = compute_core_periphery(G)
    
    # Lưu vào session state
    st.session_state['node_classes'] = node_classes
    st.session_state['core_nums'] = core_nums
    st.session_state['max_k'] = max_k
    
    # Thống kê phân bố
    counts = Counter(node_classes.values())
    
    # ==================== METRICS CHÍNH ====================
    st.markdown('<div class="section-title">Phân bố cấu trúc Core-Periphery</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{max_k}</div>
            <div class="info-label">Max K-Core</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value" style="color: #d62728;">{counts.get('Core Zone', 0):,}</div>
            <div class="info-label">Core Zone</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value" style="color: #ff7f0e;">{counts.get('Inner Periphery', 0):,}</div>
            <div class="info-label">Inner Periphery</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value" style="color: #1f77b4;">{counts.get('Outer Periphery', 0):,}</div>
            <div class="info-label">Outer Periphery</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== GIẢI THÍCH PHÂN LOẠI ====================
    st.markdown(f"""
    <div class="interpretation-box">
        <strong>Tiêu chí phân loại (K-max = {max_k}):</strong><br>
        • <span class="core-zone">Core Zone:</span> Các node có K-core = {max_k} (Sản phẩm Anchor - trung tâm mạng lưới)<br>
        • <span class="inner-periphery">Inner Periphery:</span> Các node có K-core ≥ {int(max_k * 0.7)} (Vùng đệm - kết nối Core với Outer)<br>
        • <span class="outer-periphery">Outer Periphery:</span> Các node có K-core < {int(max_k * 0.7)} (Sản phẩm ngách - rìa mạng lưới)
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== TABS CHI TIẾT ====================
    st.markdown('<div class="section-title">Phân tích chi tiết</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "Trực quan hóa mạng", 
        "Ma trận mật độ",
        "Danh sách node"
    ])
    
    # ==================== TAB 1: TRỰC QUAN HÓA ====================
    with tab1:
        st.markdown('<div class="section-title">Trực quan hóa cấu trúc Lõi-Ngoại vi</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Biểu đồ tròn phân bố
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            colors = {'Core Zone': '#d62728', 'Inner Periphery': '#ff7f0e', 'Outer Periphery': '#1f77b4'}
            labels = list(counts.keys())
            sizes = list(counts.values())
            pie_colors = [colors.get(l, 'gray') for l in labels]
            ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=90)
            ax_pie.set_title("Tỷ lệ các nhóm", fontweight='bold', color='#232F3E')
            st.pyplot(fig_pie)
        
        with col1:
            st.markdown("""
            <div class="highlight-box">
                <strong>Network Visualization (Sample):</strong> Hiển thị mẫu mạng lưới với màu sắc phân biệt theo nhóm
            </div>
            """, unsafe_allow_html=True)
            
            # Chọn node để vẽ
            nodes_to_draw = [n for n, c in node_classes.items() if c != 'Outer Periphery']
            outer_sample = [n for n, c in node_classes.items() if c == 'Outer Periphery']
            
            sample_size = st.slider("Số node Outer Periphery để hiển thị:", 50, 500, 200)
            
            if len(outer_sample) > sample_size:
                nodes_to_draw += random.sample(outer_sample, sample_size)
            else:
                nodes_to_draw += outer_sample
            
            H_draw = G.subgraph(nodes_to_draw)
            
            with st.spinner("Đang vẽ đồ thị..."):
                fig_net, ax_net = plt.subplots(figsize=(12, 10))
                pos = nx.spring_layout(H_draw, k=0.15, seed=42)
                
                color_map = {'Core Zone': '#d62728', 'Inner Periphery': '#ff7f0e', 'Outer Periphery': '#1f77b4'}
                
                for label in ['Outer Periphery', 'Inner Periphery', 'Core Zone']:
                    nodelist = [n for n in H_draw.nodes() if node_classes.get(n) == label]
                    if not nodelist:
                        continue
                    sizes = [min(G.degree(n) * 5, 300) for n in nodelist]
                    nx.draw_networkx_nodes(H_draw, pos, nodelist=nodelist, 
                                          node_color=color_map[label],
                                          node_size=sizes, label=label, alpha=0.8, ax=ax_net)
                
                nx.draw_networkx_edges(H_draw, pos, alpha=0.1, edge_color='gray', ax=ax_net)
                ax_net.set_title(f"Trực quan hóa cấu trúc Lõi-Ngoại vi (K-max = {max_k})", fontweight='bold', fontsize=14, color='#232F3E')
                ax_net.legend(loc='upper left')
                ax_net.axis('off')
                st.pyplot(fig_net)
        
        # Nhận xét
        st.markdown(f"""
        <div class="interpretation-box">
            <strong>Nhận xét:</strong><br>
            • <span class="core-zone">Core Zone ({counts.get('Core Zone', 0):,} nodes):</span> Các sản phẩm "Anchor" - được mua kèm nhiều nhất, là trung tâm của các chiến dịch cross-sell<br>
            • <span class="inner-periphery">Inner Periphery ({counts.get('Inner Periphery', 0):,} nodes):</span> Vùng đệm, kết nối sản phẩm Core với các sản phẩm ngách<br>
            • <span class="outer-periphery">Outer Periphery ({counts.get('Outer Periphery', 0):,} nodes):</span> Sản phẩm ngách, ít liên kết nhưng có tiềm năng cross-sell khi ghép với sản phẩm Core
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 2: MA TRẬN MẬT ĐỘ ====================
    with tab2:
        st.markdown('<div class="section-title">Ma trận mật độ (Block Modeling)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Định nghĩa:</strong> Ma trận mật độ cho thấy xác suất kết nối giữa các nhóm node. 
            Giá trị cao = kết nối dày đặc, giá trị thấp = kết nối thưa thớt.
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("Đang tính toán ma trận mật độ..."):
            density_matrix, actual_edges, node_counts = compute_density_matrix(G, node_classes)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Số cạnh thực tế giữa các nhóm:**")
            st.dataframe(actual_edges, use_container_width=True)
            
            st.markdown("**Số node trong mỗi nhóm:**")
            st.dataframe(pd.DataFrame.from_dict(dict(node_counts), orient='index', columns=['Số node']), use_container_width=True)
        
        with col2:
            st.markdown("**Ma trận mật độ:**")
            fig_hm, ax_hm = plt.subplots(figsize=(8, 7))
            sns.heatmap(
                density_matrix,
                annot=True,
                cmap='YlOrRd',
                fmt=".4f",
                cbar_kws={'label': 'Mật độ kết nối'},
                linewidths=.5,
                linecolor='lightgray',
                ax=ax_hm
            )
            ax_hm.set_title('Ma trận mật độ Core-Periphery', fontweight='bold', fontsize=13, color='#232F3E')
            ax_hm.set_xlabel('Nhóm (Kết nối đến)', fontsize=11)
            ax_hm.set_ylabel('Nhóm (Kết nối từ)', fontsize=11)
            st.pyplot(fig_hm)
        
        # Nhận xét
        core_density = density_matrix.loc['Core Zone', 'Core Zone'] if 'Core Zone' in density_matrix.index else 0
        outer_density = density_matrix.loc['Outer Periphery', 'Outer Periphery'] if 'Outer Periphery' in density_matrix.index else 0
        
        st.markdown(f"""
        <div class="interpretation-box">
            <strong>Nhận xét:</strong><br>
            • <strong>Core-Core:</strong> Mật độ = {core_density:.4f} - Các sản phẩm Core kết nối chặt chẽ với nhau<br>
            • <strong>Outer-Outer:</strong> Mật độ = {outer_density:.4f} - Sản phẩm ngách ít kết nối với nhau<br>
            • <strong>Ý nghĩa Cross-sell:</strong> Gợi ý sản phẩm Core cho khách mua sản phẩm Outer sẽ có xác suất chuyển đổi cao hơn
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 3: DANH SÁCH NODE ====================
    with tab3:
        st.markdown('<div class="section-title">Danh sách node theo phân loại</div>', unsafe_allow_html=True)
        
        class_filter = st.selectbox("Chọn nhóm để xem:", ['Core Zone', 'Inner Periphery', 'Outer Periphery'])
        
        filtered_nodes = [n for n, c in node_classes.items() if c == class_filter]
        node_data = []
        for n in filtered_nodes[:100]:  # Giới hạn 100 node
            node_data.append({
                'Node ID': n,
                'K-Core': core_nums[n],
                'Degree': G.degree(n)
            })
        
        df_nodes = pd.DataFrame(node_data).sort_values('Degree', ascending=False)
        st.dataframe(df_nodes, use_container_width=True, hide_index=True)
        
        st.caption(f"Hiển thị {len(df_nodes)} / {len(filtered_nodes)} node trong nhóm {class_filter}")
        
        # Nhận xét theo nhóm được chọn
        if class_filter == 'Core Zone':
            st.markdown("""
            <div class="interpretation-box">
                <strong>Core Zone - Sản phẩm Anchor:</strong><br>
                • Đây là các sản phẩm có số liên kết cao nhất trong mạng lưới<br>
                • Thường là sản phẩm best-seller hoặc sản phẩm chính của danh mục<br>
                • <strong>Chiến lược:</strong> Đặt làm sản phẩm chính trong bundle, hiển thị đầu tiên trong gợi ý
            </div>
            """, unsafe_allow_html=True)
        elif class_filter == 'Inner Periphery':
            st.markdown("""
            <div class="interpretation-box">
                <strong>Inner Periphery - Vùng đệm:</strong><br>
                • Kết nối giữa sản phẩm Core và sản phẩm ngách<br>
                • Có tiềm năng trở thành sản phẩm Core nếu được đẩy mạnh marketing<br>
                • <strong>Chiến lược:</strong> Cross-sell cùng sản phẩm Core để tăng độ phủ
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="interpretation-box">
                <strong>Outer Periphery - Sản phẩm ngách:</strong><br>
                • Ít liên kết, thường là sản phẩm phụ kiện hoặc sản phẩm đặc thù<br>
                • Khách hàng mua sản phẩm này có nhu cầu cụ thể<br>
                • <strong>Chiến lược:</strong> Gợi ý sản phẩm Core phù hợp để tăng giá trị đơn hàng
            </div>
            """, unsafe_allow_html=True)
    
    return node_classes, core_nums, max_k
