"""
PART 6: COMPARATIVE CASE STUDY
Mục đích:
- So sánh trực tiếp hiệu quả của ML và Heuristic trên Core Node và Periphery Node
- Sử dụng Ego Network để minh họa sự khác biệt về mật độ
- Biểu đồ cột và phân tích đóng góp đặc trưng
- Đánh giá cuối cùng: ML vs Heuristic vs Ground Truth
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from sklearn.metrics import roc_curve, roc_auc_score


def get_score_details(graph, u, v):
    """Tính toán các chỉ số tương đồng giữa hai node"""
    try:
        jaccard = list(nx.jaccard_coefficient(graph, [(u, v)]))[0][2]
        aa = list(nx.adamic_adar_index(graph, [(u, v)]))[0][2]
        pa = graph.degree(u) * graph.degree(v)
        ra = list(nx.resource_allocation_index(graph, [(u, v)]))[0][2]
    except:
        jaccard, aa, pa, ra = 0, 0, 0, 0
    return jaccard, aa, pa, ra


def get_smart_recommendations(graph, anchor_node, model, scaler, popular_items, n_samples=500):
    """Gợi ý thông minh cho một node"""
    candidates = set()
    
    # Tìm ứng viên là "hàng xóm của hàng xóm"
    for nbr in graph.neighbors(anchor_node):
        candidates.update(graph.neighbors(nbr))
    
    candidates.discard(anchor_node)
    candidates.difference_update(set(graph.neighbors(anchor_node)))
    candidate_list = list(candidates)[:n_samples]
    
    if candidate_list and model is not None and scaler is not None:
        data = []
        valid_cands = []
        # Tính k-core cho toàn bộ node trong graph
        try:
            core_nums = nx.core_number(graph)
        except:
            core_nums = {n: 0 for n in graph.nodes()}
        for v in candidate_list:
            try:
                jc = list(nx.jaccard_coefficient(graph, [(anchor_node, v)]))[0][2]
                aa = list(nx.adamic_adar_index(graph, [(anchor_node, v)]))[0][2]
                pa = graph.degree(anchor_node) * graph.degree(v)
                ra = list(nx.resource_allocation_index(graph, [(anchor_node, v)]))[0][2]
                deg_diff = abs(graph.degree(anchor_node) - graph.degree(v))
                kcore_u = core_nums.get(anchor_node, 0)
                kcore_v = core_nums.get(v, 0)
                data.append([jc, aa, pa, ra, deg_diff, kcore_u, kcore_v])
                valid_cands.append(v)
            except:
                continue
        if data:
            feature_cols = ['Jaccard', 'AA', 'PA', 'RA', 'DegDiff', 'KCore_u', 'KCore_v']
            df_features = pd.DataFrame(data, columns=feature_cols)
            X_input = scaler.transform(df_features)
            ml_scores = model.predict_proba(X_input)[:, 1]
            results = df_features.copy()
            results['Candidate'] = valid_cands
            results['ML_Score'] = ml_scores
            results['Strategy'] = 'Cá nhân hóa (ML)'
            high_quality = results.sort_values('ML_Score', ascending=False).head(5)
            cols = ['Candidate', 'Strategy', 'ML_Score', 'Jaccard', 'AA', 'PA']
            return high_quality[cols]
    
    # Fallback
    fallback_recs = pd.DataFrame({
        'Candidate': popular_items[:5] if len(popular_items) >= 5 else popular_items,
        'Strategy': 'Dự phòng phổ biến',
        'ML_Score': [0.99] * min(5, len(popular_items)),
        'Jaccard': [0.0] * min(5, len(popular_items)),
        'AA': [0.0] * min(5, len(popular_items)),
        'PA': [0.0] * min(5, len(popular_items))
    })
    return fallback_recs


def draw_ego_network(g, central_node, ax, central_color, neighbor_color, central_size, neighbor_size, title):
    """Vẽ đồ thị ego xung quanh một node"""
    if g.has_node(central_node):
        try:
            ego_g = nx.ego_graph(g, central_node)
            colors = []
            sizes = []
            
            for node in ego_g.nodes():
                if node == central_node:
                    colors.append(central_color)
                    sizes.append(central_size)
                else:
                    colors.append(neighbor_color)
                    sizes.append(neighbor_size)
            
            pos_ego = nx.spring_layout(ego_g, seed=42)
            nx.draw(ego_g, pos_ego, ax=ax, node_color=colors, node_size=sizes, 
                   with_labels=False, alpha=0.8, edge_color='lightgray')
            ax.set_title(title)
        except:
            ax.set_title(f"{title} (Không thể vẽ)")
    else:
        ax.set_title(f"{title} (Node không tồn tại)")


def render_part6(G, node_classes, model=None, scaler=None, df_test=None, G_full=None):
    """Render giao dien Part 6: Case Study
    
    Args:
        G: G_category - subgraph để phân loại node và hiển thị
        node_classes: Dict phân loại node (Core/Inner/Outer)
        model: Mô hình ML đã train
        scaler: StandardScaler đã fit
        df_test: DataFrame test chứa features để đánh giá
        G_full: Đồ thị gốc đầy đủ để tạo gợi ý (theo notebook)
    """
    # Nếu không có G_full, fallback về G
    if G_full is None:
        G_full = G
    
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
    .conclusion-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #FFFFFF 100%);
        border-left: 4px solid #2196F3;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .core-color { color: #d62728; font-weight: 600; }
    .periphery-color { color: #1f77b4; font-weight: 600; }
    .ml-color { color: #FF6B00; font-weight: 600; }
    
    /* Button styling */
    .stButton > button {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    
    /* Node card */
    .node-card {
        background: #FFFFFF;
        border: 2px solid #E8E8E8;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .node-card-core {
        border-color: #d62728;
        background: linear-gradient(135deg, #FFEBEE 0%, #FFFFFF 100%);
    }
    .node-card-periphery {
        border-color: #1f77b4;
        background: linear-gradient(135deg, #E3F2FD 0%, #FFFFFF 100%);
    }
    .node-title {
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .node-info {
        font-size: 0.85rem;
        color: #666;
    }
    
    /* Summary section */
    .summary-card {
        background: linear-gradient(135deg, #232F3E 0%, #37475A 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .summary-title {
        color: #FF9900;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .summary-item {
        color: #FFFFFF;
        font-size: 0.95rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    .summary-item:last-child {
        border-bottom: none;
    }
    .summary-highlight {
        color: #FF9900;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class="part-header">
        <div class="part-title">Comparative Case Study</div>
        <div class="part-desc">
            So sánh trực tiếp hiệu quả của ML và Heuristic, phân tích Core vs Periphery Node
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== THÔNG TIN ĐỒ THỊ ĐANG PHÂN TÍCH ====================
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    selected_cat = st.session_state.get('selected_cat_idx', None)
    
    # ==================== MỤC ĐÍCH ====================
    st.markdown("""
    <div class="highlight-box">
        <strong>Mục đích:</strong> So sánh trực tiếp hiệu quả của <span class="ml-color">Machine Learning</span> 
        và <strong>Heuristic</strong> trên các đối tượng cụ thể, phân tích chiến lược gợi ý cho 
        <span class="core-color">Core Node</span> vs <span class="periphery-color">Periphery Node</span>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== TÍNH NODE_CLASSES TỪ G_FULL ====================
    # Để tránh data leakage, lấy node từ G_full (không phải subgraph đã train)
    @st.cache_data
    def compute_node_classes_for_full(_G_full):
        """Tính node_classes từ G_full bằng K-core decomposition"""
        core_nums = nx.core_number(_G_full)
        max_k = max(core_nums.values()) if core_nums else 1
        
        # Ngưỡng phân loại (giống Part 3)
        core_threshold = max_k * 0.7
        inner_threshold = max_k * 0.3
        
        node_classes_full = {}
        for node, k in core_nums.items():
            if k >= core_threshold:
                node_classes_full[node] = 'Core Zone'
            elif k >= inner_threshold:
                node_classes_full[node] = 'Inner Periphery'
            else:
                node_classes_full[node] = 'Outer Periphery'
        
        return node_classes_full
    
    # Tính node_classes từ G_full
    node_classes_full = compute_node_classes_for_full(G_full)
    
    # ==================== CHỌN NODE ĐỂ PHÂN TÍCH ====================
    st.markdown('<div class="section-title">Chọn Node để phân tích (từ G_full)</div>', unsafe_allow_html=True)
    
    # Lấy các node phổ biến từ G_full
    top_popular_items = sorted(dict(G_full.degree()).items(), key=lambda x: x[1], reverse=True)[:20]
    popular_ids = [n for n, d in top_popular_items]
    
    # Phân loại node từ G_full
    core_nodes = [n for n, c in node_classes_full.items() if c == 'Core Zone']
    periphery_nodes = [n for n, c in node_classes_full.items() if c == 'Outer Periphery']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="node-card node-card-core">
            <div class="node-title" style="color: #d62728;">🔴 Core Node</div>
            <div class="node-info">Node có bậc cao, nằm ở vùng lõi của mạng (từ G_full)</div>
        </div>
        """, unsafe_allow_html=True)
        if core_nodes:
            u_core = st.selectbox(
                "Chọn Core Node:",
                options=core_nodes[:50],
                format_func=lambda x: f"Node {x} (Degree: {G_full.degree(x)})",
                key="core_node_select"
            )
        else:
            u_core = popular_ids[0] if popular_ids else list(G.nodes())[0]
            st.warning("⚠️ Không tìm thấy Core Node, sử dụng node phổ biến nhất.")
    
    with col2:
        st.markdown("""
        <div class="node-card node-card-periphery">
            <div class="node-title" style="color: #1f77b4;">🔵 Periphery Node</div>
            <div class="node-info">Node có bậc thấp, nằm ở vùng ngoại vi của mạng (từ G_full)</div>
        </div>
        """, unsafe_allow_html=True)
        if periphery_nodes:
            u_periphery = st.selectbox(
                "Chọn Periphery Node:",
                options=periphery_nodes[:50],
                format_func=lambda x: f"Node {x} (Degree: {G_full.degree(x)})",
                key="periphery_node_select"
            )
        else:
            u_periphery = list(G_full.nodes())[-1]
            st.warning("⚠️ Không tìm thấy Periphery Node.")
    
    # ==================== TABS ====================
    st.markdown('<div class="section-title">Phân tích chi tiết</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "Gợi ý chiến lược", 
        "Ego Network",
        "Đánh giá cuối cùng"
    ])
    
    # ==================== TAB 1: GỢI Ý CHIẾN LƯỢC ====================
    with tab1:
        st.markdown('<div class="section-title">Chiến lược gợi ý cho từng loại Node</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="node-card node-card-core">
                <div class="node-title" style="color: #d62728;">Core Node: {u_core}</div>
                <div class="node-info">Degree: {G_full.degree(u_core)} | Class: {node_classes_full.get(u_core, 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)
            def sort_core_periphery(df, node_label):
                if 'Core_Label' in df.columns:
                    if node_label == 'Core Zone':
                        df = df.sort_values(['Core_Label', 'ML_Score'], ascending=[True, False])
                        df = pd.concat([
                            df[df['Core_Label'] == 'Outer Periphery'],
                            df[df['Core_Label'] != 'Outer Periphery']
                        ])
                    elif node_label == 'Outer Periphery':
                        df = df.sort_values(['Core_Label', 'ML_Score'], ascending=[False, False])
                        df = pd.concat([
                            df[df['Core_Label'] == 'Core Zone'],
                            df[df['Core_Label'] != 'Core Zone']
                        ])
                    else:
                        df = df.sort_values('ML_Score', ascending=False)
                return df
            if model is not None and scaler is not None:
                recs_core = get_smart_recommendations(G_full, u_core, model, scaler, popular_ids)
                # Bổ sung nhãn core-periphery cho các candidate
                recs_core['Core_Label'] = recs_core['Candidate'].map(node_classes_full)
                recs_core = sort_core_periphery(recs_core, node_classes_full.get(u_core, 'Unknown'))
                st.dataframe(recs_core.style.format({
                    'ML_Score': '{:.4f}',
                    'Jaccard': '{:.4f}',
                    'AA': '{:.4f}',
                    'PA': '{:,.0f}'
                }).background_gradient(subset=['ML_Score'], cmap='OrRd'), use_container_width=True, hide_index=True)
            else:
                st.markdown("""
                <div class="highlight-box">
                    ⚠️ Chưa có mô hình ML. Vui lòng huấn luyện ở Part 5.
                </div>
                """, unsafe_allow_html=True)
                neighbors = list(G_full.neighbors(u_core))[:5]
                st.write("**Hàng xóm trực tiếp:**", neighbors)
        
        with col2:
            st.markdown(f"""
            <div class="node-card node-card-periphery">
                <div class="node-title" style="color: #1f77b4;">Periphery Node: {u_periphery}</div>
                <div class="node-info">Degree: {G_full.degree(u_periphery)} | Class: {node_classes_full.get(u_periphery, 'Unknown')}</div>
            </div>
            """, unsafe_allow_html=True)
            if model is not None and scaler is not None:
                recs_peri = get_smart_recommendations(G_full, u_periphery, model, scaler, popular_ids)
                recs_peri['Core_Label'] = recs_peri['Candidate'].map(node_classes_full)
                recs_peri = sort_core_periphery(recs_peri, node_classes_full.get(u_periphery, 'Unknown'))
                st.dataframe(recs_peri.style.format({
                    'ML_Score': '{:.4f}',
                    'Jaccard': '{:.4f}',
                    'AA': '{:.4f}',
                    'PA': '{:,.0f}'
                }).background_gradient(subset=['ML_Score'], cmap='Blues'), use_container_width=True, hide_index=True)
            else:
                st.markdown("""
                <div class="highlight-box">
                    ⚠️ Chưa có mô hình ML. Vui lòng huấn luyện ở Part 5.
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="interpretation-box">
            <strong>Nhận xét:</strong><br>
            • <span class="core-color">Core Node</span> thường có nhiều gợi ý chất lượng hơn do có nhiều kết nối<br>
            • <span class="periphery-color">Periphery Node</span> có thể cần chiến lược <strong>"Popularity-based"</strong> thay vì "Similarity-based"<br>
            • ML Score càng cao → khả năng kết nối (mua chung) càng lớn
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 2: EGO NETWORK ====================
    with tab2:
        st.markdown('<div class="section-title">Trực quan hóa Ego Network</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Ego Network</strong> là đồ thị con bao gồm một node trung tâm và tất cả các hàng xóm trực tiếp của nó.
            So sánh mật độ kết nối giữa Core và Periphery Node.
        </div>
        """, unsafe_allow_html=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#F7F8F8')
        
        draw_ego_network(
            G_full, u_core, ax[0],
            central_color='#d62728',
            neighbor_color='#ff9896',
            central_size=600,
            neighbor_size=80,
            title=f"Ego Network - Core Node ({u_core})"
        )
        ax[0].set_facecolor('#FAFAFA')
        
        draw_ego_network(
            G_full, u_periphery, ax[1],
            central_color='#1f77b4',
            neighbor_color='#aec7e8',
            central_size=600,
            neighbor_size=80,
            title=f"Ego Network - Periphery Node ({u_periphery})"
        )
        ax[1].set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # So sánh metrics
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            core_degree = G_full.degree(u_core)
            try:
                core_clustering = nx.clustering(G_full, u_core)
            except:
                core_clustering = 0
            
            st.markdown(f"""
            <div class="info-card" style="border-left: 4px solid #d62728;">
                <div style="font-weight: 600; color: #d62728; margin-bottom: 0.8rem;">Core Node ({u_core})</div>
                <div style="display: flex; justify-content: space-around;">
                    <div>
                        <div class="info-value" style="color: #d62728;">{core_degree:,}</div>
                        <div class="info-label">Degree</div>
                    </div>
                    <div>
                        <div class="info-value" style="color: #d62728;">{core_clustering:.4f}</div>
                        <div class="info-label">Clustering Coef.</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            peri_degree = G_full.degree(u_periphery)
            try:
                peri_clustering = nx.clustering(G_full, u_periphery)
            except:
                peri_clustering = 0
            
            st.markdown(f"""
            <div class="info-card" style="border-left: 4px solid #1f77b4;">
                <div style="font-weight: 600; color: #1f77b4; margin-bottom: 0.8rem;">Periphery Node ({u_periphery})</div>
                <div style="display: flex; justify-content: space-around;">
                    <div>
                        <div class="info-value" style="color: #1f77b4;">{peri_degree:,}</div>
                        <div class="info-label">Degree</div>
                    </div>
                    <div>
                        <div class="info-value" style="color: #1f77b4;">{peri_clustering:.4f}</div>
                        <div class="info-label">Clustering Coef.</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="interpretation-box">
            <strong>So sánh:</strong><br>
            • Core Node có <strong>{core_degree}</strong> kết nối, Periphery Node có <strong>{peri_degree}</strong> kết nối<br>
            • Chênh lệch: <strong>{core_degree - peri_degree:,}</strong> kết nối (Core cao hơn {(core_degree/max(peri_degree,1)):.1f}x)<br>
            • Core Node nằm ở vị trí trung tâm → phù hợp cho chiến lược Cross-sell dựa trên tương đồng
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 3: ĐÁNH GIÁ CUỐI CÙNG ====================
    with tab3:
        st.markdown('<div class="section-title">Đánh giá cuối cùng: ML vs Heuristic</div>', unsafe_allow_html=True)
        
        if model is not None and df_test is not None and 'y_test_lr' in st.session_state:
            X_test = st.session_state.get('X_test_lr')
            y_test = st.session_state.get('y_test_lr')
            
            if X_test is not None and y_test is not None:
                fig, ax = plt.subplots(figsize=(10, 7))
                fig.patch.set_facecolor('#F7F8F8')
                ax.set_facecolor('#FAFAFA')
                
                # ML ROC
                y_pred_ml = model.predict_proba(X_test)[:, 1]
                auc_ml = roc_auc_score(y_test, y_pred_ml)
                fpr_ml, tpr_ml, _ = roc_curve(y_test, y_pred_ml)
                ax.plot(fpr_ml, tpr_ml, color='#FF6B00', lw=3, 
                       label=f'Machine Learning - AUC={auc_ml:.4f}')
                
                # Heuristics ROC
                heuristics = ['Jaccard', 'AA', 'PA', 'RA']
                colors = ['#2196F3', '#4CAF50', '#9C27B0', '#795548']
                
                best_heuristic_auc = 0
                best_heuristic_name = ""
                for metric, color in zip(heuristics, colors):
                    if metric in df_test.columns:
                        y_scores = df_test[metric].fillna(0)
                        try:
                            auc_h = roc_auc_score(y_test, y_scores)
                            if auc_h > best_heuristic_auc:
                                best_heuristic_auc = auc_h
                                best_heuristic_name = metric
                            fpr_h, tpr_h, _ = roc_curve(y_test, y_scores)
                            ax.plot(fpr_h, tpr_h, color=color, linestyle='--', alpha=0.7, lw=2,
                                   label=f'{metric} - AUC={auc_h:.4f}')
                        except:
                            continue
                
                ax.plot([0, 1], [0, 1], 'k:', label='Random (AUC=0.5)')
                ax.set_xlabel('False Positive Rate', fontsize=11)
                ax.set_ylabel('True Positive Rate', fontsize=11)
                ax.set_title('Kiểm định với dữ liệu thật: ML vs Heuristics', fontweight='bold', fontsize=13, color='#232F3E')
                ax.legend(loc="lower right")
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
                
                # Metrics comparison
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-value" style="color: #FF6B00;">{auc_ml:.4f}</div>
                        <div class="info-label">ML AUC Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-value" style="color: #2196F3;">{best_heuristic_auc:.4f}</div>
                        <div class="info-label">Best Heuristic ({best_heuristic_name})</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    improvement = (auc_ml - best_heuristic_auc) * 100
                    imp_color = "#00b894" if improvement > 0 else "#e74c3c"
                    st.markdown(f"""
                    <div class="info-card">
                        <div class="info-value" style="color: {imp_color};">{improvement:+.2f}%</div>
                        <div class="info-label">ML Improvement</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Kết luận
                if improvement > 0:
                    st.markdown(f"""
                    <div class="interpretation-box">
                        <strong>KẾT LUẬN:</strong> Mô hình <span class="ml-color">Machine Learning</span> 
                        cải thiện độ chính xác thêm <strong>{improvement:.2f}%</strong> so với thuật toán 
                        Heuristic tốt nhất (<strong>{best_heuristic_name}</strong>).<br><br>
                        → <strong>Khuyến nghị:</strong> Sử dụng ML để tạo gợi ý Cross-sell chính xác hơn.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="conclusion-box">
                        <strong>KẾT LUẬN:</strong> Heuristic <strong>{best_heuristic_name}</strong> 
                        đạt hiệu quả tương đương hoặc tốt hơn ML trong trường hợp này.<br><br>
                        → <strong>Khuyến nghị:</strong> Cân nhắc sử dụng Heuristic đơn giản hơn hoặc cải thiện features cho ML.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="highlight-box">
                    ⚠️ Dữ liệu kiểm thử không khả dụng.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="highlight-box">
                ⚠️ <strong>Chưa có kết quả mô hình ML để đánh giá.</strong><br>
                Vui lòng huấn luyện mô hình ở <strong>Part 5: Link Prediction Model</strong> trước.
            </div>
            """, unsafe_allow_html=True)
        
        # ==================== TÓM TẮT DỰ ÁN ====================
        st.markdown('<div class="section-title">Tóm tắt dự án</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="summary-card">
            <div class="summary-title">Những gì đã làm được</div>
            <div class="summary-item">
                <span class="summary-highlight">Part 1:</span> Nạp và tiền xử lý dữ liệu Amazon Co-purchasing Network
            </div>
            <div class="summary-item">
                <span class="summary-highlight">Part 2:</span> Phân tích thống kê mạng lưới (Degree, Clustering, Assortativity)
            </div>
            <div class="summary-item">
                <span class="summary-highlight">Part 3:</span> Phân tích cấu trúc Core-Periphery bằng K-core Decomposition
            </div>
            <div class="summary-item">
                <span class="summary-highlight">Part 4:</span> Xây dựng hệ thống gợi ý Cross-sell dựa trên Heuristics
            </div>
            <div class="summary-item">
                <span class="summary-highlight">Part 5:</span> Huấn luyện mô hình Machine Learning (Logistic Regression, Random Forest)
            </div>
            <div class="summary-item">
                <span class="summary-highlight">Part 6:</span> So sánh và đánh giá hiệu quả của các phương pháp
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="info-card" style="border-top: 4px solid #FF9900;">
                <div style="font-weight: 600; color: #232F3E; margin-top: 0.5rem;">Core-Periphery</div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.3rem;">
                    Xác định được cấu trúc lõi-ngoại vi trong mạng lưới sản phẩm
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="info-card" style="border-top: 4px solid #00b894;">
                <div style="font-weight: 600; color: #232F3E; margin-top: 0.5rem;">ML > Heuristics</div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.3rem;">
                    Machine Learning vượt trội hơn trong hầu hết các trường hợp
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="info-card" style="border-top: 4px solid #2196F3;">
                <div style="font-weight: 600; color: #232F3E; margin-top: 0.5rem;">Cross-sell Ready</div>
                <div style="font-size: 0.85rem; color: #666; margin-top: 0.3rem;">
                    Core Nodes có tiềm năng cao cho chiến lược bán chéo
                </div>
            </div>
            """, unsafe_allow_html=True)
