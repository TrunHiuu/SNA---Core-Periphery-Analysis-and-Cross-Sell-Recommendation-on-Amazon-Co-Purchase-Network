"""
PART 4: CROSS-SELL RECOMMENDATION (HEURISTICS)
Mục đích:
- Xây dựng danh sách gợi ý cơ bản (Baseline) dựa trên các chỉ số cấu trúc mạng
- Các chỉ số sử dụng: Jaccard, Adamic-Adar, Preferential Attachment, Resource Allocation
- Tạo bảng tổng hợp Cross-sell cho các sản phẩm Core
- Đánh giá kết quả với ROC Curve
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import roc_curve, roc_auc_score


def get_score_details(graph, u, v):
    """Tính toán các chỉ số tương đồng và liên kết giữa hai node"""
    try:
        jaccard = list(nx.jaccard_coefficient(graph, [(u, v)]))[0][2]
        aa = list(nx.adamic_adar_index(graph, [(u, v)]))[0][2]
        pa = graph.degree(u) * graph.degree(v)
        ra = list(nx.resource_allocation_index(graph, [(u, v)]))[0][2]
    except:
        jaccard, aa, pa, ra = 0, 0, 0, 0
    return jaccard, aa, pa, ra


@st.cache_data
def generate_cross_sell_recommendations(_G, node_classes, top_n_anchors=20, n_candidates=50):
    """Tạo gợi ý bán chéo dựa trên Heuristics"""
    
    # Xác định các Anchor (Core Nodes)
    anchors = [n for n, c in node_classes.items() if c == 'Core Zone']
    top_anchors = sorted(anchors, key=lambda n: _G.degree(n), reverse=True)[:top_n_anchors]
    
    # Tìm các cạnh hiện có
    existing_rows = []
    for u in top_anchors:
        for v in _G.neighbors(u):
            jc, aa, pa, ra = get_score_details(_G, u, v)
            existing_rows.append({
                'Base_Item': u, 
                'Candidate': v, 
                'Type': 'Existing',
                'Jaccard': jc, 
                'Adamic_Adar': aa, 
                'Pref_Attach': pa, 
                'Resource_Alloc': ra
            })
    
    df_existing = pd.DataFrame(existing_rows)
    
    # Tìm các ứng viên mới (Gợi ý)
    new_rows = []
    full_nodes = list(_G.nodes())
    
    for u in top_anchors:
        nbrs = set(_G.neighbors(u))
        potential = random.sample(full_nodes, min(n_candidates, len(full_nodes)))
        
        for v in potential:
            if v != u and v not in nbrs:
                jc, aa, pa, ra = get_score_details(_G, u, v)
                if jc > 0:
                    new_rows.append({
                        'Base_Item': u, 
                        'Candidate': v, 
                        'Type': 'Suggested',
                        'Jaccard': jc, 
                        'Adamic_Adar': aa, 
                        'Pref_Attach': pa, 
                        'Resource_Alloc': ra
                    })
    
    df_new = pd.DataFrame(new_rows)
    
    # Nối kết quả
    cross_sell_table = pd.concat([df_existing, df_new], ignore_index=True)
    cross_sell_table = cross_sell_table.sort_values(['Type', 'Jaccard'], ascending=[True, False])
    
    return cross_sell_table, top_anchors


@st.cache_data
def evaluate_heuristics(_G, n_samples=2000):
    """Đánh giá hiệu quả của các phương pháp Heuristic"""
    eval_data = []
    
    # Mẫu dương (cạnh thực)
    real_edges = list(_G.edges())
    if len(real_edges) > n_samples:
        pos_samples = random.sample(real_edges, n_samples)
    else:
        pos_samples = real_edges
    
    for u, v in pos_samples:
        eval_data.append({'u': u, 'v': v, 'Label': 1})
    
    # Mẫu âm (không có cạnh)
    all_nodes_list = list(_G.nodes())
    count = 0
    while count < n_samples:
        u, v = random.sample(all_nodes_list, 2)
        if not _G.has_edge(u, v):
            eval_data.append({'u': u, 'v': v, 'Label': 0})
            count += 1
    
    df_eval = pd.DataFrame(eval_data)
    
    # Tính điểm heuristic
    scores = []
    for _, row in df_eval.iterrows():
        jc, aa, pa, ra = get_score_details(_G, row['u'], row['v'])
        scores.append({'Jaccard': jc, 'AA': aa, 'PA': pa, 'RA': ra})
    
    df_scores = pd.DataFrame(scores)
    df_eval = pd.concat([df_eval, df_scores], axis=1)
    
    return df_eval


def render_part4(G, node_classes):
    """Render giao dien Part 4: Cross-Sell Recommendation"""
    
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
    .metric-existing { color: #2196F3; }
    .metric-suggested { color: #4CAF50; }
    
    /* Button styling */
    .stButton > button {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class="part-header">
        <div class="part-title">Cross-Sell Recommendation (Heuristics)</div>
        <div class="part-desc">
            Xây dựng danh sách gợi ý bán chéo dựa trên các chỉ số cấu trúc mạng lưới
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
        <strong>Mục đích:</strong> Xây dựng danh sách gợi ý cơ bản (Baseline) dựa trên các chỉ số cấu trúc mạng để đề xuất sản phẩm bán kèm
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== GIẢI THÍCH CÁC CHỈ SỐ ====================
    st.markdown('<div class="section-title">Các chỉ số Heuristics sử dụng</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight: 600; color: #232F3E; margin-bottom: 0.5rem;">Jaccard Coefficient</div>
            <div style="font-size: 0.85rem; color: #666;">Đo lường sự chồng chéo của hàng xóm chung giữa hai node</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <div style="font-weight: 600; color: #232F3E; margin-bottom: 0.5rem;">Preferential Attachment</div>
            <div style="font-size: 0.85rem; color: #666;">Tích của bậc hai node - node bậc cao có xu hướng kết nối thêm</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight: 600; color: #232F3E; margin-bottom: 0.5rem;">Adamic-Adar Index</div>
            <div style="font-size: 0.85rem; color: #666;">Trọng số hóa hàng xóm chung theo nghịch đảo log bậc</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <div style="font-weight: 600; color: #232F3E; margin-bottom: 0.5rem;">Resource Allocation</div>
            <div style="font-size: 0.85rem; color: #666;">Đánh giá tài nguyên được chia sẻ qua hàng xóm chung</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TABS CHÍNH ====================
    st.markdown('<div class="section-title">Phân tích chi tiết</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["Gợi ý bán chéo", "Đánh giá Heuristics"])
    
    # ==================== TAB 1: GỢI Ý BÁN CHÉO ====================
    with tab1:
        st.markdown('<div class="section-title">Bảng gợi ý bán chéo (Cross-sell Recommendation)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Cách hoạt động:</strong> Chọn các sản phẩm Anchor (Core Zone) và tìm các ứng viên tiềm năng 
            dựa trên các chỉ số heuristics. Gợi ý mới là các cặp sản phẩm chưa có liên kết nhưng có điểm tương đồng cao.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            n_anchors = st.slider("Số lượng Anchor (Core nodes):", 5, 50, 20)
        with col2:
            n_candidates = st.slider("Số ứng viên mỗi Anchor:", 20, 100, 50)
        
        if st.button("Tạo gợi ý", key="gen_recommendations"):
            with st.spinner("Đang tạo gợi ý bán chéo..."):
                cross_sell_table, top_anchors = generate_cross_sell_recommendations(
                    G, node_classes, n_anchors, n_candidates
                )
            
            st.session_state['cross_sell_table'] = cross_sell_table
            st.session_state['top_anchors'] = top_anchors
        
        if 'cross_sell_table' in st.session_state:
            cross_sell_table = st.session_state['cross_sell_table']
            
            # Thống kê với info-card
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value">{len(cross_sell_table):,}</div>
                    <div class="info-label">Tổng số gợi ý</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                existing_count = len(cross_sell_table[cross_sell_table['Type']=='Existing'])
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value" style="color: #2196F3;">{existing_count:,}</div>
                    <div class="info-label">Cạnh hiện có</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                suggested_count = len(cross_sell_table[cross_sell_table['Type']=='Suggested'])
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value" style="color: #4CAF50;">{suggested_count:,}</div>
                    <div class="info-label">Gợi ý mới</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Filter
            type_filter = st.radio("Lọc theo loại:", ['Tất cả', 'Existing', 'Suggested'], horizontal=True)
            
            if type_filter != 'Tất cả':
                display_df = cross_sell_table[cross_sell_table['Type'] == type_filter]
            else:
                display_df = cross_sell_table
            
            st.dataframe(display_df.head(100).style.format({
                'Jaccard': '{:.4f}',
                'Adamic_Adar': '{:.4f}',
                'Pref_Attach': '{:,.0f}',
                'Resource_Alloc': '{:.4f}'
            }), use_container_width=True, hide_index=True)
            
            # Biểu đồ Top gợi ý
            st.markdown('<div class="section-title">Top 10 gợi ý bán chéo (theo Jaccard)</div>', unsafe_allow_html=True)
            
            top_sugg = cross_sell_table[cross_sell_table['Type'] == 'Suggested'].head(10)
            if not top_sugg.empty:
                fig, ax = plt.subplots(figsize=(12, 5))
                x = range(len(top_sugg))
                bars = ax.bar(x, top_sugg['Jaccard'], color='#FF9900', alpha=0.8, edgecolor='#232F3E')
                ax.set_xticks(x)
                ax.set_xticklabels([f"{row['Base_Item']}→{row['Candidate']}" for _, row in top_sugg.iterrows()], rotation=45, ha='right')
                ax.set_ylabel("Jaccard Score", fontsize=11)
                ax.set_title("Top 10 gợi ý bán chéo mới", fontweight='bold', fontsize=13, color='#232F3E')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Nhận xét
                st.markdown(f"""
                <div class="interpretation-box">
                    <strong>Nhận xét:</strong><br>
                    • Tìm thấy <strong>{suggested_count:,}</strong> gợi ý mới từ <strong>{n_anchors}</strong> sản phẩm Anchor<br>
                    • Các cặp sản phẩm có Jaccard cao cho thấy có nhiều hàng xóm chung → khách hàng mua sản phẩm này có xu hướng mua sản phẩm kia<br>
                    • <strong>Ứng dụng:</strong> Sử dụng danh sách này để hiển thị "Khách hàng cũng mua" hoặc tạo bundle sản phẩm
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Không tìm thấy gợi ý mới. Thử tăng số ứng viên hoặc kiểm tra dữ liệu.")
    
    # ==================== TAB 2: ĐÁNH GIÁ HEURISTICS ====================
    with tab2:
        st.markdown('<div class="section-title">Đánh giá hiệu quả của các phương pháp Heuristic</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Phương pháp đánh giá:</strong> So sánh khả năng phân biệt cạnh thực (có liên kết) và cạnh giả (không liên kết) 
            của từng chỉ số heuristic bằng ROC-AUC Score.
        </div>
        """, unsafe_allow_html=True)
        
        n_eval_samples = st.slider("Số mẫu đánh giá:", 500, 5000, 2000)
        
        if st.button("Đánh giá", key="eval_heuristics"):
            with st.spinner("Đang đánh giá hiệu quả các phương pháp..."):
                df_eval = evaluate_heuristics(G, n_eval_samples)
            st.session_state['df_eval_heuristics'] = df_eval
        
        if 'df_eval_heuristics' in st.session_state:
            df_eval = st.session_state['df_eval_heuristics']
            
            # Vẽ ROC Curve
            heuristics = ['Jaccard', 'AA', 'PA', 'RA']
            heuristic_names = {'Jaccard': 'Jaccard Coefficient', 'AA': 'Adamic-Adar', 'PA': 'Preferential Attachment', 'RA': 'Resource Allocation'}
            colors = ['#FF9900', '#00b894', '#6c5ce7', '#e17055']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            results = []
            for metric, color in zip(heuristics, colors):
                y_true = df_eval['Label']
                y_scores = df_eval[metric].fillna(0)
                
                try:
                    auc = roc_auc_score(y_true, y_scores)
                    results.append({'Phương pháp': heuristic_names[metric], 'AUC': auc})
                    fpr, tpr, _ = roc_curve(y_true, y_scores)
                    ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{heuristic_names[metric]} (AUC={auc:.3f})')
                except:
                    continue
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.5)')
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title('ROC Curve - So sánh hiệu suất các phương pháp Heuristic', fontweight='bold', fontsize=13, color='#232F3E')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
            # Bảng tóm tắt với styling
            st.markdown('<div class="section-title">Tóm tắt AUC của các phương pháp</div>', unsafe_allow_html=True)
            
            df_results = pd.DataFrame(results).sort_values('AUC', ascending=False)
            st.dataframe(df_results.style.format({'AUC': '{:.4f}'}).background_gradient(subset=['AUC'], cmap='YlOrRd'), 
                        use_container_width=True, hide_index=True)
            
            best_method = df_results.iloc[0]['Phương pháp']
            best_auc = df_results.iloc[0]['AUC']
            
            # Nhận xét
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>Nhận xét:</strong><br>
                • <strong>Phương pháp tốt nhất:</strong> {best_method} với AUC = {best_auc:.4f}<br>
                • AUC > 0.7 được coi là hiệu quả tốt, AUC > 0.8 là rất tốt<br>
                • Các phương pháp Heuristics này có thể làm baseline để so sánh với mô hình Machine Learning
            </div>
            """, unsafe_allow_html=True)
    
    return st.session_state.get('cross_sell_table', pd.DataFrame())
