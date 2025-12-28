"""
PART 5: LINK PREDICTION MODEL (MACHINE LEARNING)
Mục đích:
- Huấn luyện mô hình Machine Learning (Logistic Regression, Random Forest)
- Sử dụng StandardScaler và Hyperparameter Tuning
- Đánh giá mô hình bằng AUC, Precision@K, Recall@K
- So sánh ML với Heuristics
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score


def smart_negative_sampling_mixed(graph, pos_edges, n_samples):
    """Chiến lược lấy mẫu hỗn hợp (Mixed Strategy)"""
    negatives = set()
    nodes = list(graph.nodes())
    degrees = dict(graph.degree())
    
    # Ngưỡng xác định Hub (Top 10%)
    threshold = np.percentile(list(degrees.values()), 90)
    hubs = [n for n, d in degrees.items() if d >= threshold]
    
    def is_valid_neg(u, v):
        if u == v:
            return False
        if graph.has_edge(u, v):
            return False
        if (u, v) in negatives or (v, u) in negatives:
            return False
        return True
    
    # 1. Hub-Hub pairs (30%)
    target_hub = int(n_samples * 0.3)
    attempts = 0
    while len(negatives) < target_hub and attempts < target_hub * 50:
        if len(hubs) < 2:
            break
        u, v = random.sample(hubs, 2)
        if is_valid_neg(u, v):
            negatives.add((u, v))
        attempts += 1
    
    # 2. Hard negatives (30%)
    target_hard = int(n_samples * 0.3)
    target_total_hard = len(negatives) + target_hard
    
    for u, v in pos_edges:
        if len(negatives) >= target_total_hard:
            break
        neighbors_u = list(graph.neighbors(u))
        if len(neighbors_u) > 1:
            w = random.choice(neighbors_u)
            if w != v and is_valid_neg(v, w):
                negatives.add((v, w))
    
    # 3. Random pairs (40%)
    attempts = 0
    while len(negatives) < n_samples and attempts < n_samples * 50:
        u, v = random.sample(nodes, 2)
        if is_valid_neg(u, v):
            negatives.add((u, v))
        attempts += 1
    
    return list(negatives)


def extract_features(graph, pairs, label, core_nums=None):
    """Trích xuất đặc trưng cho các cặp node, bổ sung k-core"""
    data = []
    for u, v in pairs:
        try:
            jc = list(nx.jaccard_coefficient(graph, [(u, v)]))[0][2]
            aa = list(nx.adamic_adar_index(graph, [(u, v)]))[0][2]
            pa = graph.degree(u) * graph.degree(v)
            ra = list(nx.resource_allocation_index(graph, [(u, v)]))[0][2]
            deg_diff = abs(graph.degree(u) - graph.degree(v))
            kcore_u = core_nums[u] if core_nums else 0
            kcore_v = core_nums[v] if core_nums else 0
            data.append([jc, aa, pa, ra, deg_diff, kcore_u, kcore_v, label])
        except:
            continue
    return pd.DataFrame(data, columns=['Jaccard', 'AA', 'PA', 'RA', 'DegDiff', 'KCore_u', 'KCore_v', 'Label'])


def extract_features_rf(graph, pairs, label, core_nums=None):
    """Trích xuất đặc trưng cho Random Forest (thêm các đặc trưng cấu trúc và k-core)"""
    data = []
    clustering = nx.clustering(graph)
    for u, v in pairs:
        try:
            jc = list(nx.jaccard_coefficient(graph, [(u, v)]))[0][2]
            aa = list(nx.adamic_adar_index(graph, [(u, v)]))[0][2]
            pa = graph.degree(u) * graph.degree(v)
            ra = list(nx.resource_allocation_index(graph, [(u, v)]))[0][2]
            deg_diff = abs(graph.degree(u) - graph.degree(v))
            cn_set = set(nx.common_neighbors(graph, u, v))
            triangles = len([n for n in cn_set if graph.has_edge(u, n) or graph.has_edge(v, n)])
            clust_avg = (clustering.get(u, 0) + clustering.get(v, 0)) / 2
            kcore_u = core_nums[u] if core_nums else 0
            kcore_v = core_nums[v] if core_nums else 0
            data.append([jc, aa, pa, ra, deg_diff, triangles, clust_avg, kcore_u, kcore_v, label])
        except:
            continue
    return pd.DataFrame(data, columns=[
        'Jaccard', 'AA', 'PA', 'RA', 'DegDiff', 'Triangles', 'Clust_Avg', 'KCore_u', 'KCore_v', 'Label'
    ])


def evaluate_at_k(y_true, y_scores, k_list):
    """Tính các chỉ số tại Top K"""
    df = pd.DataFrame({'label': y_true, 'score': y_scores}).sort_values('score', ascending=False)
    metrics = []
    total_pos = df['label'].sum()
    
    for k in k_list:
        top_k = df.head(k)
        pk = top_k['label'].sum() / k
        rk = top_k['label'].sum() / total_pos if total_pos > 0 else 0
        
        hits = 0
        precs = []
        for i in range(k):
            if df.iloc[i]['label'] == 1:
                hits += 1
                precs.append(hits / (i + 1))
        mapk = sum(precs) / min(k, total_pos) if precs else 0
        
        metrics.append({'K': k, 'P@K': pk, 'R@K': rk, 'MAP@K': mapk})
    
    return pd.DataFrame(metrics)


@st.cache_resource
def train_logistic_regression(_G, max_edges=20000):
    """Huấn luyện mô hình Logistic Regression"""
    edges = list(_G.edges())
    if len(edges) > max_edges:
        edges = random.sample(edges, max_edges)
    
    train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)
    # Tạo đồ thị huấn luyện
    G_train = _G.edge_subgraph(train_edges).copy()
    G_train.add_nodes_from(_G.nodes())
    # Tính k-core cho toàn bộ node
    core_nums = nx.core_number(G_train)
    # Lấy mẫu âm
    train_neg = smart_negative_sampling_mixed(G_train, train_edges, len(train_edges))
    test_neg = smart_negative_sampling_mixed(G_train, test_edges, len(test_edges))
    # Trích xuất đặc trưng (bổ sung k-core)
    df_train = pd.concat([
        extract_features(G_train, train_edges, 1, core_nums),
        extract_features(G_train, train_neg, 0, core_nums)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = pd.concat([
        extract_features(G_train, test_edges, 1, core_nums),
        extract_features(G_train, test_neg, 0, core_nums)
    ]).reset_index(drop=True)
    # Chuẩn hóa
    features = ['Jaccard', 'AA', 'PA', 'RA', 'DegDiff', 'KCore_u', 'KCore_v']
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    y_train = df_train['Label']
    X_test = scaler.transform(df_test[features])
    y_test = df_test['Label']
    # Tìm tham số C tốt nhất
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    best_auc = 0
    best_model = None
    best_c = 0
    results = []
    for c in C_values:
        model_tmp = LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=42)
        model_tmp.fit(X_train, y_train)
        y_probs_tmp = model_tmp.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_probs_tmp)
        results.append({'C': c, 'AUC': score})
        if score > best_auc:
            best_auc = score
            best_model = model_tmp
            best_c = c
    return best_model, scaler, X_test, y_test, df_test, features, best_c, best_auc, pd.DataFrame(results)


@st.cache_resource
def train_random_forest(_G, max_edges=20000):
    """Huấn luyện mô hình Random Forest"""
    edges = list(_G.edges())
    if len(edges) > max_edges:
        edges = random.sample(edges, max_edges)
    
    train_edges, test_edges = train_test_split(edges, test_size=0.2, random_state=42)
    G_train = _G.edge_subgraph(train_edges).copy()
    G_train.add_nodes_from(_G.nodes())
    core_nums = nx.core_number(G_train)
    train_neg = smart_negative_sampling_mixed(G_train, train_edges, len(train_edges))
    test_neg = smart_negative_sampling_mixed(G_train, test_edges, len(test_edges))
    df_train_rf = pd.concat([
        extract_features_rf(G_train, train_edges, 1, core_nums),
        extract_features_rf(G_train, train_neg, 0, core_nums)
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_test_rf = pd.concat([
        extract_features_rf(G_train, test_edges, 1, core_nums),
        extract_features_rf(G_train, test_neg, 0, core_nums)
    ]).reset_index(drop=True)
    features_rf = ['Jaccard', 'AA', 'PA', 'RA', 'DegDiff', 'Triangles', 'Clust_Avg', 'KCore_u', 'KCore_v']
    scaler_rf = StandardScaler()
    X_train_rf = scaler_rf.fit_transform(df_train_rf[features_rf])
    y_train_rf = df_train_rf['Label']
    X_test_rf = scaler_rf.transform(df_test_rf[features_rf])
    y_test_rf = df_test_rf['Label']
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=4,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train_rf, y_train_rf)
    y_probs_rf = rf_model.predict_proba(X_test_rf)[:, 1]
    auc_rf = roc_auc_score(y_test_rf, y_probs_rf)
    return rf_model, scaler_rf, X_test_rf, y_test_rf, df_test_rf, features_rf, auc_rf


def render_part5(G):
    """Render giao dien Part 5: Link Prediction Model"""
    
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
    .model-lr { color: #FF6B00; font-weight: 600; }
    .model-rf { color: #2E7D32; font-weight: 600; }
    
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
        <div class="part-title">Link Prediction Model (Machine Learning)</div>
        <div class="part-desc">
            Huấn luyện mô hình Machine Learning để dự đoán liên kết và gợi ý sản phẩm bán kèm
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
        <strong>Mục đích:</strong> Huấn luyện mô hình Machine Learning để dự đoán các liên kết tiềm năng (Link Prediction), 
        từ đó tạo ra gợi ý bán chéo chính xác hơn so với phương pháp Heuristics
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== GIẢI THÍCH PHƯƠNG PHÁP ====================
    st.markdown('<div class="section-title">Phương pháp tiếp cận</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight: 600; color: #FF6B00; margin-bottom: 0.5rem;">Logistic Regression</div>
            <div style="font-size: 0.85rem; color: #666;">Mô hình tuyến tính với Hyperparameter Tuning (C) và StandardScaler</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <div style="font-weight: 600; color: #2E7D32; margin-bottom: 0.5rem;">Random Forest</div>
            <div style="font-size: 0.85rem; color: #666;">Ensemble model với Extended Features (Triangles, Clustering)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="interpretation-box">
        <strong>Mixed Negative Sampling Strategy:</strong><br>
        • <strong>30% Hub-Hub pairs:</strong> Cặp node bậc cao không kết nối<br>
        • <strong>30% Hard negatives:</strong> Cặp node gần nhau nhưng không kết nối<br>
        • <strong>40% Random pairs:</strong> Cặp node ngẫu nhiên không kết nối
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== TABS CHO CÁC MÔ HÌNH ====================
    st.markdown('<div class="section-title">Huấn luyện và đánh giá mô hình</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "Logistic Regression", 
        "Random Forest",
        "So sánh mô hình"
    ])
    
    # ==================== TAB 1: LOGISTIC REGRESSION ====================
    with tab1:
        st.markdown('<div class="section-title">Logistic Regression với Mixed Negative Sampling</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Đặc trưng sử dụng:</strong> Jaccard, Adamic-Adar, Preferential Attachment, Resource Allocation, Degree Difference
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Huấn luyện Logistic Regression", key="train_lr"):
            with st.spinner("Đang huấn luyện mô hình Logistic Regression..."):
                model_lr, scaler_lr, X_test_lr, y_test_lr, df_test_lr, features_lr, best_c, best_auc_lr, tuning_results = train_logistic_regression(G)
            
            st.session_state['model_lr'] = model_lr
            st.session_state['scaler_lr'] = scaler_lr
            st.session_state['X_test_lr'] = X_test_lr
            st.session_state['y_test_lr'] = y_test_lr
            st.session_state['df_test_lr'] = df_test_lr
            st.session_state['features_lr'] = features_lr
            st.session_state['best_c'] = best_c
            st.session_state['best_auc_lr'] = best_auc_lr
            st.session_state['tuning_results'] = tuning_results
        
        if 'model_lr' in st.session_state:
            model_lr = st.session_state['model_lr']
            scaler_lr = st.session_state['scaler_lr']
            X_test_lr = st.session_state['X_test_lr']
            y_test_lr = st.session_state['y_test_lr']
            df_test_lr = st.session_state['df_test_lr']
            features_lr = st.session_state['features_lr']
            best_c = st.session_state['best_c']
            best_auc_lr = st.session_state['best_auc_lr']
            tuning_results = st.session_state['tuning_results']
            
            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value" style="color: #FF6B00;">{best_auc_lr:.4f}</div>
                    <div class="info-label">AUC Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value">{best_c}</div>
                    <div class="info-label">Best C (Regularization)</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value">{len(features_lr)}</div>
                    <div class="info-label">Số đặc trưng</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Kết quả Hyperparameter Tuning:**")
                st.dataframe(tuning_results.style.format({'AUC': '{:.4f}'}).background_gradient(subset=['AUC'], cmap='YlOrRd'), 
                            use_container_width=True, hide_index=True)
                
                st.markdown("**Trọng số mô hình:**")
                weights = pd.DataFrame({
                    'Đặc trưng': features_lr,
                    'Trọng số': model_lr.coef_[0]
                }).sort_values('Trọng số', ascending=False)
                st.dataframe(weights.style.format({'Trọng số': '{:.4f}'}), use_container_width=True, hide_index=True)
            
            with col2:
                # ROC Curve
                y_probs = model_lr.predict_proba(X_test_lr)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_lr, y_probs)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, label=f'LR (AUC={best_auc_lr:.3f})', color='#FF6B00', lw=2)
                ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
                ax.set_title('ROC Curve - Logistic Regression', fontweight='bold', fontsize=13, color='#232F3E')
                ax.set_xlabel('False Positive Rate', fontsize=11)
                ax.set_ylabel('True Positive Rate', fontsize=11)
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
            
            # Metrics @ K
            st.markdown('<div class="section-title">Đánh giá tại Top K</div>', unsafe_allow_html=True)
            eval_df = evaluate_at_k(y_test_lr, y_probs, [10, 50, 100, 200])
            st.dataframe(eval_df.style.format({
                'P@K': '{:.4f}', 'R@K': '{:.4f}', 'MAP@K': '{:.4f}'
            }).background_gradient(subset=['P@K', 'MAP@K'], cmap='Greens'), use_container_width=True, hide_index=True)
            
            # Nhận xét
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>Nhận xét Logistic Regression:</strong><br>
                • Mô hình đạt AUC = <strong>{best_auc_lr:.4f}</strong> với C = {best_c}<br>
                • Đặc trưng quan trọng nhất: <strong>{weights.iloc[0]['Đặc trưng']}</strong> (trọng số = {weights.iloc[0]['Trọng số']:.4f})<br>
                • P@100 = {eval_df[eval_df['K']==100]['P@K'].values[0]:.4f} - Trong Top 100 gợi ý, có {eval_df[eval_df['K']==100]['P@K'].values[0]*100:.1f}% là đúng
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== TAB 2: RANDOM FOREST ====================
    with tab2:
        st.markdown('<div class="section-title">Random Forest với Extended Features</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Đặc trưng mở rộng:</strong> Jaccard, Adamic-Adar, Preferential Attachment, Resource Allocation, 
            Degree Difference, <strong>Triangles</strong>, <strong>Clustering Average</strong>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Huấn luyện Random Forest", key="train_rf"):
            with st.spinner("Đang huấn luyện mô hình Random Forest..."):
                model_rf, scaler_rf, X_test_rf, y_test_rf, df_test_rf, features_rf, auc_rf = train_random_forest(G)
            
            st.session_state['model_rf'] = model_rf
            st.session_state['scaler_rf'] = scaler_rf
            st.session_state['X_test_rf'] = X_test_rf
            st.session_state['y_test_rf'] = y_test_rf
            st.session_state['df_test_rf'] = df_test_rf
            st.session_state['features_rf'] = features_rf
            st.session_state['auc_rf'] = auc_rf
        
        if 'model_rf' in st.session_state:
            model_rf = st.session_state['model_rf']
            X_test_rf = st.session_state['X_test_rf']
            y_test_rf = st.session_state['y_test_rf']
            features_rf = st.session_state['features_rf']
            auc_rf = st.session_state['auc_rf']
            
            # Metrics
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value" style="color: #2E7D32;">{auc_rf:.4f}</div>
                    <div class="info-label">AUC Score</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value">150</div>
                    <div class="info-label">Số cây (n_estimators)</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value">{len(features_rf)}</div>
                    <div class="info-label">Số đặc trưng</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature Importance
                st.markdown("**Feature Importance:**")
                importances = model_rf.feature_importances_
                feat_imp = pd.DataFrame({
                    'Feature': features_rf,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                st.dataframe(feat_imp.style.format({'Importance': '{:.4f}'}).background_gradient(subset=['Importance'], cmap='Greens'), 
                            use_container_width=True, hide_index=True)
                
                fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                colors = ['#2E7D32' if i == 0 else '#66BB6A' for i in range(len(feat_imp))]
                ax_imp.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors)
                ax_imp.set_xlabel('Importance', fontsize=11)
                ax_imp.set_title('Feature Importance (Random Forest)', fontweight='bold', fontsize=13, color='#232F3E')
                ax_imp.invert_yaxis()
                ax_imp.spines['top'].set_visible(False)
                ax_imp.spines['right'].set_visible(False)
                st.pyplot(fig_imp)
            
            with col2:
                # ROC Curve
                y_probs_rf = model_rf.predict_proba(X_test_rf)[:, 1]
                fpr, tpr, _ = roc_curve(y_test_rf, y_probs_rf)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, label=f'RF (AUC={auc_rf:.3f})', color='#2E7D32', lw=2)
                ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
                ax.set_title('ROC Curve - Random Forest', fontweight='bold', fontsize=13, color='#232F3E')
                ax.set_xlabel('False Positive Rate', fontsize=11)
                ax.set_ylabel('True Positive Rate', fontsize=11)
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                st.pyplot(fig)
            
            # Metrics @ K
            st.markdown('<div class="section-title">Đánh giá tại Top K</div>', unsafe_allow_html=True)
            eval_df_rf = evaluate_at_k(y_test_rf, y_probs_rf, [10, 50, 100, 200])
            st.dataframe(eval_df_rf.style.format({
                'P@K': '{:.4f}', 'R@K': '{:.4f}', 'MAP@K': '{:.4f}'
            }).background_gradient(subset=['P@K', 'MAP@K'], cmap='Greens'), use_container_width=True, hide_index=True)
            
            # Nhận xét
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>Nhận xét Random Forest:</strong><br>
                • Mô hình đạt AUC = <strong>{auc_rf:.4f}</strong><br>
                • Đặc trưng quan trọng nhất: <strong>{feat_imp.iloc[0]['Feature']}</strong> (importance = {feat_imp.iloc[0]['Importance']:.4f})<br>
                • Các đặc trưng mở rộng (Triangles, Clust_Avg) giúp mô hình học được cấu trúc cục bộ của mạng
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== TAB 3: SO SÁNH ====================
    with tab3:
        st.markdown('<div class="section-title">So sánh Logistic Regression vs Random Forest</div>', unsafe_allow_html=True)
        
        if 'model_lr' in st.session_state and 'model_rf' in st.session_state:
            best_auc_lr = st.session_state['best_auc_lr']
            auc_rf = st.session_state['auc_rf']
            
            # Metrics comparison
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value" style="color: #FF6B00;">{best_auc_lr:.4f}</div>
                    <div class="info-label">Logistic Regression AUC</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="info-card">
                    <div class="info-value" style="color: #2E7D32;">{auc_rf:.4f}</div>
                    <div class="info-label">Random Forest AUC</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # So sánh ROC
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # LR
            y_probs_lr = st.session_state['model_lr'].predict_proba(st.session_state['X_test_lr'])[:, 1]
            fpr_lr, tpr_lr, _ = roc_curve(st.session_state['y_test_lr'], y_probs_lr)
            ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={best_auc_lr:.3f})', color='#FF6B00', lw=2)
            
            # RF
            y_probs_rf = st.session_state['model_rf'].predict_proba(st.session_state['X_test_rf'])[:, 1]
            fpr_rf, tpr_rf, _ = roc_curve(st.session_state['y_test_rf'], y_probs_rf)
            ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={auc_rf:.3f})', color='#2E7D32', lw=2)
            
            ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
            ax.set_xlabel('False Positive Rate', fontsize=11)
            ax.set_ylabel('True Positive Rate', fontsize=11)
            ax.set_title('So sánh ROC Curve: Logistic Regression vs Random Forest', fontweight='bold', fontsize=13, color='#232F3E')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            st.pyplot(fig)
            
            # Nhận xét
            if auc_rf > best_auc_lr:
                winner = "Random Forest"
                winner_color = "#2E7D32"
                improvement = (auc_rf - best_auc_lr) * 100
            else:
                winner = "Logistic Regression"
                winner_color = "#FF6B00"
                improvement = (best_auc_lr - auc_rf) * 100
            
            st.markdown(f"""
            <div class="interpretation-box">
                <strong>Kết luận:</strong><br>
                • <strong style="color: {winner_color};">{winner}</strong> cho kết quả tốt hơn với mức cải thiện <strong>{improvement:.2f}%</strong> AUC<br>
                • Logistic Regression: Đơn giản, nhanh, dễ giải thích trọng số<br>
                • Random Forest: Phức tạp hơn nhưng có thể học được quan hệ phi tuyến<br>
                • <strong>Khuyến nghị:</strong> Sử dụng mô hình tốt hơn để tạo gợi ý bán chéo trong Part 6
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="highlight-box">
                ⚠️ <strong>Lưu ý:</strong> Vui lòng huấn luyện cả hai mô hình (Logistic Regression và Random Forest) trước khi so sánh.
            </div>
            """, unsafe_allow_html=True)
    
    return st.session_state.get('model_lr'), st.session_state.get('scaler_lr')
