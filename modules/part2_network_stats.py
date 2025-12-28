"""
PART 2: NETWORK STATISTICS ANALYSIS
Mục đích:
- Degree Distribution: Tính toán và vẽ biểu đồ phân phối bậc
- Clustering Coefficient: Tính hệ số phân cụm trung bình
- Connected Components: Kiểm tra tính liên thông của đồ thị
- Assortativity: Phân tích xu hướng kết nối giữa các node
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


@st.cache_data
def compute_network_stats(_G, _cat_idx=None):
    """Tính toán các thống kê mạng lưới
    
    Args:
        _G: NetworkX graph object
        _cat_idx: Category index để buộc cache invalidate khi thay đổi category
    """
    stats = {}
    
    # Phân phối bậc
    degree_sequence = sorted([d for n, d in _G.degree()], reverse=True)
    degree_counts = Counter(degree_sequence)
    stats['degree_sequence'] = degree_sequence
    stats['degree_counts'] = degree_counts
    stats['avg_degree'] = np.mean(degree_sequence)
    stats['max_degree'] = max(degree_sequence) if degree_sequence else 0
    stats['min_degree'] = min(degree_sequence) if degree_sequence else 0
    
    # Thành phần liên thông
    if _G.is_directed():
        stats['num_components'] = nx.number_weakly_connected_components(_G)
        largest_cc = max(nx.weakly_connected_components(_G), key=len)
    else:
        stats['num_components'] = nx.number_connected_components(_G)
        largest_cc = max(nx.connected_components(_G), key=len)
    
    stats['largest_cc_size'] = len(largest_cc)
    stats['largest_cc_ratio'] = len(largest_cc) / _G.number_of_nodes()
    
    # Hệ số phân cụm
    stats['avg_clustering'] = nx.average_clustering(_G)
    
    # Assortativity
    stats['degree_assortativity'] = nx.degree_assortativity_coefficient(_G)
    
    return stats


@st.cache_data  
def get_top_degree_clustering(_G, top_n=10, _cat_idx=None):
    """Lấy hệ số phân cụm cho các node có bậc cao nhất
    
    Args:
        _G: NetworkX graph object
        top_n: Số lượng node top degree
        _cat_idx: Category index để buộc cache invalidate khi thay đổi category
    """
    top_degree_nodes = [n for n, d in sorted(_G.degree(), key=lambda item: item[1], reverse=True)][:top_n]
    clustering_coefficients = nx.clustering(_G, top_degree_nodes)
    return pd.DataFrame({
        'Node': list(clustering_coefficients.keys()),
        'Degree': [_G.degree(n) for n in clustering_coefficients.keys()],
        'Clustering Coefficient': list(clustering_coefficients.values())
    })


def render_part2(G):
    """Render giao diện Part 2: Network Statistics Analysis"""
    
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
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== HEADER ====================
    st.markdown("""
    <div class="part-header">
        <div class="part-title">Network Statistics Analysis</div>
        <div class="part-desc">
            Phân tích các đặc trưng thống kê của mạng lưới: phân phối bậc, hệ số phân cụm, thành phần liên thông và tính đồng cấu
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== THÔNG TIN ĐỒ THỊ ĐANG PHÂN TÍCH ====================
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    # Lấy thông tin category đã chọn từ session_state
    selected_cat = st.session_state.get('selected_cat_idx', None)
    
    if selected_cat is not None:
        st.success(f"Đang phân tích: Subgraph của Danh mục {selected_cat} ({num_nodes:,} nodes, {num_edges:,} cạnh)")
    else:
        st.info(f"Đang phân tích: Đồ thị gốc ({num_nodes:,} nodes, {num_edges:,} cạnh). Vào Part 1 để chọn danh mục cụ thể.")
    
    # ==================== MỤC ĐÍCH ====================
    st.markdown("""
    <div class="highlight-box">
        <strong>Mục đích:</strong> Tính toán và trực quan hóa các thống kê mạng để hiểu rõ cấu trúc và đặc điểm của đồ thị đồng mua hàng Amazon
    </div>
    """, unsafe_allow_html=True)
    
    # Tính toán thống kê (truyền selected_cat để cache biết khi nào cần tính lại)
    with st.spinner("Đang tính toán thống kê mạng lưới..."):
        stats = compute_network_stats(G, _cat_idx=selected_cat)
    
    # ==================== METRICS CHÍNH ====================
    st.markdown('<div class="section-title">Tổng quan thống kê</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{stats['avg_degree']:.2f}</div>
            <div class="info-label">Bậc trung bình</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{stats['max_degree']:,}</div>
            <div class="info-label">Bậc cao nhất</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{stats['avg_clustering']:.4f}</div>
            <div class="info-label">Hệ số phân cụm TB</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-value">{stats['num_components']:,}</div>
            <div class="info-label">Thành phần liên thông</div>
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TABS CHI TIẾT ====================
    st.markdown('<div class="section-title">Phân tích chi tiết</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Phân phối bậc", 
        "Thành phần liên thông", 
        "Hệ số phân cụm",
        "Assortativity"
    ])
    
    # ==================== TAB 1: PHÂN PHỐI BẬC ====================
    with tab1:
        st.markdown('<div class="section-title">Phân phối bậc (Degree Distribution)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Định nghĩa:</strong> Phân phối bậc cho biết số lượng node có một giá trị bậc nhất định. 
            Đây là đặc trưng quan trọng nhất của mạng lưới.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.spinner("Đang vẽ biểu đồ..."):
                fig, ax = plt.subplots(figsize=(10, 6))
                degree_counts = stats['degree_counts']
                ax.bar(degree_counts.keys(), degree_counts.values(), width=0.8, color='#FF9900', alpha=0.8, edgecolor='#232F3E')
                ax.set_title("Phân phối bậc của các node", fontweight='bold', fontsize=14, color='#232F3E')
                ax.set_xlabel("Bậc", fontsize=11)
                ax.set_ylabel("Số lượng node", fontsize=11)
                ax.set_xlim(0, min(50, max(degree_counts.keys())))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            st.markdown("**Top 15 bậc phổ biến nhất:**")
            top_degrees = pd.DataFrame(
                stats['degree_counts'].most_common(15),
                columns=['Bậc', 'Số node']
            )
            st.dataframe(top_degrees, use_container_width=True, hide_index=True)
        
        # Nhận xét
        st.markdown(f"""
        <div class="interpretation-box">
            <strong>Nhận xét:</strong><br>
            • Bậc trung bình: <strong>{stats['avg_degree']:.2f}</strong> → Mỗi sản phẩm được mua kèm với khoảng <strong>{int(stats['avg_degree'])}</strong> sản phẩm khác<br>
            • Bậc cao nhất: <strong>{stats['max_degree']:,}</strong> → Đây là sản phẩm "hub" có nhiều liên kết nhất, tiềm năng cross-sell cao<br>
            • Phân phối có đuôi dài (long-tail) → Đặc trưng của mạng scale-free, phần lớn sản phẩm có ít liên kết, một số ít sản phẩm là trung tâm kết nối
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 2: THÀNH PHẦN LIÊN THÔNG ====================
    with tab2:
        st.markdown('<div class="section-title">Phân tích thành phần liên thông</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Định nghĩa:</strong> Thành phần liên thông là tập các node có đường đi tới nhau. 
            Mạng có 1 thành phần lớn nghĩa là hầu hết sản phẩm đều liên quan gián tiếp với nhau.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-value">{stats['num_components']:,}</div>
                <div class="info-label">Số thành phần liên thông</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-card">
                <div class="info-value">{stats['largest_cc_size']:,}</div>
                <div class="info-label">Kích thước CC lớn nhất</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-card">
                <div class="info-value">{stats['largest_cc_ratio']:.1%}</div>
                <div class="info-label">Tỷ lệ node trong CC lớn nhất</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            with st.spinner("Đang vẽ biểu đồ..."):
                fig, ax = plt.subplots(figsize=(6, 6))
                other_nodes = G.number_of_nodes() - stats['largest_cc_size']
                
                # Xử lý trường hợp chỉ có 1 thành phần liên thông (100% nodes)
                if other_nodes <= 0:
                    sizes = [stats['largest_cc_size']]
                    labels = ['Thành phần lớn nhất (100%)']
                    colors = ['#FF9900']
                    explode = (0.02,)
                else:
                    sizes = [stats['largest_cc_size'], other_nodes]
                    labels = ['Thành phần lớn nhất', 'Các thành phần khác']
                    colors = ['#FF9900', '#232F3E']
                    explode = (0.02, 0)
                
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, 
                       explode=explode, shadow=False, textprops={'fontsize': 11})
                ax.set_title("Tỷ lệ node trong các thành phần", fontweight='bold', fontsize=13, color='#232F3E')
                st.pyplot(fig)
        
        # Nhận xét
        connectivity = 'cao' if stats['largest_cc_ratio'] > 0.9 else 'trung bình' if stats['largest_cc_ratio'] > 0.5 else 'thấp'
        coverage = 'hầu hết' if stats['largest_cc_ratio'] > 0.9 else 'phần lớn' if stats['largest_cc_ratio'] > 0.5 else 'một phần'
        st.markdown(f"""
        <div class="interpretation-box">
            <strong>Nhận xét:</strong><br>
            • Thành phần liên thông lớn nhất chứa <strong>{stats['largest_cc_ratio']:.1%}</strong> tổng số node<br>
            • Mạng có tính liên thông <strong>{connectivity}</strong><br>
            • Điều này cho thấy <strong>{coverage}</strong> sản phẩm có liên kết gián tiếp với nhau → Có thể tìm đường gợi ý từ sản phẩm này đến sản phẩm khác<br>
            • Ý nghĩa kinh doanh: Khách hàng mua một sản phẩm có thể được gợi ý các sản phẩm trong cùng thành phần liên thông
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 3: HỆ SỐ PHÂN CỤM ====================
    with tab3:
        st.markdown('<div class="section-title">Hệ số phân cụm (Clustering Coefficient)</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Định nghĩa:</strong> Hệ số phân cụm đo lường mức độ các láng giềng của một node cũng kết nối với nhau. 
            Giá trị cao → mạng có nhiều "cliques" (nhóm liên kết chặt chẽ).
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-card" style="max-width: 300px;">
            <div class="info-value">{stats['avg_clustering']:.4f}</div>
            <div class="info-label">Hệ số phân cụm trung bình</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Hệ số phân cụm cho các node có bậc cao nhất:**")
            with st.spinner("Đang tính toán..."):
                df_clustering = get_top_degree_clustering(G, _cat_idx=selected_cat)
            st.dataframe(df_clustering.style.format({
                'Clustering Coefficient': '{:.4f}'
            }), use_container_width=True, hide_index=True)
        
        with col2:
            # Biểu đồ scatter Degree vs Clustering
            with st.spinner("Đang vẽ biểu đồ..."):
                fig, ax = plt.subplots(figsize=(8, 6))
                degrees = [d for n, d in G.degree()]
                clusterings = list(nx.clustering(G).values())
                ax.scatter(degrees, clusterings, alpha=0.4, s=15, color='#FF9900', edgecolors='#232F3E', linewidths=0.3)
                ax.set_xlabel("Bậc", fontsize=11)
                ax.set_ylabel("Hệ số phân cụm", fontsize=11)
                ax.set_title("Mối quan hệ Bậc vs Hệ số phân cụm", fontweight='bold', fontsize=13, color='#232F3E')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.grid(alpha=0.3)
                st.pyplot(fig)
        
        # Nhận xét
        clust_level = '(cao)' if stats['avg_clustering'] > 0.3 else '(trung bình)' if stats['avg_clustering'] > 0.1 else '(thấp)'
        clust_struct = 'phân cụm mạnh' if stats['avg_clustering'] > 0.3 else 'phân cụm vừa phải' if stats['avg_clustering'] > 0.1 else 'phân tán'
        st.markdown(f"""
        <div class="interpretation-box">
            <strong>Nhận xét:</strong><br>
            • Hệ số phân cụm TB = <strong>{stats['avg_clustering']:.4f}</strong> {clust_level}<br>
            • Node có bậc cao thường có hệ số phân cụm thấp hơn (xu hướng nghịch đảo)<br>
            • Mạng có cấu trúc <strong>{clust_struct}</strong><br>
            • Ý nghĩa cross-sell: Các sản phẩm trong cùng "cụm" thường được mua cùng nhau → Gợi ý sản phẩm trong cụm sẽ có tỷ lệ chuyển đổi cao
        </div>
        """, unsafe_allow_html=True)
    
    # ==================== TAB 4: ASSORTATIVITY ====================
    with tab4:
        st.markdown('<div class="section-title">Phân tích Assortativity theo bậc</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="highlight-box">
            <strong>Định nghĩa:</strong> Assortativity đo xu hướng các node kết nối với node có bậc tương tự.<br>
            • <strong>Assortative (+):</strong> Node bậc cao kết nối với node bậc cao<br>
            • <strong>Disassortative (−):</strong> Node bậc cao kết nối với node bậc thấp
        </div>
        """, unsafe_allow_html=True)
        
        assortativity = stats['degree_assortativity']
        
        st.markdown(f"""
        <div class="info-card" style="max-width: 300px;">
            <div class="info-value">{assortativity:.4f}</div>
            <div class="info-label">Hệ số Assortativity theo bậc</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Biểu đồ minh họa
        with st.spinner("Đang vẽ biểu đồ..."):
            fig, ax = plt.subplots(figsize=(10, 2))
            
            # Gradient bar
            gradient = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(gradient, aspect='auto', cmap='RdYlGn', extent=[-1, 1, 0, 1])
            
            # Marker cho vị trí hiện tại
            ax.axvline(x=assortativity, color='#232F3E', linewidth=3, linestyle='-')
            ax.plot(assortativity, 0.5, 'v', markersize=20, color='#232F3E')
            
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_xticks([-1, -0.5, 0, 0.5, 1])
            ax.set_xticklabels(['Disassortative\n(-1)', '-0.5', 'Neutral\n(0)', '+0.5', 'Assortative\n(+1)'])
            ax.set_title(f"Vị trí Assortativity: {assortativity:.4f}", fontweight='bold', fontsize=13, color='#232F3E', pad=10)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Nhận xét dựa trên giá trị
        if assortativity > 0:
            interpretation = """
            <div class="interpretation-box">
                Nhận xét - Đồ thị có tính Assortative (Đồng cấu):<br>
                • Các node có bậc cao có xu hướng kết nối với các node có bậc cao khác<br>
                • Điều này phổ biến trong <strong>mạng xã hội</strong><br>
                • <strong>Ý nghĩa cross-sell:</strong> Sản phẩm phổ biến thường được mua cùng với sản phẩm phổ biến khác → Chiến lược "best-seller bundle"
            </div>
            """
        elif assortativity < 0:
            interpretation = f"""
            <div class="interpretation-box">
                <strong>Nhận xét:</strong><br>
                • <strong>Cấu trúc Phân ly (Disassortative):</strong> Có xu hướng kết nối mạnh giữa các Node bậc cao và Node bậc thấp, thay vì giữa các Hub với nhau<br>
                • Phản ánh mô hình "Sản phẩm chính + Phụ kiện", khách hàng thường mua kèm các món bổ trợ thay vì mua cùng lúc nhiều sản phẩm Best-seller<br>
                • Đúng với ý nghĩa và cấu trúc của mạng đồng mua hàng thực tế
            </div>
            """
        else:
            interpretation = """
            <div class="interpretation-box">
                Nhận xét:<br>
                • Không có xu hướng rõ rệt trong việc kết nối giữa các node có bậc khác nhau<br>
                • Cấu trúc kết nối mang tính ngẫu nhiên hơn<br>
                • <strong>Ý nghĩa cross-sell:</strong> Cần phân tích sâu hơn các đặc trưng khác để xây dựng chiến lược gợi ý phù hợp
            </div>
            """
        
        st.markdown(interpretation, unsafe_allow_html=True)
    
    return stats
