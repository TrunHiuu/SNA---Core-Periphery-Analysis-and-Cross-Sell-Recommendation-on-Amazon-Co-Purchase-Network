"""
PART 0: TRANG CHỦ
Mục đích:
- Hiển thị thông tin tổng quan về đề tài
- Giới thiệu vấn đề nghiên cứu
- Pipeline thực hiện
- Mô tả bộ dữ liệu
"""

import streamlit as st


def render_home():
    """
    Render trang chủ với thông tin tổng quan về đề tài nghiên cứu
    """
    
    # ==================== CUSTOM CSS ====================
    st.markdown("""
    <style>
    .hero-section {
        background: linear-gradient(135deg, #232F3E 0%, #37475A 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-title {
        color: #FF9900;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.3;
    }
    .hero-subtitle {
        color: #FFFFFF;
        font-size: 1rem;
        font-style: italic;
        opacity: 0.9;
    }
    .card {
        background: #FFFFFF;
        border: 1px solid #E8E8E8;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .card-title {
        color: #232F3E;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #FF9900;
        padding-bottom: 0.5rem;
    }
    .card-text {
        color: #555;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .highlight-box {
        background: linear-gradient(90deg, #FFF8E7 0%, #FFFFFF 100%);
        border-left: 4px solid #FF9900;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .section-header {
        color: #232F3E;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .section-header::before {
        content: '';
        width: 4px;
        height: 24px;
        background: #FF9900;
        border-radius: 2px;
    }
    .data-card {
        background: #F7F8F8;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E0E0E0;
    }
    .data-title {
        color: #FF9900;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.6rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ==================== HERO SECTION ====================
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">
            PHÂN TÍCH CẤU TRÚC LÕI-NGOẠI VI VÀ GỢI Ý BÁN CHÉO<br>
            TRÊN MẠNG LƯỚI ĐỒNG MUA HÀNG AMAZON
        </div>
        <div class="hero-subtitle">
            Core-Periphery Analysis and Cross-Sell Recommendation on Amazon Co-Purchase Network
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ==================== GIỚI THIỆU VẤN ĐỀ ====================
    st.markdown('<div class="section-header">Giới thiệu vấn đề</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Motivation</div>
            <div class="card-text">
                Thương mại điện tử phát triển mạnh, hệ thống gợi ý sản phẩm trở thành yếu tố 
                then chốt. Mạng đồng mua hàng Amazon chứa thông tin quý giá về hành vi khách hàng 
                cần được khai thác.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Problem Statement</div>
            <div class="card-text">
                Làm sao khai thác cấu trúc mạng để gợi ý bán chéo hiệu quả? 
                Các nghiên cứu trước chưa kết hợp mô hình Core-Periphery với Machine Learning 
                trong bài toán này.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Objectives</div>
            <div class="card-text">
                Phân tích cấu trúc mạng, phân loại sản phẩm Core/Periphery, 
                xây dựng heuristics bán chéo và mô hình Link Prediction bằng ML.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ==================== PIPELINE ====================
    st.markdown('<div class="section-header">Pipeline thực hiện</div>', unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 4, 1])
    with col_center:
        st.image("pipeline.drawio.png", use_container_width=True)

    # ==================== DATASET ====================
    st.markdown('<div class="section-header">Bộ dữ liệu</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
        <strong>Amazon Product Co-purchasing Network</strong> từ Stanford SNAP - 
        mạng lưới sản phẩm với quan hệ "Frequently Bought Together"
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="data-card">
            <div class="data-title">com-amazon.ungraph.txt</div>
            <div class="card-text">
                File chứa danh sách cạnh của đồ thị vô hướng, mỗi dòng biểu diễn một cạnh 
                giữa hai sản phẩm thường được mua cùng nhau.
                <br><br>
                <b>Nodes:</b> Sản phẩm Amazon<br>
                <b>Edges:</b> Quan hệ "Frequently Bought Together"<br>
                <b>Type:</b> Undirected Graph
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="data-card">
            <div class="data-title">com-amazon.all.dedup.cmty.txt</div>
            <div class="card-text">
                File chứa thông tin về các cộng đồng (communities/categories) sản phẩm, 
                mỗi dòng là một tập các node thuộc cùng một cộng đồng.
                <br><br>
                <b>Communities:</b> Nhóm sản phẩm cùng danh mục<br>
                <b>Format:</b> Mỗi dòng là một community<br>
                <b>Usage:</b> Phân tích theo category
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.caption("Nguồn: Stanford SNAP - snap.stanford.edu/data/com-Amazon.html")
