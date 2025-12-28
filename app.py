"""
Amazon Network Analysis - Streamlit App
Phân tích Cấu trúc Lõi-Ngoại vi & Gợi ý Bán chéo
"""

import streamlit as st
import pandas as pd
import numpy as np

# ======================================================
# Lazy imports (KHÔNG cache để luôn nhận code mới khi Rerun)
# ======================================================
def get_home_module():
    from modules.part0_home import render_home
    return render_home

def get_part1_module():
    from modules.part1_data_loading import load_data, render_part1, get_category_graph
    return load_data, render_part1, get_category_graph

def get_part2_module():
    from modules.part2_network_stats import render_part2
    return render_part2

def get_part3_module():
    from modules.part3_core_periphery import render_part3, compute_core_periphery
    return render_part3, compute_core_periphery

def get_part4_module():
    from modules.part4_cross_sell import render_part4
    return render_part4

def get_part5_module():
    from modules.part5_link_prediction import render_part5
    return render_part5

def get_part6_module():
    from modules.part6_case_study import render_part6
    return render_part6


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Amazon Network Analysis",
    page_icon="https://www.amazon.com/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """
### Thông tin nhóm Jet2Holiday

**Môn học:** Mạng Xã Hội - UIT

---

**Thành viên:**

| STT | Họ và Tên | MSSV |
|-----|-----------|------|
| 1 | Đào Trung Hiếu | 21520430 |
| 2 | Nguyễn Bá Hưng | 21520512 |
| 3 |Nguyễn Thị trinh | 21521539 |
| 4 | Trần Đức Hùng | 21520525 |

---

© 2025 Jet2Holiday - UIT
        """
    }
)

# ======================================================
# SAFE AMAZON THEME CSS (NO DATAFRAME TEXT OVERRIDE)
# ======================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --amazon-orange: #FF9900;
    --amazon-dark: #232F3E;
    --amazon-light: #F7F8F8;
}

html, body, .stApp {
    background-color: #FFFFFF !important;
    color: var(--amazon-dark) !important;
    font-family: 'Inter', sans-serif;
}

/* Header */
header[data-testid="stHeader"] {
    background-color: #FFFFFF !important;
    border-bottom: 1px solid #DDD;
}

/* Hide Rerun option in main menu */
[data-testid="stMainMenu"] ul li:has(span:contains("Rerun")),
[data-testid="stMainMenu"] [role="option"]:first-child {
    display: none !important;
}

/* Hide keyboard shortcut in Clear cache menu item */
[data-testid="stMainMenu"] ul li span[data-testid="stMarkdownContainer"] + span,
[data-testid="stMainMenu"] [role="option"] kbd,
[data-testid="stMainMenu"] [role="option"] span:last-child:not(:first-child) {
    display: none !important;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: var(--amazon-light) !important;
}

/* Headings */
h1 { font-size: 2.2rem; font-weight: 700; color: var(--amazon-dark); }
h2 { font-size: 1.6rem; font-weight: 600; color: var(--amazon-dark); }
h3 { font-size: 1.3rem; font-weight: 600; color: var(--amazon-dark); }

/* Buttons */
.stButton > button {
    background-color: var(--amazon-orange);
    color: var(--amazon-dark);
    font-weight: 600;
    border-radius: 8px;
    border: none;
}
.stButton > button:hover {
    background-color: #FEBD69;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: var(--amazon-orange);
    font-weight: 700;
}

/* SAFE DataFrame styling (NO text color override) */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] table {
    background-color: #FFFFFF !important;
}

[data-testid="stDataFrame"] thead th {
    background-color: var(--amazon-light) !important;
    font-weight: 600;
}

/* st.table styling */
[data-testid="stTable"] th {
    background-color: var(--amazon-orange);
    font-weight: 600;
}

[data-testid="stTable"] td {
    background-color: #FFFFFF;
}

/* Alerts */
.stAlert {
    border-left: 4px solid var(--amazon-orange);
    background-color: #FFF8E7;
}

/* ====== NAVIGATION LINK MENU STYLING ====== */
.nav-menu {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 0;
    margin: 0;
}

.nav-link {
    display: block;
    padding: 12px 16px;
    text-decoration: none !important;
    color: var(--amazon-dark) !important;
    font-size: 15px;
    font-weight: 500;
    border-radius: 8px;
    border-left: 3px solid transparent;
    transition: all 0.2s ease;
    cursor: pointer;
}

.nav-link:hover {
    background-color: rgba(255, 153, 0, 0.1);
    border-left: 3px solid var(--amazon-orange);
    color: var(--amazon-orange) !important;
}

.nav-link.active {
    background-color: rgba(255, 153, 0, 0.15);
    border-left: 3px solid var(--amazon-orange);
    color: var(--amazon-orange) !important;
    font-weight: 600;
}

/* Menu title */
.menu-title {
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #DDD;
}

/* Custom Radio Button Styling - Sidebar Navigation */
[data-testid="stSidebar"] .stRadio > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: #232F3E !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Radio group container */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 4px !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Each radio option label - width đồng nhất */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label {
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    border-radius: 6px !important;
    transition: all 0.25s ease !important;
    color: #232F3E !important;
    cursor: pointer !important;
    margin: 2px 0 !important;
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* Ẩn radio circle icon - chỉ ẩn cái nút tròn */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label > div:first-child {
    display: none !important;
}

/* Text inside radio - hiển thị đậm */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label p {
    font-weight: 600 !important;
    color: #232F3E !important;
    margin: 0 !important;
    transition: all 0.25s ease !important;
}

/* Hover effect */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover {
    background-color: rgba(255, 153, 0, 0.15) !important;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:hover p {
    color: #FF9900 !important;
}

/* Selected/Checked state - multiple selectors for compatibility */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-selected="true"],
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[aria-checked="true"],
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:has(input[type="radio"]:checked) {
    background-color: rgba(255, 153, 0, 0.2) !important;
    border-left: 3px solid #FF9900 !important;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] p,
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-selected="true"] p,
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[aria-checked="true"] p,
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label:has(input[type="radio"]:checked) p {
    color: #FF9900 !important;
    font-weight: 700 !important;
}

/* Fallback: style based on svg fill for checked state */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label svg[fill="currentColor"] ~ div p {
    color: #FF9900 !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stRadio [role="radiogroup"] > label[data-checked="true"] p {
    color: #FF9900 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ======================================================
# SIDEBAR NAVIGATION (RADIO BUTTONS - STABLE)
# ======================================================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg",
    width=150
)


# Menu items: key = query param, label = text hiển thị
menu_items = [
    ("home", "Trang chủ"),
    ("data_loading", "Data Loading"),
    ("network_stats", "Network Statistics"),
    ("core_periphery", "Core-Periphery"),
    ("cross_sell", "Cross-Sell (Heuristics)"),
    ("link_prediction", "Link Prediction (ML)"),
    ("case_study", "Case Study"),
]

# Tạo dict để map
menu_keys = [k for k, _ in menu_items]
menu_labels = [label for _, label in menu_items]
key_to_label = {k: label for k, label in menu_items}
label_to_key = {label: k for k, label in menu_items}

# Lấy current page từ session state (stable across reruns)
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

current_page = st.session_state["current_page"]

# Validate current page
if current_page not in menu_keys:
    current_page = "home"
    st.session_state["current_page"] = current_page

# Tìm index hiện tại
current_index = menu_keys.index(current_page) if current_page in menu_keys else 0

# Radio button navigation
st.sidebar.markdown('<div class="menu-title">Navigation</div>', unsafe_allow_html=True)

selected_label = st.sidebar.radio(
    label="Navigation",
    options=menu_labels,
    index=current_index,
    key="nav_radio",
    label_visibility="collapsed"
)

# Update session state khi user chọn menu khác
selected_key = label_to_key.get(selected_label, "home")
if selected_key != st.session_state["current_page"]:
    st.session_state["current_page"] = selected_key
    st.rerun()

current_page = st.session_state["current_page"]
# Footer
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 Jet2Holiday - UIT")

# Map to options variable (dùng cùng key với các phần bên dưới)
options = current_page

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_resource
def initialize_data():
    load_data, _, _ = get_part1_module()
    return load_data()

G_full, communities = initialize_data()


# ======================================================
# SYNC G_CATEGORY FROM SELECTED_CAT_IDX
# Đảm bảo G_category luôn được tính khi có selected_cat_idx
# (quan trọng khi chuyển trang vì Part 1 không render)
# ======================================================
def sync_category_graph():
    """Đồng bộ G_category từ selected_cat_idx nếu cần"""
    selected_idx = st.session_state.get("selected_cat_idx", None)
    
    if selected_idx is not None:
        # Kiểm tra xem G_category hiện tại có đúng với selected_idx không
        current_cat_for_graph = st.session_state.get("_current_cat_for_graph", None)
        
        if current_cat_for_graph != selected_idx:
            # Category đã thay đổi - cần tính lại G_category
            _, _, get_category_graph = get_part1_module()
            G_category = get_category_graph(G_full, communities, selected_idx)
            st.session_state["G_category"] = G_category
            st.session_state["_current_cat_for_graph"] = selected_idx
            
            # Clear các kết quả phụ thuộc vào G_category cũ
            # để các Part khác sẽ tính lại với G_category mới
            keys_to_clear = [
                "network_stats",      # Part 2
                "node_classes",       # Part 3
                "core_nums",          # Part 3
                "max_k",              # Part 3
                "model",              # Part 5
                "scaler",             # Part 5
                "df_test_lr",         # Part 5
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]

# Gọi sync mỗi khi app chạy
sync_category_graph()


# ======================================================
# HOME
# ======================================================
if options == "home":
    render_home = get_home_module()
    render_home()


# ======================================================
# PART 1
# ======================================================
elif options == "data_loading":
    _, render_part1, _ = get_part1_module()
    render_part1(G_full, communities)
    # G_category va selected_cat_idx duoc luu trong session_state boi Part 1


# ======================================================
# PART 2
# ======================================================
elif options == "network_stats":
    # Uu tien su dung G_category da luu tu Part 1
    G = st.session_state.get("G_category", None)
    
    # Neu chua co G_category, tinh lai tu selected_cat_idx
    if G is None:
        selected_idx = st.session_state.get("selected_cat_idx", None)
        if selected_idx is not None:
            _, _, get_category_graph = get_part1_module()
            G = get_category_graph(G_full, communities, selected_idx)
            st.session_state["G_category"] = G
        else:
            st.warning("Chưa chọn danh mục ở Part 1. Đang sử dụng đồ thị gốc.")
            G = G_full

    render_part2 = get_part2_module()
    st.session_state["network_stats"] = render_part2(G)


# ======================================================
# PART 3
# ======================================================
elif options == "core_periphery":
    G = st.session_state.get("G_category", G_full)
    render_part3, _ = get_part3_module()
    node_classes, core_nums, max_k = render_part3(G)

    st.session_state["node_classes"] = node_classes
    st.session_state["core_nums"] = core_nums
    st.session_state["max_k"] = max_k


# ======================================================
# PART 4
# ======================================================
elif options == "cross_sell":
    G = st.session_state.get("G_category", G_full)
    node_classes = st.session_state.get("node_classes")

    if node_classes is None:
        _, compute = get_part3_module()
        node_classes, _, _ = compute(G)

    render_part4 = get_part4_module()
    render_part4(G, node_classes)


# ======================================================
# PART 5
# ======================================================
elif options == "link_prediction":
    G = st.session_state.get("G_category", G_full)
    render_part5 = get_part5_module()

    model, scaler = render_part5(G)
    if model:
        st.session_state["model"] = model
        st.session_state["scaler"] = scaler


# ======================================================
# PART 6
# ======================================================
elif options == "case_study":
    G = st.session_state.get("G_category", G_full)
    node_classes = st.session_state.get("node_classes")
    model = st.session_state.get("model")
    scaler = st.session_state.get("scaler")
    df_test = st.session_state.get("df_test_lr")

    render_part6 = get_part6_module()
    # Truyền G_full để tạo gợi ý từ toàn bộ mạng (theo notebook)
    render_part6(G, node_classes, model, scaler, df_test, G_full=G_full)
