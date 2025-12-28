# Phân Tích Core-Periphery và Đề Xuất Cross-Sell trên Mạng Đồng Mua Amazon

## Giới thiệu
Dự án này thực hiện phân tích mạng xã hội trên dữ liệu đồng mua của Amazon, tập trung vào:
- Phân tích cấu trúc core-periphery của mạng
- Đề xuất sản phẩm cross-sell dựa trên cấu trúc mạng
- Thống kê, trực quan hóa và nghiên cứu case study thực tế

## Cấu trúc thư mục
- `app.py`: File chính để chạy ứng dụng
- `modules/`: Chứa các module xử lý, phân tích, trực quan hóa
- `data/`: Chứa dữ liệu mạng Amazon (định dạng txt)
- `Nhom1_Jet2Holiday_Official_Final_Optimized.ipynb`: Notebook phân tích chi tiết

## Hướng dẫn cài đặt
1. **Clone repository:**
   ```bash
   git clone https://github.com/TrunHiuu/SNA---Core-Periphery-Analysis-and-Cross-Sell-Recommendation-on-Amazon-Co-Purchase-Network.git
   ```
2. **Cài đặt môi trường ảo (khuyến nghị):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # hoặc
   source .venv/bin/activate  # Mac/Linux
   ```
3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```
   (Nếu chưa có file requirements.txt, hãy cài đặt các thư viện: networkx, pandas, numpy, matplotlib, streamlit, v.v.)

## Cách chạy ứng dụng
- Chạy ứng dụng web với Streamlit:
  ```bash
  streamlit run app.py
  ```
- Hoặc mở notebook để xem phân tích chi tiết.

## Tác giả
- Nhóm 1 - Mạng Xã Hội, UIT
- Liên hệ: TrunHiuu (https://github.com/TrunHiuu)

## Bản quyền
Dự án phục vụ mục đích học tập và nghiên cứu, không sử dụng cho mục đích thương mại.
