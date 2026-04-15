import streamlit as st
import numpy as np

# Thiết lập giao diện
st.set_page_config(page_title="Simplex Solver - Huỳnh Phong", layout="wide")
st.title("🧮 Trình giải thuật toán Đơn hình (Simplex Method)")
st.caption("Phát triển bởi Huỳnh Phong - Khoa Toán Tin học")

# --- CÁC HÀM LOGIC (GIỮ NGUYÊN VÀ CHỈNH SỬA OUTPUT) ---

def print_dictionary_st(tableau, basis, n, m, title):
    st.subheader(title)
    
    # Hiển thị hàm mục tiêu
    z_rhs = -tableau[m, -1]
    z_terms = []
    for j in range(n + m):
        if j not in basis:
            coef = tableau[m, j]
            if abs(coef) > 1e-9:
                var_name = f"x_{{{j+1}}}" if j < n else f"W_{{{j-n+1}}}"
                sign = "+" if coef > 0 else "-"
                z_terms.append(f"{sign} {abs(coef):.3g}{var_name}")
    
    z_expr = " ".join(z_terms) if z_terms else "+ 0"
    if z_expr.startswith("+ "): z_expr = z_expr[2:]
    st.latex(f"Z = {z_rhs:.3g} {z_expr}")

    # Hiển thị các ràng buộc
    cols = st.columns(2)
    for i in range(m):
        b_val = tableau[i, -1]
        var_idx = basis[i]
        var_name = f"x_{{{var_idx+1}}}" if var_idx < n else f"W_{{{var_idx-n+1}}}"
        
        terms = []
        for j in range(n + m):
            if j not in basis:
                coef = -tableau[i, j] 
                if abs(coef) > 1e-9:
                    name = f"x_{{{j+1}}}" if j < n else f"W_{{{j-n+1}}}"
                    sign = "+" if coef > 0 else "-"
                    terms.append(f"{sign} {abs(coef):.3g}{name}")
        
        expr = " ".join(terms) if terms else "+ 0"
        if expr.startswith("+ "): expr = expr[2:]
        with cols[i % 2]:
            st.latex(f"{var_name} = {b_val:.3g} {expr}")

# --- GIAO DIỆN NHẬP LIỆU BÊN SIDEBAR ---
st.sidebar.header("Cấu hình bài toán")
n = st.sidebar.number_input("Số biến quyết định (n)", min_value=1, value=2)
m = st.sidebar.number_input("Số ràng buộc (m)", min_value=1, value=2)

st.write("### 1. Nhập hệ số bài toán")
st.info("Mặc định bài toán là Min Z và các ràng buộc là dấu ≤.")

col_c, col_a, col_b = st.columns([1, 2, 1])

with col_c:
    c_raw = st.text_input("Hệ số Min(Z) (cách nhau dấu cách)", "3 -2")
    c = np.array([float(x) for x in c_raw.split()])

with col_a:
    a_raw = st.text_area("Ma trận A (dòng cách nhau bởi ';')", "1 1; 2 1")
    rows = [r.strip() for r in a_raw.split(';') if r.strip()]
    A = np.array([[float(x) for x in r.split()] for r in rows])

with col_b:
    b_raw = st.text_input("Vectơ vế phải b (cách nhau bởi ';')", "4; 5")
    b = np.array([float(x) for x in b_raw.split(';') if x.strip()])

# --- THỰC THI GIẢI ---
if st.button("Bắt đầu giải thuật"):
    if len(c) != n or A.shape != (m, n) or len(b) != m:
        st.error("Kích thước ma trận không khớp với n và m đã chọn!")
    else:
        # Khởi tạo Tableau
        tableau = np.zeros((m + 1, n + m + 1))
        tableau[:m, :n] = A
        tableau[:m, n:n+m] = np.eye(m)
        tableau[:m, -1] = b
        tableau[m, :n] = c 
        basis = list(range(n, n + m)) 
        
        rule = 'bland' if np.any(b == 0) else 'dantzig'
        st.success(f"Sử dụng quy tắc: **{rule.upper()}**")

        print_dictionary_st(tableau, basis, n, m, "Từ điển khởi tạo")

        iteration = 1
        max_iter = 50
        while iteration <= max_iter:
            z_row = tableau[m, :-1] 
            neg_indices = np.where(z_row < -1e-9)[0] 
            
            if len(neg_indices) == 0:
                st.balloons()
                st.write("### ✨ ĐÃ ĐẠT TỐI ƯU!")
                break
                
            col_in = neg_indices[0] if rule == 'bland' else neg_indices[np.argmin(z_row[neg_indices])]
            col_vals = tableau[:m, col_in]
            pos_indices = np.where(col_vals > 1e-9)[0]
            
            if len(pos_indices) == 0:
                st.error("Bài toán không giới hạn (Unbounded).")
                break
                
            ratios = tableau[pos_indices, -1] / tableau[pos_indices, col_in]
            min_ratio = np.min(ratios)
            candidates = pos_indices[np.where(ratios == min_ratio)[0]]
            row_out = candidates[np.argmin([basis[i] for i in candidates])] if rule == 'bland' else candidates[0]
            
            # Xoay
            pivot_val = tableau[row_out, col_in]
            tableau[row_out, :] /= pivot_val
            for i in range(m + 1):
                if i != row_out:
                    tableau[i, :] -= tableau[i, col_in] * tableau[row_out, :]
            
            basis[row_out] = col_in
            print_dictionary_st(tableau, basis, n, m, f"Bước lặp {iteration}")
            iteration += 1

        # Kết quả cuối cùng
        st.divider()
        st.write("## 🏆 Kết quả tối ưu")
        st.metric("Giá trị Min(Z)", f"{-tableau[m, -1]:.4g}")
        
        sol_cols = st.columns(n)
        for i in range(n):
            val = tableau[np.where(np.array(basis) == i)[0][0], -1] if i in basis else 0
            sol_cols[i].latex(f"x_{{{i+1}}} = {val:.4g}")