import streamlit as st
import numpy as np

# Thiết lập giao diện
st.set_page_config(page_title="Simplex Solver - Huỳnh Phong", layout="wide")
st.title("🧮 Trình giải thuật toán Đơn hình (Simplex Method)")
st.caption("Phát triển bởi Huỳnh Phong - Khoa Toán Tin học")

# --- CÁC HÀM LOGIC ---
def print_dictionary_st(tableau, basis, n, m, title):
    st.subheader(title)
    
    # 1. Xử lý hàm mục tiêu Z
    z_rhs = -tableau[m, -1]
    z_terms = []
    for j in range(n + m):
        if j not in basis:
            coef = tableau[m, j]
            if abs(coef) > 1e-9:
                var_name = f"x_{{{j+1}}}" if j < n else f"W_{{{j-n+1}}}"
                sign = "+" if coef > 0 else "-"
                z_terms.append(f"{sign} {abs(coef):.3g}{var_name}")
    
    z_expr = " ".join(z_terms)
    
    if abs(z_rhs) < 1e-9:
        if z_expr.startswith("+ "): 
            z_expr = z_expr[2:] 
        z_line = f"Z &= {z_expr if z_expr else '0'}"
    else:
        z_line = f"Z &= {z_rhs:.3g} {z_expr}"
        
    latex_lines = [z_line]

    # 2. Xử lý các dòng ràng buộc
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
        
        expr = " ".join(terms)
        
        if abs(b_val) < 1e-9:
            if expr.startswith("+ "): 
                expr = expr[2:]
            line = f"{var_name} &= {expr if expr else '0'}"
        else:
            line = f"{var_name} &= {b_val:.3g} {expr}"
            
        latex_lines.append(line)

    # 3. Gom tất cả vào môi trường aligned của LaTeX
    latex_str = "\\begin{aligned}\n" + " \\\\\n".join(latex_lines) + "\n\\end{aligned}"
    st.latex(latex_str)

# --- GIAO DIỆN NHẬP LIỆU ---
st.write("### 1. Nhập hệ số bài toán")
st.info("Mặc định bài toán là Min Z và các ràng buộc là dấu ≤. Chương trình sẽ tự động nhận diện số biến và số ràng buộc.")

col_c, col_a, col_b = st.columns([1, 2, 1])

with col_c:
    c_raw = st.text_input("Hệ số Min(Z) (cách nhau dấu cách)", "3 -2")

with col_a:
    a_raw = st.text_area("Ma trận A (dòng cách nhau bởi ';')", "1 1\n2 1") # Hỗ trợ nhập xuống dòng hoặc dấu chấm phẩy

with col_b:
    b_raw = st.text_input("Vectơ vế phải b (cách nhau bởi ';')", "4; 5")

# --- THỰC THI GIẢI ---
if st.button("Bắt đầu giải thuật"):
    try:
        # Chuyển đổi dữ liệu chuỗi sang numpy array
        c = np.array([float(x) for x in c_raw.split()])
        b = np.array([float(x) for x in b_raw.replace('\n', ';').split(';') if x.strip()])
        
        # Xử lý ma trận A (chấp nhận cả dấu ';' hoặc xuống dòng)
        a_rows_raw = a_raw.replace('\n', ';').split(';')
        rows = [r.strip() for r in a_rows_raw if r.strip()]
        A = np.array([[float(x) for x in r.split()] for r in rows])

        # TỰ ĐỘNG NHẬN DIỆN KÍCH THƯỚC
        n = len(c)
        m = len(b)

        # KIỂM TRA ĐIỀU KIỆN (VALIDATION)
        if A.shape != (m, n):
            st.error(f"🚨 **Lỗi kích thước ma trận!**\n- Phát hiện **{n}** biến quyết định (từ hàm Z).\n- Phát hiện **{m}** ràng buộc (từ vectơ b).\n- Nhưng ma trận A lại có kích thước {A.shape} (cần là ({m}, {n})).\n\nVui lòng kiểm tra lại hệ số bạn vừa nhập.")
        else:
            st.success(f"✅ Đã tự động nhận diện: **{n}** biến quyết định và **{m}** ràng buộc.")
            
            # Khởi tạo Tableau
            tableau = np.zeros((m + 1, n + m + 1))
            tableau[:m, :n] = A
            tableau[:m, n:n+m] = np.eye(m)
            tableau[:m, -1] = b
            tableau[m, :n] = c 
            basis = list(range(n, n + m)) 
            
            rule = 'bland' if np.any(b == 0) else 'dantzig'
            st.info(f"Kích hoạt quy tắc xoay: **{rule.upper()}**")

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
                
                # Phép xoay Gauss
                pivot_val = tableau[row_out, col_in]
                tableau[row_out, :] /= pivot_val
                for i in range(m + 1):
                    if i != row_out:
                        tableau[i, :] -= tableau[i, col_in] * tableau[row_out, :]
                
                basis[row_out] = col_in
                
                # Xác định tên biến vào/ra để in ra màn hình
                var_in_name = f"x_{col_in+1}" if col_in < n else f"W_{col_in-n+1}"
                var_out_name = f"x_{basis[row_out]+1}" if basis[row_out] < n else f"W_{basis[row_out]-n+1}"
                
                print_dictionary_st(tableau, basis, n, m, f"Bước lặp {iteration} (Vào: ${var_in_name}$ | Ra: ${var_out_name}$)")
                iteration += 1

            # Trích xuất kết quả cuối cùng
            st.divider()
            st.write("## 🏆 Kết quả tối ưu")
            st.metric("Giá trị Min(Z)", f"{-tableau[m, -1]:.4g}")
            
            sol_cols = st.columns(n)
            for i in range(n):
                val = tableau[np.where(np.array(basis) == i)[0][0], -1] if i in basis else 0
                sol_cols[i].latex(f"x_{{{i+1}}} = {val:.4g}")

    except ValueError:
        st.error("🚨 Dữ liệu nhập vào chứa ký tự không hợp lệ. Vui lòng chỉ nhập số, dấu cách và dấu chấm phẩy (;).")
    except Exception as e:
        st.error(f"🚨 Đã xảy ra lỗi hệ thống: {e}")
