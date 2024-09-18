import numpy as np

# Hàm nhập ma trận từ bàn phím
def input_matrix(rows, cols, name):
    print(f"Nhập dữ liệu đầu vào cần kiểm tra {name} ({rows}x{cols}):")
    matrix = []
    for i in range(rows):
        row = []
        for j in range(cols):
            #value = float(input(f"Nhập phần tử [{i}, {j}]: "))
            value = float(input(f"Nhập dữ liệu x{j+1}: "))
            row.append(value)
        matrix.append(row)
    return np.array(matrix)

NUY = 1
E = 0
K = 4
flag = 0

# Khai báo ma trận x 
x = np.array([[-1, 0, 0],
              [-1, 0, 1],
              [-1, 1, 0],
              [-1, 1, 1]])

# Khai báo vector d 
d = np.array([[1],
              [0],
              [1],
              [1]])

# Khai báo vector w
w = np.array([[-0.1],
              [0.2],
              [-0.3]])

epsilon = 1e-10  # Ngưỡng để so sánh với 0

def train():
    global w, d, E, K, flag  # Sử dụng biến toàn cục
    while True:  # Vòng lặp vô hạn, sẽ dừng khi E = 0
        E = 0  # Đặt lại E ở mỗi vòng lặp
        for k in range(K):
            # Chuyển vị của w 
            wt = w.T
            net = wt @ x[k, :]

            # Làm tròn net đến hàng phần chục
            net_rounded = np.round(net, 1)

            # Kiểm tra giá trị net và gán y
            if abs(net_rounded) < epsilon:  # Nếu net gần bằng 0
                y = 0
            elif net_rounded > 0:  # net dương
                y = 1
            else:  # net âm
                y = 0

            # Cập nhật trọng số w
            w = w + NUY * (d[k] - y) * x[k, :].reshape(-1, 1)

            # Tính toán lỗi E
            E += 0.5 * ((d[k] - y) ** 2)

            # In giá trị w trong từng vòng lặp
            print(f"Vòng lặp {k + 1}:")
            print(f"  d = {d[k].flatten()}, x = {x[k, :]}, net = {net_rounded}, y = {y}, E = {E}")

        # Kiểm tra điều kiện dừng
        if E == 0:  # Nếu E bằng 0, thoát vòng lặp
            break
        flag += 1

    # In kết quả cuối cùng
    print("Giá trị cuối cùng của w:", w.flatten())
    print("Giá trị cuối cùng của E:", E)
    print("Số vòng lặp đã thực hiện:", K)
    print("Số chu kỳ thực hiện:", flag)
    
    # Trả về ma trận w cuối cùng
    return w

# Gọi hàm train() và lấy kết quả ma trận w cuối cùng
final_w = train()

# Sử dụng ma trận w cuối cùng cho các tính toán tiếp theo
print("Ma trận w sau huấn luyện:", final_w)

# Nhập ma trận x mới từ bàn phím
x_new = input_matrix(1, 2, "x mới")  # Nhập ma trận x mới (1 hàng, 2 cột)

# Tính phương trình
z = final_w[1, 0] * x_new[0, 0] + final_w[2, 0] * x_new[0, 1] - final_w[0, 0]
if z > 0:
    y = 1
else:
    y = 0

print("Dữ liệu ngõ ra:", y)
