# import 할것들
import numpy as np
from scipy.io import loadmat
from sympy import isprime
from enc_func import *
import matplotlib.pyplot as plt   # ✅ 플로팅용

# 파라미터 생성
Ts = 1  # 샘플링타임 1초
env = Params()  # 환경 변수 
sk = Seret_key(env) # 비밀키

# print("q 소수: ", isprime(env.q))

# 오프라인 행렬들 준비
# 양자화 s 파라미터는 10000으로 고정
offline = np.load("offline_mats.npz", allow_pickle=True)

F_bar = offline["F_bar"]          # 24x24
G_bar = offline["G_bar"]          # 24x6
H_bar = offline["H_bar"]          # 60x24
T1_all = offline["T1_all"]        # 60x23x24
T2_all = offline["T2_all"]        # 60x1x24
V1_all = offline["V1_all"]        # 60x24x23
V2_all = offline["V2_all"]        # 60x24x1
S_xi_all = offline["S_xi_all"]    # 60x23x23
S_v_all  = offline["S_v_all"]     # 60x23x6
Psi_all  = offline["Psi_all"]     # 60x1x23
Sigma_all = offline["Sigma_all"]  # 60x1x6
Sigma_pinv_all = offline["Sigma_pinv_all"]  # 60x6x1 (dtype=object)




print("\nF_bar dtype:", F_bar.dtype)
print("G_bar:", G_bar.dtype)
print("H_bar:", H_bar.dtype)

print("T1 dtype:", T1_all.dtype)
print("T2 dtype:", T2_all.dtype)
print("V1 dtype:", V1_all.dtype)
print("V2 dtype:", V2_all.dtype)
print("S_xi dtype:", S_xi_all.dtype)
print("S_v dtype:", S_v_all.dtype)
print("Psi dtype:", Psi_all.dtype)
print("Sigma dtype:", Sigma_all.dtype)
print("Sigma_pinv_all dtype:", Sigma_pinv_all.dtype)






# 이산화된 플랜트 A B C 행렬
A = np.array([
    [ 0.572915, 0.222492, 0.294165, 0.228264, 0.132920, 0.268409 ],
    [ 0.790746, 0.417171, 4.709138, 0.134381, -5.499884, -0.054966 ],
    [ 0.294165, 0.228264, 0.411670, 0.262637, 0.294165, 0.228264 ],
    [ 4.709138, 0.134381, -9.418275, 0.227824, 4.709138, 0.134381 ],
    [ 0.132920, 0.268409, 0.294165, 0.228264, 0.572915, 0.222492 ],
    [ -5.499884, -0.054966, 4.709138, 0.134381, 0.790746, 0.417171 ]
], dtype=np.float64)

B = np.array([ 13.613318, 22.249166, 13.301577, 22.826353, 13.204555, 26.840867 ], dtype=np.float64)  # shape (6,)

C = np.array([
    [ 1, 0, 0, 0, 0, 0 ],
    [ 0, 0, 1, 0, 0, 0 ],
    [ 0, 0, 0, 0, 1, 0 ],
    [ 1, 0, -1, 0, 0, 0 ],
    [ 0, 0, 1, 0, -1, 0 ]
], dtype=np.int64)

# K 피드백 게인
K = np.array([ -0.010617, -0.010772, -0.009818, -0.010584, -0.007183, -0.011472 ], dtype=np.float64)  # shape (6,)

# Phi_pinv_bar 
Phi_pinv_bar = np.array([
    [  658,   -921,   896,   912,  -2327,   3662,   717,   940,  1166,  2102,   170,    83,   622,   715,   359,   930,   608,   352,   759,  1560,   375,   587,   644,  1172 ],
    [ -8296,  17959, -12313, 18278, 12766, -29386, -8885, 19007,  -764, -11054, 11395, -32194, 31836, -24889, -28947, 41807, -7821, -22733, -23988, -18332, -31832,  4224, -40846, -52861 ],
    [   338,     52,   390,   662,   -16,   2102,  1372,   398,  -114,  4380,   338,    52,   390,   662,   -16,  2102,  -692,   140,   684, -2278,   692,  -140,  -684,  2278 ],
    [    20,  10665, -11625,  6312, 17121,  -1127, 25681, -37446, 12641, 35682,    20, 10665, -11625,  6312, 17121, -1127, -25641, 28819, -18060, -36808, 25641, -28819, 18060, 36808 ],
    [   170,     83,   622,   715,   359,   930,   717,   940,  1166,  2102,   658,   -921,   896,   912, -2327,  3662,  -375,  -587,  -644, -1172,  -608,  -352,  -759, -1560 ],
    [ 11395, -32194, 31836, -24889, -28947, 41807, -8885, 19007,  -764, -11054, -8296,  17959, -12313, 18278, 12766, -29386, 31832,  -4224, 40846,  52861,  7821,  22733, 23988, 18332 ]
], dtype=np.int64)


# 초기값 설정
iter = 100
n_channels = 60 # j_index
execution_times = []  # 실행 시간을 저장할 리스트

# 양자화 파라미터
r_quant = 10000
s_quant = 10000

# 초기값들
xp0 = np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1]])          # 플랜트 초기값
z_hat0 = np.full((24, 1), 0.1, dtype=float)                         # z_hat 초기값
attack_arr = np.zeros(iter)   # 각 k에서 주입한 공격 신호 저장

xp = [xp0]
u = []
y = []

# 상태추정 리스트 (각 원소: 6x1)
x_hat_list = []

# residue: 각 채널 r_j(k)의 실수 복원값을 저장: (iter, 60)
residue_real_mat = np.zeros((iter, n_channels))

# 초기 양자화된 옵저버 상태 (모든 채널에서 동일 초기 z_hat0 사용)
z_hat_bar = np.round(z_hat0 * r_quant * s_quant).astype(object)

# 채널별 암호화 상태 Z_hat_j, 마스킹 상태 b_xi_j 초기화
Z_hat_list = []   # 길이 60, 각 원소는 24 x (N+2)
b_xi_list = []    # 길이 60, 각 원소는 23 x 1

for j in range(n_channels):
    T1_j = T1_all[j]           # 23x24
    T2_j = T2_all[j]           # 1x24
    V2_j = V2_all[j]           # 24x1
    Z_hat_j, b_xi_j = Enc_state(z_hat_bar, sk, env, T1_j, T2_j, V2_j)
    Z_hat_list.append(Z_hat_j)
    b_xi_list.append(b_xi_j)



######### 디버그용

# ==== Phi_pinv_bar 기반 테스트 ====
# 1) 암호화 전: z_hat_bar 에 Phi_pinv_bar 곱한 결과
X_plain_q = Phi_pinv_bar @ z_hat_bar        # 6 x 1 (정수 스케일)

# 2) 암호화 후: Z_hat_list[0] 에 Phi_pinv_bar 곱한 뒤 복호화
X_cipher = Mod(Phi_pinv_bar @ Z_hat_list[0], env.q)  # 6 x (N+2)
X_dec_q  = Dec(X_cipher, sk, env)                    # 6 x 1

# 3) 둘 다 s_quant 로 스케일 다운해서 비교
X_plain = np.asarray(X_plain_q, dtype=float) / (s_quant * s_quant * r_quant)
X_dec   = np.asarray(X_dec_q,  dtype=float) / (s_quant * s_quant * r_quant * env.L)

print("\nX_cipher first column:\n", X_cipher[:, [0]])

print("X_plain (before enc, scaled by s_quant):")
print(X_plain)

print("\nX_dec (after enc+dec, scaled by s_quant):")
print(X_dec)

print("\nmax |X_plain - X_dec| =", np.max(np.abs(X_plain - X_dec)))


######### 추가 디버그: z = [1,...,1]^T 테스트 #########

# 1) 테스트용 z_test: 24x1, 모든 원소 1.0
z_test = np.ones((24, 1), dtype=float)

# 2) 양자화: z_test_bar = round(z_test * r_quant * s_quant)
z_test_bar = np.round(z_test * r_quant * s_quant).astype(int)   # 24x1

# 3) Enc_state로 암호화 (채널 0의 T1,T2,V2 사용)
T1_0 = T1_all[0]
T2_0 = T2_all[0]
V2_0 = V2_all[0]

Z_hat_test, b_xi_test = Enc_state(z_test_bar, sk, env, T1_0, T2_0, V2_0)
# Z_hat_test: 24 x (N+2)

# 4) 암호화 전: Phi_pinv_bar @ z_test_bar
X_plain2_q = Mod(Phi_pinv_bar @ z_test_bar, env.q)           # 6 x 1

# 5) 암호화 후: Phi_pinv_bar @ Z_hat_test → Dec
X_cipher2   = Mod(Phi_pinv_bar @ Z_hat_test, env.q)  # 6 x (N+2)

innL_X_cipher2 = Mod(X_cipher2 * env.invL, env.q) 


X_dec2_q    = Dec(innL_X_cipher2, sk, env)                # 6 x 1
# X_dec2_q    = Dec(X_cipher2, sk, env) 

# 6) 스케일 복원 (z_test_bar 만들 때 곱해준 r*s*s로 나눔)
X_plain2 = np.asarray(X_plain2_q, dtype=float) / (r_quant * s_quant * s_quant)
X_dec2   = np.asarray(X_dec2_q,  dtype=float) / (r_quant * s_quant * s_quant)

print("\n===== z_test = [1,...,1]^T 기반 테스트 =====")
print("X_plain2 (before enc, scaled back):")
print(X_plain2)

print("\nX_dec2 (after enc+dec, scaled back):")
print(X_dec2)

print("\nmax |X_plain2 - X_dec2| =", np.max(np.abs(X_plain2 - X_dec2)))

#### Enc_t 테스트

######### Enc_t 선형성 테스트: v = [1,...,1]^T, S는 6x6 정수 행렬 #########

# 1) 테스트용 v_test: 6x1, 모든 원소 1.0
v_test = np.ones((6, 1), dtype=float)

# 2) 양자화 (스케일은 네가 실제 [u;y]에 쓰는 걸로 맞춰)
#    z_test에서는 r_quant * s_quant * s_quant 을 썼는데,
#    v_test 에서는 일단 r_quant * s_quant 만 썼다고 가정
v_test_bar = np.round(v_test * r_quant * s_quant).astype(int)   # 6x1

# 3) 동적 마스킹 초기값 (위 Enc_state 테스트에서 나온 b_xi_test를 재사용)
b_xi0 = b_xi_test.copy()    # 23x1

# 4) Enc_t 로 암호화
C_t, b_v_test = Enc_t(v_test_bar, sk, b_xi0, Sigma_pinv_all[1], Sigma_all[1], Psi_all[1], env)
# C_t : 6 x (N+2)

# 5) 평문에서 임의의 6x6 정수 행렬 S를 곱한 결과 (참값)
S = np.random.randint(-5, 6, size=(6, 6), dtype=int)    # 예: -5 ~ 5
w_plain_q = S @ v_test_bar     # 6x1 (still integer, quantized)
w_plain   = w_plain_q.astype(float) / (r_quant * s_quant)

# 6) 암호문에 같은 S를 곱한 후 복호화
#    구조는 [c0 | Av | b_prime] 이라서, 왼쪽에서 S를 곱해도
#    각 row 마다 여전히 같은 구조라 Dec 가 그대로 먹힘
W_cipher = Mod(S @ C_t, env.q)           # 6 x (N+2)

# 7) L^{-1} 스케일링 후 복호화 (Dec가 바로 w를 내도록)
innL_W_cipher = Mod(W_cipher * env.invL, env.q)   # element-wise * invL
w_dec_q  = Dec(innL_W_cipher, sk, env)           # 6 x 1

# 8) 양자화 스케일 되돌리기
w_dec = np.asarray(w_dec_q, dtype=float) / (r_quant * s_quant)

print("\n===== Enc_t 선형성 테스트 (w = S @ v) =====")
print("w_plain (before enc, scaled back):")
print(w_plain)

print("\nw_dec (after enc + matmul + dec, scaled back):")
print(w_dec)

print("\nmax |w_plain - w_dec| =", np.max(np.abs(w_plain - w_dec)))

# ==================
#  Simulation loop
# ==================
attack_arr = np.zeros(iter)   # 공격 크기 기록 용 (원하면)

attack_start = iter // 2      # 절반 시점부터 공격 시작

for k in range(iter):

    # 1) 플랜트 출력 y_k = C x_k
    y_k = C @ xp[-1]          # 5x1, float64
    y.append(y_k)

    # 2) 피드백 제어 입력 u_k = K x_k
    u_k = float(K @ xp[-1])   # 스칼라
    u.append(u_k)

    # 3) v = [u; y] (6x1, float)
    v = np.vstack([
        np.array([[u_k]]),  # 1x1
        y_k                 # 5x1
    ])  # -> 6x1, float64

    # === 공격 신호 (실수 영역에서 6x1 벡터로 정의) ===
    # 4번째 성분만 사용, 나머지는 0
    if k >= attack_start:
        # 예시: 점점 커지면서 진동하는 공격 (원하면 식은 바꿔도 됨)
        alpha = (k - attack_start) / attack_start   # 0 → 1 정도 스케일
        alpha = 0
        attack_scalar = alpha * np.sin(k - attack_start)
    else:
        attack_scalar = 0.0

    attack_vec_real = np.zeros((6, 1), dtype=float)
    attack_vec_real[3, 0] = attack_scalar   # 4번째 성분에만 공격

    attack_arr[k] = attack_scalar           # 로그용

    # === 여기까지는 평문 영역, v에는 공격 안 넣음! ===

    # 4) 양자화 (평문 입력에 대한 v_bar)
    v_bar = np.round(v * r_quant).astype(int)  # 6x1

    # 5) 각 인덱스에 대하여 r_j, Z_hat_j, 동적 마스킹 파트 업데이트
    for j in range(n_channels):
        # (a) 이 채널의 오프라인 행렬들
        H_j = H_bar[j, :].reshape(1, 24)              # 1x24
        S_xi_j = S_xi_all[j]                          # 23x23
        S_v_j  = S_v_all[j]                           # 23x6
        Psi_j  = Psi_all[j]                           # 1x23
        Sigma_j = Sigma_all[j]                        # 1x6
        Sigma_pinv_j = Sigma_pinv_all[j]              # 6x1

        Z_hat_j = Z_hat_list[j]                       # 24 x (N+2)
        b_xi_j  = b_xi_list[j]                        # 23 x 1

        # (b) residue R_j = H_j @ Z_hat_j (1 x (N+2))
        R_bar_j = Mod(H_j @ Z_hat_j, env.q)           # 1 x (N+2)

        # 첫 번째 항만 잔차로 사용 (스칼라, mod q 정수로 정리)
        r_j = Mod(R_bar_j[0, 0], env.q)              # Python int, mod q
        
        test = Dec(R_bar_j, sk, env)

        # print("r_j", r_j)
        # print("test", test)
        # --- 디버깅 출력 1: 원래 r_j (mod q) ---
        # print(f"[k={k}, j={j}] R_bar_j[0,0] (mod q) = {r_j}")

        # r_j / L (mod q) = r_j * L^{-1} (mod q)
        r_j_divL = Mod(r_j * env.invL ,env.q)              # 여전히 정수 (mod q)

        # --- 디버깅 출력 2: r_j * L^{-1} (mod q) ---
        # print(f"[k={k}, j={j}] r_j * L^(-1) (mod q) = {r_j_divL}")
        # print("L * invL (mod q) =", Mod(env.L * env.invL, env.q))
        # 이제 r_quant * s_quant^2 로 나눠서 실수로 스케일
        r_j_real_test = float(r_j_divL) / (r_quant * s_quant * s_quant)
        r_j_real = float(r_j) / (r_quant * s_quant * s_quant * env.L)
        # print("r_j_real", r_j_real)
        residue_real_mat[k, j] = r_j_real
        #  print("Z_hat_j", Z_hat_j)
        # (c) Enc_t로 출력 암호화 (채널별 S_xi_j, S_v_j, ...)
        V_j, b_v_j = Enc_t(v_bar, sk, b_xi_j, Sigma_pinv_j, Sigma_j, Psi_j, env)
        # V_j: 6 x (N+2), b_v_j: 6 x 1

        # === (d) 암호문 위에 공격 신호를 추가 ===
        #   - 실수 공격벡터 attack_vec_real (6x1)을
        #   - env.L, r_quant로 스케일 후 정수로 양자화
        #   - 첫 번째 열에 더해 줌 (모듈러 q)
        attack_vec_q = np.round(attack_vec_real * env.L * r_quant).astype(object)  # 6x1

        # V_j의 첫 열 shape: (6,), attack_vec_q: (6,1) → reshape 맞춰서 더하기
        col0 = V_j[:, 0].reshape(6, 1)                       # 6x1
        col0_attacked = Mod(col0 + attack_vec_q, env.q)      # 6x1 (mod q)
        V_j[:, 0] = col0_attacked.reshape(6,)                # 다시 1D로 넣기

        # (d) 암호화된 옵저버 상태 업데이트: Z_hat_{k+1}^j
        Z_hat_j_next = Mod(F_bar @ Z_hat_j + G_bar @ V_j, env.q)

        # (e) 마스킹 상태 업데이트: b_xi_{k+1}^j = S_xi_j b_xi_j + S_v_j b_v_j
        b_xi_j_next = S_xi_j @ b_xi_j + S_v_j @ b_v_j
        b_xi_j_next = Mod(b_xi_j_next, env.q)

        # 업데이트 반영
        Z_hat_list[j] = Z_hat_j_next
        b_xi_list[j]  = b_xi_j_next

    # 5-1) z(t) \in 24x1 로 상태 추정
    # Phi_pinv_bar  : 6 x 24  -> X_hat_cipher : 6 x (N+2)
    Z_hat_ref = Z_hat_list[1]  # 하나 가져오기 (Z_hat은 마스킹 파트만 다르고 메세지는 같음)
    X_hat_cipher = Mod(Phi_pinv_bar @ Z_hat_ref, env.q)  # 6 x (N+2)

    # 복호화
    x_hat_int = Dec(X_hat_cipher, sk, env)
    x_hat_list.append(x_hat_int/ (r_quant * s_quant * s_quant *env.L))

    # 6) 플랜트 상태 업데이트: x_{k+1} = A x_k + B u_k
    xp_next = A @ xp[-1] + B.reshape(-1, 1) * u_k   # B: (6,) -> (6,1)
    xp.append(xp_next)


    




# ==========================
#   시뮬레이션 결과 플롯
# ==========================

# 시간축
t_x = np.arange(len(xp))        # 상태는 k=0..iter → 길이 iter+1
t_u = np.arange(len(u))         # u, y, residue는 k=0..iter-1 → 길이 iter

# 상태 xp: 리스트(각각 6x1)를 6x(iter+1)로 쌓기
xp_arr = np.hstack(xp)          # 6 x (iter+1)

# x_hat: 6 x iter
x_hat_arr = np.hstack(x_hat_list)  # 6 x iter

# 입력 u: (iter,)
u_arr = np.array(u)             # (iter,)

# 출력 y: 5x(iter)
y_arr = np.hstack(y)            # 5 x iter


# 1) 상태 xp vs x_hat (6개 상태) - 같은 k 범위(iter)에서 비교
t_k = np.arange(iter)  # 0..iter-1

fig, axes = plt.subplots(6, 1, figsize=(8, 10), sharex=True)

for i_state in range(6):
    ax = axes[i_state]
    ax.plot(t_k, xp_arr[i_state, :iter], label=f"x[{i_state}] (true)")
    ax.plot(t_k, x_hat_arr[i_state, :], '--', label=f"x_hat[{i_state}] (est)")
    ax.set_ylabel(f"x{i_state+1}")
    ax.grid(True)
    ax.legend(loc="best", fontsize=8)

    if i_state == 0:
        ax.set_title("Plant States xp vs Estimated States x_hat_[1, 2, 3]")

axes[-1].set_xlabel("time")

plt.tight_layout()

# 2) 공격 신호 (attack)
plt.figure(figsize=(8, 4))
plt.plot(t_u, attack_arr)
plt.xlabel("time")
plt.ylabel("attack")
plt.title("Injected Attack on sensor 3 (e.g. y[2])")
plt.grid(True)

# 3) 출력 y (5개 출력)
plt.figure(figsize=(8, 5))
for i_out in range(5):
    plt.plot(t_u, y_arr[i_out, :], label=f"y[{i_out}]")
plt.xlabel("time")
plt.ylabel("y")
plt.title("Outputs y")
plt.legend()
plt.grid(True)

# r: 60x1 벡터의 inf norm (시간마다 하나씩)
# residue_real_mat : (iter, 60)  라고 가정 (각 행이 시간 k에서의 r^T)

# 각 시간 k마다 전체 60 채널에 대한 ∞-norm 계산
r_inf = np.max(np.abs(residue_real_mat), axis=1)  # shape: (iter,)

# 플롯
plt.figure(figsize=(8, 4))
plt.plot(t_u, r_inf)
plt.title("Residual ∞-norm ||r||∞ (60 channels)")
plt.xlabel("time")
plt.ylabel("||r||∞")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.tight_layout(rect=[0, 0, 1, 0.96])  # suptitle 안 가리게 여백 조정
plt.show()

