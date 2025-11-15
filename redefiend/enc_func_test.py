import numpy as np
import random


# 이 파일이 해야하는 일:
# 비밀키 만들기
# 모듈러, 암호화 (초기값, 출력), 복호화 (24길이 혹은 길이 1 짜리 둘 다 호환 되도록)


class Params:
    def __init__(self):
        # 여기서 K의 최댓값은 355,700,000 (참고용)
        self.p = int(2**54)   # p 
        self.L = int(2**10)   # L 
        self.r = 10           # 오류 범위
        self.N = 5            # 키 차원 
        # 2^64 근처 소수
        self.q = self.p * self.L - 59    # 18446744073709551557

        
# class Params:
#     def __init__(self):
#         self.p = int(2**118)  # p 
#         self.L = int(2**10)  # L 
#         self.r = 10         # 오류 범위
#         self.N = 4    # 키 차원 
#         self.q = self.p * self.L -159 # 근처 소수




env = Params()

def Seret_key(env):
    # sk를 N 크기만큼 -1, 0, 1 중에서 각각 랜덤으로 선택
    sk = np.array([random.choice([-1, 0, 1]) for _ in range(env.N)], dtype=object)
   
   # print(sk)

    return sk.reshape(-1, 1)  # N x 1 형태로 반환

def Mod(x, p):
    """
    Centered modular:
    x (스칼라 또는 배열)을 정수로 보고 mod p 한 뒤,
    결과를 [-p/2, p/2] 범위로 옮겨줌.
    모든 연산은 파이썬 int 기반.
    """
    x_arr = np.asarray(x, dtype=object)

    def centered(v):
        v_int = int(v)
        r = v_int % p
        if r >= p // 2:
            r -= p
        return r

    return np.vectorize(centered, otypes=[object])(x_arr)



def Enc_state(z_hat_bar, sk, env, T1, T2, V2):
    """
    Enc_state (테스트 버전)

    입력:
        z_hat_bar : (24x1) 양자화된 초기 옵저버 상태
        sk        : (N x 1) 비밀키
        env       : Params()
        T1        : 23x24 (현재 채널용)
        T2        : 1x24  (현재 채널용)
        V2        : 24x1  (현재 채널용)

    출력:
        C_state   : 24 x (N+2) 암호문
        b_xi_ini  : 23 x 1
    """
    n = 24
    N = env.N

    z_hat_bar = np.asarray(z_hat_bar, dtype=object).reshape(n, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)

    # 1) A, e (테스트용)
    A = np.random.randint(-10, 11, size=(n, N)).astype(object)  # 24xN # -10부터 10까지의 랜덤 값으로 (24xN) 배열 생성
    e = np.zeros((n, 1), dtype=object)     # 24x1
    
    # print ("A @ sk", A @ sk)


    # 2) b_ini = A sk + e
    b_ini = A @ sk + e
    b_ini = Mod(b_ini, env.q)

    # 3) b_tilde, b_xi_ini
    b_tilde = T2 @ b_ini           # 1x1
    b_tilde = Mod(b_tilde, env.q)

    b_xi_ini = T1 @ b_ini          # 23x1
    b_xi_ini = Mod(b_xi_ini, env.q)

    # 4) b_prime = V2 @ b_tilde
    b_prime = V2 @ b_tilde         # 24x1
    b_prime = Mod(b_prime, env.q)

    # 5) 첫 번째 컬럼
    C0 = Mod(z_hat_bar + b_ini - b_prime, env.q)  # 24x1

    # 6) 최종 암호문
    C_state = np.hstack([C0, A, b_prime])         # 24 x (N+2)

    # print("[Enc_state] C_state shape:", C_state.shape)
    # print("[Enc_state] b_xi_ini shape:", b_xi_ini.shape)

    return C_state, b_xi_ini


def Enc_t(v, sk, b_xi, S_xi, S_v, Sigma_pinv, Sigma, Psi, env):
    """
    Enc_t (동적 암호화, 테스트 버전)

    입력:
        v           : 6x1 벡터 (양자화된 [u;y])
        sk          : N x 1 비밀키
        b_xi        : 23 x 1 (마스킹 상태, 업데이트는 함수 밖에서)
        S_xi, S_v   : 23x23, 23x6 (지금은 함수 안에서 사용 X, 추후 b_xi 업데이트용)
        Sigma_pinv  : 6 x 1
        Sigma       : 1 x 6
        Psi         : 1 x 23
        env         : Params()

    출력:
        C_t         : 6 x (N+2) 암호문 [v + b_v - b_prime | Av | b_prime]
    """
    n_v = 6
    N = env.N

    # 형 맞추기
    v = np.asarray(v, dtype=object).reshape(n_v, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)
    b_xi = np.asarray(b_xi, dtype=object).reshape(23, 1)
    Sigma = np.asarray(Sigma, dtype=object).reshape(1, 6)
    Sigma_pinv = np.asarray(Sigma_pinv, dtype=object).reshape(6, 1)
    Psi = np.asarray(Psi, dtype=object).reshape(1, 23)

    # 1) Av, e 생성
    Av = np.random.randint(-10, 11, size=(n_v, N)).astype(object)  # 6xN, -10~10
    e = np.zeros((n_v, 1), dtype=object)                           # 6x1

    # 2) b_v = Av sk + e (6x1)
    b_v = Av @ sk + e
    b_v = Mod(b_v, env.q)

    # 3) b_prime = Sigma_pinv @ (Sigma @ b_v + Psi @ b_xi) (6x1)
    tmp = Sigma @ b_v + Psi @ b_xi   # 1x1
    tmp = Mod(tmp, env.q)
    b_prime = Sigma_pinv @ tmp       # 6x1
    b_prime = Mod(b_prime, env.q)

    # 4) 첫 번째 컬럼: c0 = v + b_v - b_prime (6x1, mod q)
    c0 = Mod(v + b_v - b_prime, env.q)

    # 5) 최종 암호문: [c0 | Av | b_prime] (6 x (N+2))
    C_t = np.hstack([c0, Av, b_prime])

    # 디버깅용 출력 (원하면 주석처리)
    # print("[Enc_t] C_t shape:", C_t.shape)

    return C_t, b_v


def Dec(ciphertext, sk, env):
    h, N_plus_2 = ciphertext.shape
    N = env.N
    
    sk_neg = -sk
    one_sk_one = np.vstack([
        np.ones((1, 1), dtype=object),
        sk_neg,
        np.ones((1, 1), dtype=object)
    ])
    
    # 1) 모듈러 곱 (0..q-1)
    m_bar = matmul_mod(ciphertext, one_sk_one, env.q)  # (h x 1)

    # 2) centered 로 다시 잡기 [-q/2, q/2)
    m_center = Mod(m_bar, env.q)   # 여기서 Mod는 네가 정의한 centered 함수

    # 3) 스케일링은 호출하는 쪽에서 알아서 (L, r_quant, s_quant 등)
    return m_center


# 이거 안쓸거같은데
def matmul_mod(A, B, q):
    """Z_q 위에서의 행렬 곱: C = A @ B (mod q)"""
    n, m = A.shape
    m2, k = B.shape
    assert m == m2
    C = np.zeros((n, k), dtype=object)
    for i in range(n):
        for j in range(k):
            s = 0
            for l in range(m):
                s += int(A[i, l]) * int(B[l, j])
            C[i, j] = s % q
    return C




def build_TV(H1, q):
    """
    입력:  
        H1 : 1×24 numpy object vector  
        q  : modulus

    출력:  
        T1, T2, T, V, V1, V2

    설계:
      - pivot column = 20번째 열 (0-based index 19)
      - T1 : 23x24, 20열만 제외한 나머지 23개 열에 대한 선택행렬
      - T2 = H1
      - T  = [T1; T2]
      - V  = T^{-1} (mod q) 를 closed-form으로 직접 구성
      - V  = [V1 V2] with V1: 24x23, V2: 24x1
    """

    H1 = np.asarray(H1, dtype=object).reshape(24,)
    n = 24
    pivot = 19  # 0-based index, = 20번째 열

    # -------------------------
    # 1) T1 만들기 : "pivot 열만 제외한 표준 기저"
    # -------------------------
    T1 = np.zeros((n - 1, n), dtype=object)  # 23x24
    cols = list(range(n))
    cols.remove(pivot)   # [0,1,...,18,20,21,22,23]
    for row, col in enumerate(cols):
        T1[row, col] = 1

    # 2) T2 = H1
    T2 = H1.reshape(1, n)

    # 3) T = [T1; T2]
    T = np.vstack([T1, T2])

    # print("[build_TV] T1 shape:", T1.shape)
    # print("[build_TV] T  shape:", T.shape)

    # -------------------------
    # 4) V = T^{-1} (mod q) 를 직접 구성
    #    일반 패턴 (5x5 예제의 확장판)
    # -------------------------
    V = np.zeros((n, n), dtype=object)

    # pivot 값 및 역원
    h_p = int(H1[pivot]) % q
    if h_p == 0:
        raise ValueError("H1[19] (20번째 원소) ≡ 0 (mod q) 입니다. pivot으로 사용할 수 없습니다.")
    inv_h_p = pow(h_p, -1, q)


    # 앞의 n-1개 열 (1~23열) 채우기
    #
    # 열 j (0 <= j < n-1) 는 "원래 인덱스 orig_idx"에 대응:
    #   j < pivot  ->  orig_idx = j
    #   j >= pivot ->  orig_idx = j + 1   (pivot 열을 건너뛴 것)
    #
    # 행 i != pivot:
    #   orig_idx == i 이면 V[i,j] = 1  (항등)
    #   아니면 0
    #
    # 행 i == pivot:
    #   V[pivot, j] = - H1[orig_idx] / H1[pivot]  (mod q)
    #
    for j in range(n - 1):  # 0..22
        orig_idx = j if j < pivot else j + 1  # 0-based
        h_orig = int(H1[orig_idx]) % q

        for i in range(n):
            if i == pivot:
                # pivot 행: -h_orig / h_p (mod q)
                V[i, j] = (-h_orig * inv_h_p) % q
            else:
                # identity mapping
                if i == orig_idx:
                    V[i, j] = 1
                else:
                    V[i, j] = 0

    # 마지막 열 (23, 즉 24번째 열):
    #   모든 행 0, 단 pivot 행에만 1/h_p
    for i in range(n):
        V[i, n - 1] = 0
    V[pivot, n - 1] = inv_h_p % q

    # print("inverse test:", Mod(T@V, env.q))
    # -------------------------
    # 5) V1, V2 분리
    # -------------------------
    V1 = V[:, :n - 1].copy()          # 24x23
    V2 = V[:, n - 1].reshape(n, 1)    # 24x1

    return T1, T2, T, V, V1, V2


if __name__ == "__main__":
    # ===== 테스트 파라미터 / 키 생성 =====
    env = Params()
    sk = Seret_key(env)

    print("=== Test: Enc_state + Dec round trip ===")
    print("q =", env.q)
    print("L =", env.L)
    print("N =", env.N)
    print("sk^T =", sk.reshape(1, -1))

    # ===== 1. 원래 메시지 m (24x1, 대충 1 근처 실수) =====
    np.random.seed(0)  # 재현 가능하도록
    m = 1.0 + 0.1 * np.random.randn(24, 1)   # 평균 1, 표준편차 0.1
    print("\n[1] Original m (24x1, real values near 1):")
    print(m)

    # ===== 2. 양자화: z_hat_bar = round(m * 10000) =====
    # (여기서는 env.L 대신 10000을 양자화 스케일로 사용)
    z_hat_bar = np.round(m * 10000).astype(object)   # 24x1, 정수
    print("\n[2] Quantized z_hat_bar = round(m * 10000):")
    print(z_hat_bar)

    # ===== 3. T1, T2, V2 만드는 H1 생성 (pivot 열 19만 1, 나머지 0) =====
    H1 = np.zeros(24, dtype=object)
    H1[19] = 1  # pivot 자리만 1로, build_TV에서 역원 존재하도록
    T1, T2, T, V, V1, V2 = build_TV(H1, env.q)

    # ===== 4. Enc_state로 z_hat_bar 암호화 =====
    C_state, b_xi_ini = Enc_state(z_hat_bar, sk, env, T1, T2, V2)

    print("\n[3] Enc_state output:")
    print("C_state shape:", C_state.shape)   # 기대: (24, N+2)
    print("b_xi_ini shape:", b_xi_ini.shape) # 기대: (23, 1)

    # ===== 5. Dec로 복호화 (원래 벡터 m 복원 테스트) =====
    # Dec는 내부에서 / env.L 를 하고 있으니, 여기서는 추가로 /10000 해서 다시 실수 스케일로
    m_dec = Dec(C_state, sk, env) / 10000  # shape: (24, 1) (object)

    print("\n[4] Dec(C_state) 결과 (m_dec):")
    print(m_dec)

    # float 배열로 변환해서 원래 m과 비교
    m_dec_float = np.asarray(m_dec, dtype=float)

    # ===== 6. 에러 확인 (벡터 전체) =====
    diff = m - m_dec_float
    max_err = np.max(np.abs(diff))

    print("\n[5] m vs m_dec 비교 (vector round-trip):")
    print("max |m - m_dec| =", max_err)
    print("m (첫 5개):")
    print(m[:5])
    print("m_dec (첫 5개):")
    print(m_dec_float[:5])

    if max_err < 1e-2:
        print("\n✅ Round-trip OK: 복호화 결과가 원래 m과 거의 동일합니다.")
    else:
        print("\n⚠ Round-trip WARNING: 에러가 다소 큽니다. 스케일링 또는 설계 다시 확인 필요.")

    # ============================================================
    # 7. 추가 테스트: H = [1 1 ... 1]로 선형결합 후 복호화 비교
    #    - w_plain  = H * z_hat_bar / 10000
    #    - w_dec    = Dec( H * C_state ) / 10000
    # ============================================================

    # 1x24 행벡터 H = [1 1 ... 1]
    H_row = np.ones((1, 24), dtype=object)

    # (a) 평문 쪽: z_hat_bar 에 H_row 곱한 결과
    #     z_hat_bar 는 이미 10000 * m 이므로, 여기서 /10000 해서 실수 스케일로 맞추기
    w_plain_q = H_row @ z_hat_bar          # 1x1, 정수 스케일 (10000 * 합(m_i))
    w_plain = np.asarray(w_plain_q, dtype=float) / 10000.0

    print("\n[6] w_plain = H * z_hat_bar / 10000:")
    print("w_plain (scalar) =", w_plain)

    # (b) 암호문 쪽: C_state 에 H_row 곱한 결과를 다시 복호화
    #     C_state_lin = H * C_state (1 x (N+2)), 이후 mod q
    C_state_lin = Mod(H_row @ C_state, env.q)  # 1 x (N+2)

    # 복호화 후 /10000
    w_dec = Dec(C_state_lin, sk, env) / 10000.0  # 1x1
    w_dec_float = float(np.asarray(w_dec, dtype=float).reshape(-1)[0])

    print("\n[7] w_dec = Dec( H * C_state ) / 10000:")
    print("w_dec (scalar) =", w_dec_float)

    # (c) 차이 확인
    lin_err = np.abs(w_plain - w_dec_float)

    print("\n[8] 선형결합 전/후 비교 (H = [1..1]):")
    print("w_plain =", w_plain)
    print("w_dec   =", w_dec_float)
    print("abs(w_plain - w_dec) =", lin_err)

    if lin_err < 1e-2:
        print("\n✅ Linear OK: H를 곱한 후 복호화해도 원래 합과 거의 동일합니다.")
    else:
        print("\n⚠ Linear WARNING: H를 곱한 후 복호화 결과가 원래 합과 차이가 큽니다.")
