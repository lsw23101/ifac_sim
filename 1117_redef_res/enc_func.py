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
        self.r = 19          # 오류 범위 # 균등분포
        self.N = 5            # 키 차원 # 연산량...
        # 2^64 근처 소수
        self.q = self.p * self.L - 59    # 18446744073709551557

env = Params()

def Seret_key(env):
    # -1 0 1 균등 분포
    sk = np.array([random.choice([-1, 0, 1]) for _ in range(env.N)], dtype=object)
    # print(sk)
    return sk.reshape(-1, 1)  # N x 1 형태로 반환

def Mod(x, p):
    # -q/2 q/2 인 mod 연산
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
    Enc_state, j 인덱스에 따라 60개

    입력:
        z_hat_bar : (24x1) 양자화된 초기 옵저버 상태
        sk        : (N x 1) 비밀키
        env       : Params()
        T1        : 23x24 
        T2        : 1x24  
        V2        : 24x1  

    출력:
        C_state   : 24 x (N+2) 암호문
        b_xi_ini  : 23 x 1 Enc_t 를 위한 동적 마스킹 파트
    """
    n = 24
    N = env.N

    z_hat_bar = np.asarray(z_hat_bar, dtype=object).reshape(n, 1)
    sk = np.asarray(sk, dtype=object).reshape(N, 1)

    # 1) A, e 
    A = np.random.randint(-10, 11, size=(n, N)).astype(object)  # 24xN # -10부터 10까지의 랜덤 값으로 (24xN) 배열 생성
    e = np.random.randint(-env.r, env.r + 1, size=(n, 1)).astype(object)  # 6x1 

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
    C0 = Mod(env.L * z_hat_bar + b_ini - b_prime, env.q)  # 24x1

    # 6) 최종 암호문
    C_state = np.hstack([C0, A, b_prime])         # 24 x (N+2)

    return C_state, b_xi_ini


def Enc_t(v, sk, b_xi, Sigma_pinv, Sigma, Psi, env):
    """
    Enc_t (동적 암호화)

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
    e = np.random.randint(-env.r, env.r + 1, size=(n_v, 1)).astype(object)  # 6x1              

    # 2) b_v = Av sk + e (6x1)
    b_v = Av @ sk + e
    b_v = Mod(b_v, env.q)

    # 3) b_prime = Sigma_pinv @ (Sigma @ b_v + Psi @ b_xi) (6x1)
    tmp = Sigma @ b_v + Psi @ b_xi   # 1x1
    tmp = Mod(tmp, env.q)
    b_prime = Sigma_pinv @ tmp       # 6x1
    b_prime = Mod(b_prime, env.q)

    # 4) 첫 번째 컬럼: c0 = v + b_v - b_prime (6x1, mod q)
    c0 = Mod(env.L * v + b_v - b_prime, env.q)

    # 5) 최종 암호문: [c0 | Av | b_prime] (6 x (N+2))
    C_t = np.hstack([c0, Av, b_prime])

    # 디버깅용 출력 (원하면 주석처리)
    # print("[Enc_t] C_t shape:", C_t.shape)

    return C_t, b_v


def Dec(ciphertext, sk, env):
    
    dec_sk = np.vstack([
        np.ones((1, 1), dtype=object),
        -sk,
        np.ones((1, 1), dtype=object)
    ])
    
    # 1) 모듈러 곱 (0..q-1)
    m_bar = Mod(ciphertext@ dec_sk, env.q)  # (h x 1)

    # 2) centered 로 다시 잡기 [-q/2, q/2)
    m_center = Mod(m_bar, env.q)   # 여기서 Mod는 네가 정의한 centered 함수
    
    L = env.L

    # 수정 필요
    m_dec = np.vectorize(
        lambda x: int(round(int(x) / L)), 
        otypes=[object]
    )(m_center)

    # 3) 스케일링은 호출하는 쪽에서 알아서 (L, r_quant, s_quant 등)
    return m_dec


# 상대차수가 모두 1 => H 행벡터 가져오고 나머지 T1은 표준기저로 설정
# 역행렬 V도 표준기저와 20번째 행이 Zq의 역원으로 계산
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
    # 4) V = T^{-1} (mod q) 
    # -------------------------
    V = np.zeros((n, n), dtype=object)

    # print("T", T)
    # print("V", V)

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

