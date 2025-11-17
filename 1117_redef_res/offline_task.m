
clear; clc; close all; format short;

%% 플랜트
J1=0.01; J2=0.01; J3=0.01; k1=1.37; k2=1.37;
b1=0.007; b2=0.007; b3=0.007;


A0=[ 0 1 0 0 0 0;
   -k1/J1 -b1/J1 k1/J1 0 0 0;
    0 0 0 1 0 0;
    k1/J2 0 -(k1+k2)/J2 -b2/J2 k2/J2 0;
    0 0 0 0 0 1;
    0 0 k2/J3 0 -k2/J3 -b3/J3 ];

B0=[0;1/J1;0;0;0;0];

C=[ 1 0 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 1 0;
    1 0 -1 0 0 0;
    0 0 1 0 -1 0 ];

% 센서 5개
C1 = C(1,:);
C2 = C(2,:);
C3 = C(3,:);
C4 = C(4,:);
C5 = C(5,:);

D=zeros(5,1);

Ts = 1.0;
sysd=c2d(ss(A0,B0,C,D),Ts);

A=sysd.A; B=sysd.B; C=sysd.C;

% feedback gain 

Q = eye(6);
R = eye(1);
[~, K, ~] = idare(A,B,Q,R,[],[]);
K = -K
% 
% K = [2.32 0.25 -2.47 0.04 1.70 0.12];
% K = K
% abs(eig(A))
% abs(eig(A+B*K))

% 사이즈
n=6; p=5; m=1;


% 각 센서의 관측성 확인
l1 = rank(obsv(A, C1));
l2 = rank(obsv(A, C2));
l3 = rank(obsv(A, C3));
l4 = rank(obsv(A, C4));
l5 = rank(obsv(A, C5));


%% 칼만 분해

% ========= 0) 준비 =========
% 센서별로 시스템 나누기

Ci_list = {C1; C2; C3; C4; C5};

Phi_obs = cell(5,1);     % 가관측 기저 Φ_i^(obs) (li x n)
Phi_p   = cell(5,1);     % Φ_i^(obs)의 우역행렬 (n x li)
F_raw   = cell(5,1);     % 축소된 F_i (observable 서브시스템)
G_raw   = cell(5,1);
H_raw   = cell(5,1);
li      = zeros(5,1);    % 각 센서별 관측가능 차원(rank)


for i = 1:5
    Ci = Ci_list{i};

    % full observability matrix
    Oi = obsv(A, Ci);
    li(i) = rank(Oi);

    % 앞 li(i)개의 row만 사용 (observability index까지)
    Oki = Oi(1:li(i), :);          % li(i) x n

    % row-space 직교기저 만들기
    [Q,~] = qr(Oki.', 0);          % Q: n x li(i), columns orthonormal

    % observable subspace basis (행기저)
    Phi_i_obs = Q.';               % li(i) x n
    Phi_i_p   = pinv(Phi_i_obs);   % n x li(i), 우역행렬 (Phi_i_obs*Phi_i_p ≈ I_li)

    % observable subsystem (z_i = Phi_i_obs * x)
    F_raw{i} = Phi_i_obs * A * Phi_i_p;   % li x li
    G_raw{i} = Phi_i_obs * B;            % li x m
    H_raw{i} = Ci * Phi_i_p;             % 1  x li

    Phi_obs{i} = Phi_i_obs;
    Phi_p{i}   = Phi_i_p;
end


%% 캐노니컬 폼 
Phi_final = cell(5,1);   % x -> z_i,can 매핑 (Phi_i = T_i * Phi_i^(obs))
F_can     = cell(5,1);   % 관측형 canonical F_i
G_can     = cell(5,1);
H_can     = cell(5,1);
T_obs     = cell(5,1);   % z_i -> z_i,can 좌표변환 행렬 T_i

for i = 1:5
    Fi = F_raw{i};
    Gi = G_raw{i};
    Hi = H_raw{i};

    li_i = size(Fi,1);

    % Fi,Hi에 대한 관측성 행렬
    Oi_small = obsv(Fi, Hi);      % li_i x li_i (Fi,Hi가 observable이면 full-rank)

    % basis = inv(Oi_small) * e_last
    e_last = zeros(li_i,1); 
    e_last(end) = 1;

    basis = Oi_small \ e_last;    % = inv(Oi_small)*e_last

    % inv(T_i) = [basis, Fi*basis, Fi^2*basis, ..., Fi^(li-1)*basis]
    inv_T = zeros(li_i);
    v = basis;
    for k = 1:li_i
        inv_T(:,k) = v;
        v = Fi * v;
    end

    T_i = inv(inv_T);             % 좌표변환 행렬 T_i

    % canonical representation (z_i,can = T_i * z_i)
    F_can{i} = T_i * Fi / T_i;
    G_can{i} = T_i * Gi;
    H_can{i} = Hi / T_i;

    % 최종 Φ_i: x -> z_i,can
    Phi_final{i} = T_i * Phi_obs{i};   % li_i x n
    T_obs{i}     = T_i;
end


%% 정리

Phi1 = Phi_final{1};   F1 = F_can{1};   B1 = G_can{1};   H1 = H_can{1};
Phi2 = Phi_final{2};   F2 = F_can{2};   B2 = G_can{2};   H2 = H_can{2};
Phi3 = Phi_final{3};   F3 = F_can{3};   B3 = G_can{3};   H3 = H_can{3};
Phi4 = Phi_final{4};   F4 = F_can{4};   B4 = G_can{4};   H4 = H_can{4};
Phi5 = Phi_final{5};   F5 = F_can{5};   B5 = G_can{5};   H5 = H_can{5};


%% F1과 F3는 같고, F4와 F5도 같음 (observable subspace가 동일, 2-redundancy)

% F1: l1 x l1 행렬이라고 할 때
L1 = F1(:, end);   % 제일 오른쪽 열 (l1 x 1 벡터)
L2 = F2(:, end);
L3 = F3(:, end);
L4 = F4(:, end);
L5 = F5(:, end);

% Partial systems (8a)
F1_bar = F1 - L1*H1;    G1_bar = [B1  L1];
F2_bar = F2 - L2*H2;    G2_bar = [B2  L2];
F3_bar = F3 - L3*H3;    G3_bar = [B3  L3];
F4_bar = F4 - L4*H4;    G4_bar = [B4  L4];
F5_bar = F5 - L5*H5;    G5_bar = [B5  L5];



G_ = [[B1; B2; B3; B4; B5], blkdiag(L1, L2, L3, L4, L5)];

F_bar = blkdiag(F1_bar, F2_bar, F3_bar, F4_bar, F5_bar);


Phi = [
    Phi1;
    Phi2;
    Phi3;
    Phi4;
    Phi5
];

L_bar = blkdiag(L1, L2, L3, L4, L5);

Phi_pinv = pinv(Phi);




%% ========= 5) 3개 센서 조합별 Phi_bar_pinv 계산 =========

% Phi 리스트 및 각 센서 상태크기 li 사용
Phi_list = {Phi1, Phi2, Phi3, Phi4, Phi5};
li_vec   = [size(Phi1,1), size(Phi2,1), size(Phi3,1), size(Phi4,1), size(Phi5,1)];
% 보통 li_vec = [6 4 6 4 4] 가 나올 것

% 센서 인덱스 조합 (5개 중 3개씩 → 10개 조합)
idx_comb = nchoosek(1:5, 3);   % 10 x 3

Phi_bar_set  = cell(10,1);    % 각 k에 대한 [Phi_i1; Phi_i2; Phi_i3]
Phi_bar_pinv = cell(10,1);    % 각 k에 대한 pinv(Phi_bar_k)

for k = 1:size(idx_comb,1)
    idx = idx_comb(k,:);   % 예: [1 2 3]

    % Phi_bar_k = [Phi_i1; Phi_i2; Phi_i3]
    Phi_bar_k = [
        Phi_list{idx(1)};
        Phi_list{idx(2)};
        Phi_list{idx(3)}
    ];

    Phi_bar_set{k}  = Phi_bar_k;
    Phi_bar_pinv{k} = pinv(Phi_bar_k);

    [r,c]   = size(Phi_bar_k);
    [rp,cp] = size(Phi_bar_pinv{k});

    fprintf(['k = %d, sensors = [%d %d %d], ', ...
             'Phi_bar size = [%d x %d], Phi_bar_pinv size = [%d x %d]\n'], ...
            k, idx(1), idx(2), idx(3), r, c, rp, cp);
end


%% ========= 6) 각 조합에 대해 z -> x_hat(k) 매핑 H_k 만들기 =========
% z = [z1; z2; z3; z4; z5] (크기 24x1)

% 전역 z 인덱스 (cum_li로 경계 계산)
cum_li = [0, cumsum(li_vec)];   % 예: [0 6 10 16 20 24]

H_k = cell(10,1);   % H_k{k} : 6 x 24,  z -> x_hat^(k)

for k = 1:size(idx_comb,1)
    idx_sensors = idx_comb(k,:);      % 예: [1 2 3]
    Pk = Phi_bar_pinv{k};             % 6 x (li_i1 + li_i2 + li_i3)

    Hk = zeros(6, 24);

    % Pk의 column은 [z_i1; z_i2; z_i3] 순서
    col_start = 0;
    for jj = 1:3
        s = idx_sensors(jj);          % 센서 번호 (1~5)
        lj = li_vec(s);               % 해당 센서 상태 크기

        cols_local  = col_start + (1:lj);   % Pk 내에서의 column 구간
        cols_global = (cum_li(s)+1) : cum_li(s+1);  % z_bar에서의 column 구간

        Hk(:, cols_global) = Pk(:, cols_local);

        col_start = col_start + lj;
    end

    H_k{k} = Hk;
end


%% ========= 7) r_k = x_hat^(k) - x_hat^(k+1) 를 쌓아서 H (60x24) 만들기 =========
% r = [r1; r2; ...; r10] = H * z

H = zeros(6*10, 24);

for k = 1:10
    k_next = mod(k,10) + 1;       % 1→2, 2→3, ..., 10→1 (cyclic)

    H_block = H_k{k} - H_k{k_next};     % 6 x 24

    rows = (6*(k-1)+1) : (6*k);         % 해당 r_k 위치
    H(rows, :) = H_block;
end

% 이제 H 는 60x24 행렬이고,
% z_bar = [z1; z2; z3; z4; z5] 에 대해
% r = H * z_bar  하면
% r = [r1; r2; ...; r10], 각 r_k는 6x1 (x_hat^(k) - x_hat^(k+1)).



%% s 파라미터로 양자화
s = 10000;

Phi_pinv_bar = round(s * Phi_pinv)
F_bar; % 이미 정수
H_bar = round(s * H); % H를 양자화
G_bar = round(s * G_);% G_를 양자화

rank(H)
rank(H_bar)



%% mat. 포맷으로 저장/ F_bar는 정수 , G_, H 는 실수
% save('FGH_data.mat','F_bar','G_','H');



%% Simulation over Real #
iter = 20;

% plant initial state
xp0 = 1*ones(n,1);
z_hat_0 = zeros(24,1);
z_hat = z_hat_0;
xp = xp0;
u = [];
y = [];
r = [];


for i = 1:iter
    % plant & observer output

    if i == 5 
        attack = [0; 0; 0; 4; 0];
    else 
        attack = 0;
    end

    y = [y, C*xp(:,i)+ attack];
    r = [r, H*z_hat(:,i)];
    
    % feedback input
    u = K*xp;

    % state update
    xp = [xp, A*xp(:,i) + B*u(:,i)];
    z_hat = [z_hat, F_bar*z_hat(:,i) + G_*[u(:,i); y(:,i)]];
end


%% Plotting


t = Ts*(0:iter-1);   % z_hat(:,1) ~ z_hat(:,iter) 에 해당하는 시간축

figure(1)
plot(t, u)
hold on

title('Control input u')
legend

figure(2)
plot(Ts*(0:iter-1), y)
hold on
title('Plant output y')
legend


figure(3)
plot(Ts*(0:iter-1), r)
hold on
title('resiudal r')


%% ========= z_hat 블록별 노름 플롯 (5개) =========

% z_hat 크기: 24 x (iter+1)
% 각 센서 상태 차원
idx1 = (cum_li(1)+1) : cum_li(2);
idx2 = (cum_li(2)+1) : cum_li(3);
idx3 = (cum_li(3)+1) : cum_li(4);
idx4 = (cum_li(4)+1) : cum_li(5);
idx5 = (cum_li(5)+1) : cum_li(6);



z1_norm = vecnorm(z_hat(idx1,1:iter));   % 1 x iter
z2_norm = vecnorm(z_hat(idx2,1:iter));
z3_norm = vecnorm(z_hat(idx3,1:iter));
z4_norm = vecnorm(z_hat(idx4,1:iter));
z5_norm = vecnorm(z_hat(idx5,1:iter));

figure(4);
subplot(5,1,1);
plot(t, z1_norm);
title('||z_1||');
ylabel('norm');

subplot(5,1,2);
plot(t, z2_norm);
title('||z_2||');
ylabel('norm');

subplot(5,1,3);
plot(t, z3_norm);
title('||z_3||');
ylabel('norm');

subplot(5,1,4);
plot(t, z4_norm);
title('||z_4||');
ylabel('norm');

subplot(5,1,5);
plot(t, z5_norm);
title('||z_5||');
xlabel('time [s]');
ylabel('norm');



%% residual r를 6개씩 잘라서 10개 노름 플롯

% r 크기: 60 x iter  (각 6개가 하나의 residual r_k)
t_r = Ts*(0:iter-1);

rk_norm_all = zeros(10, iter);   % 각 k에 대한 노름 저장용

% 1) 먼저 모든 r_k 노름 계산해서 최대값 찾기
for k = 1:10
    rows = (6*(k-1)+1) : (6*k);   % r_k에 해당하는 6개 행
    rk    = r(rows,1:iter);       % 6 x iter
    rk_norm_all(k,:) = vecnorm(rk);   % 1 x iter
end

ymax = max(rk_norm_all(:));       % 모든 r_k 노름 중 최대값
ymin = 0;                         % 원하면 >0 로 바꿔도 됨

% 2) 이제 동일한 y축으로 플롯 + 조합 정보 표시
figure(5);
for k = 1:10
    k_next = mod(k,10) + 1;          % 1→2, 2→3, ..., 10→1

    comb_k     = idx_comb(k,:);      % 예: [1 2 3]
    comb_knext = idx_comb(k_next,:); % 예: [1 2 4]

    subplot(10,1,k);
    plot(t_r, rk_norm_all(k,:));
    ylim([ymin, ymax]);              % 모든 subplot에 같은 y축

    title(sprintf('r_%d: [%d %d %d] - [%d %d %d]', ...
        k, ...
        comb_k(1), comb_k(2), comb_k(3), ...
        comb_knext(1), comb_knext(2), comb_knext(3)));

    if k == 10
        xlabel('time [s]');
    end
    ylabel('norm');
end



%% 파이썬 포맷 프린트

% print_numpy('A', A, 6);
% print_numpy('B', B, 6);
% print_numpy('C', C, 0);
% print_numpy('Phi_pinv_bar', Phi_pinv_bar, 0);
% print_numpy('K', K, 6);
% 
% 
% 
% print_numpy('F_bar', F_bar, 0);
% print_numpy('G_bar', G_bar, 0);
% print_numpy('H_bar', H_bar, 0);
% 
% 

function print_numpy(name, M, prec)
    if nargin < 3, prec = 4; end
    fmt = ['%0.' num2str(prec) 'f'];

    if isvector(M)
        % 1D 벡터는 한 줄로 출력 (예: H)
        fprintf('%s = np.array([ ', name);
        for k = 1:numel(M)
            fprintf(fmt, M(k));
            if k < numel(M), fprintf(', '); end
        end
        fprintf(' ], dtype=np.float64)  # shape (%d,)\n\n', numel(M));
    else
        % 2D 행렬은 행 단위로 출력 (예: F, G)
        fprintf('%s = np.array([\n', name);
        for i = 1:size(M,1)
            fprintf('    [ ');
            for j = 1:size(M,2)
                fprintf(fmt, M(i,j));
                if j < size(M,2), fprintf(', '); end
            end
            if i < size(M,1)
                fprintf(' ],\n');
            else
                fprintf(' ]\n');
            end
        end
        fprintf('], dtype=np.float64)\n\n');
    end
end