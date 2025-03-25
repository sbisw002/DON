clear all; close all; clc;
Nmax = 120;
gamma = 1;
Delta = 0*gamma;kappa = 1*gamma;lamda = 1*gamma;

U = 0.1*gamma;F0 = 0;F1 = 0;F2 = 50;
Sgm = 1;t0 = 6;
T_pts = 4e2; T_fin = 12; T_ini = 0; delt = (T_fin-T_ini)/T_pts; 
tlist = [T_ini: delt: T_fin]; % Define time vector

ini_rho = kron(basis(Nmax+1,1),basis(3,1));
rho0 = ini_rho*ini_rho';

a1= load('./N120_50_a.mat');
a1d = load('./N120_50_ad.mat');

a_C = a1.num_C;
ad_C = a1d.ad_C;

N_A = zeros(1,T_pts+1);com_A = zeros(1,T_pts+1);


for t_i = 1:T_pts+1
    t_i
    tmm = tlist(1,t_i);
    
    a_t = a_C(:,t_i);
    aS = reshape(a_t,3*(Nmax+1),3*(Nmax+1));
    ad_t = ad_C(:,t_i);
    adS = reshape(ad_t,3*(Nmax+1),3*(Nmax+1));    

    ad_a = adS * aS;
    a_ad = aS * adS;
    N_A(1,t_i)=trace(ad_a*rho0);
%    com_A(1,t_i) = (a_ad - ad_a)==eye(Nmax+1);
    com = a_ad - ad_a;
    clc;
end

figure(10);plot(tlist,N_A,'b','LineWidth',2);
figure(11);plot(tlist,com_A,'b','LineWidth',2);




figure(32);plot(tlist,N_A,'b','LineWidth',2);
