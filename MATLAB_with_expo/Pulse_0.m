clear all; close all; clc;

omg = 2;
beta = 2; alpha = 1;


omega_0 = 1/2* omg;
omega_1 = 3/2* omg;
omega_2 = 5/2* omg;

omega_j = 2.2;

del0 = omega_0 - omega_j;
del1 = omega_1 - omega_j;
del2 = omega_2 - omega_j;


lamda1 = 1; lamda2 = sqrt(2);

F_0 = basis(3,1);

psi_0 = sparse( [1; 0; 0;] );

F1_F0x = lamda1*sparse( [0,1,0; 0,0,0; 0,0,0;] );F0_F1x = lamda1*sparse( [0,0,0; 1,0,0; 0,0,0;] );
F2_F1x = lamda1*sparse( [0,0,0; 0,0,1; 0,0,0;] );F1_F2x = lamda1*sparse( [0,0,0; 0,0,0; 0,1,0;] );


F1_F0y = lamda2*sparse( [0,-1i,0; 0,0,0; 0,0,0;] );F0_F1y = lamda2*sparse( [0,0,0; 1i,0,0; 0,0,0;] );
F2_F1y = lamda2*sparse( [0,0,0; 0,0,-1i; 0,0,0;] );F1_F2y = lamda2*sparse( [0,0,0; 0,0,0; 0,1i,0;] );


F0_F0 = sparse( [1,0,0; 0,0,0; 0,0,0;] );
F1_F1 = sparse( [0,0,0; 0,1,0; 0,0,0;] );
F2_F2 = sparse( [0,0,0; 0,0,0; 0,0,1;] );


dels = sparse( [del0,0,0; 0,del1,0; 0,0,del2;] );





F2_F1 = sparse( [0,0,0; 0,0,1; 0,0,0;] );Sgm = 1;t0 = 6;
T_pts = 100; T_fin = 15; T_ini = 0; delt = (T_fin-T_ini)/T_pts; 

tlist = [T_ini: delt: T_fin]; % Define time vectorF2_F1 = sparse( [0,0,0; 0,0,1; 0,0,0;] );

sgm = 1;
rats = 4;
tp = (T_ini+T_fin)/2;

psi_C = zeros(3, length(tlist));
psi_C(:, 1) = psi_0;

Ham0 = dels;



sgm_A = 0.5:0.01:2;
slen = length(sgm_A);

omTrain = zeros(slen, 2*(T_pts+1));
FocTrain = zeros(slen, 3*(T_pts+1));


for sg_i = 1:slen
omT = zeros(1, 2*(T_pts+1));
omT(1,1) = 0;
omT(1,2) = 0;

Foc = zeros(1, 3*(T_pts+1));
Foc(1,1) = 1;
Foc(1,2) = 0;
Foc(1,3) = 0;

for t_i = 2:T_pts+1
    t_i
    tm = tlist(1,t_i); tm1 = tlist(1,t_i-1);

    omI = Omega_I(tm, Sgm, tp, rats);
    omQ = (Omega_I(tm, Sgm, tp, rats) - Omega_I(tm1, Sgm, tp, rats))/(tm-tm1);

    omT(1, 2*t_i-1) = omI;
    omT(1, 2*t_i) = omQ;

    Hamt = Ham0 + lamda1/2* omI * (F1_F0x+F0_F1x+F2_F1x+F1_F2x)+ lamda1/2* omI * (F1_F0y+F0_F1y+F2_F1y+F1_F2y);
   
    Ham = -1i*Hamt;

    psi_beg(:,:) = psi_C(:,t_i-1);
    psi_beg = sparse(psi_beg);

%    rho_end = expm( Liu*delt   )*rho_beg;
    tolf = 1e-7;
    [psi_end,err] = expv(delt,Ham,psi_beg,tolf,30);
    
    psi_C(:,t_i) = psi_end;
    Foc(1, 3*t_i-2) = psi_end(1,1);
    Foc(1, 3*t_i-1) = psi_end(2,1);
    Foc(1, 3*t_i) = psi_end(3,1);
%    Fop_A(1,t_i) = trace(Fop*roI);    
%    ph_k_t(1,t_i) = tdc;
end

omTrain(sg_i,:) = omT(:,:);
FocTrain(sg_i,:) = Foc(:,:);

end


figure(4);plot(tlist,abs(psi_C(1,:)),'b','LineWidth',2);title('\rho_{last,last}')
figure(5);plot(tlist,abs(psi_C(2,:)),'b','LineWidth',2);title('\rho_{last,last}')
figure(6);plot(tlist,abs(psi_C(3,:)),'b','LineWidth',2);title('\rho_{last,last}')

%figure(10);plot(tlist,Fop_A,'r','LineWidth',2);title('<F> for F2=20')
%figure(11);plot(tlist,var_A,'b','LineWidth',2);
%save('./Sc_N60_20.mat','rho_C','tlist','Sgm');
%save('./Sc_N80_F44.mat','N_A','N2_A','var_A','Fop_A','tlist','Sgm');
tlist3 = [T_ini: delt/3: T_fin+2*delt/3];
tlist2 = [T_ini: delt/2: T_fin+1*delt/2];

figure(9);
plot(tlist2,omT);
figure(10);
plot(tlist3,Foc);

save("ThrLev.mat","omTrain","FocTrain");

