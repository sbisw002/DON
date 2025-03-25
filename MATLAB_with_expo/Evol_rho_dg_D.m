clear all; close all; clc;
Nm = 60;

global Delta U gamma F
omegL = 400;
omeg = 1000;
gamma = 1;
%Delta = omegL - omeg;
Delta = 2*gamma;

U = 0.1*gamma;
F = 2;


rad = zeros(1,Nm);cad = zeros(1,Nm);vad = zeros(1,Nm);
ra = zeros(1,Nm);ca = zeros(1,Nm);va = zeros(1,Nm);
rNp = zeros(1,Nm);cNp = zeros(1,Nm);vNp = zeros(1,Nm);


for it = 1:Nm
rad(1,it) = it; cad(1,it) = it+1; vad(1,it) = sqrt(it);
ra(1,it) = it+1;  ca(1,it) = it;  va(1,it) = sqrt(it);
rNp(1,it) = it+1;  cNp(1,it) = it+1;  vNp(1,it) = it;
end
ad = sparse(rad,cad,vad,Nm+1,Nm+1);   
a = sparse(ra,ca,va,Nm+1,Nm+1);   
Np = sparse(rNp,cNp,vNp,Nm+1,Nm+1);   


%ad1 = diag(sqrt(1:Nm),-1);
%a1 = diag(sqrt(1:Nm),1);
%Np1 = ad1*a1;
%Np1 = sparse(Np); ad1 = sparse(ad); a1 = sparse(a);

Tmax = 0.4;
Tsteps = 1e8;
delt = Tmax/Tsteps;
tsA = 0:delt:Tmax;
t_M = 3*Tmax/4; Tsig = Tmax/16;



%der_C = zeros((Nm+1)*(Nm+1),Tsteps);
%rho_C = zeros((Nm+1)*(Nm+1),Tsteps+1);
%N_A = zeros(1,Tsteps+1);
%tr_A = zeros(1,Tsteps+1);
%F_dz = zeros(1,Tsteps+1);



ini_rho = zeros((Nm+1)*(Nm+1),1); ini_rho(1,1) = 1;
rho_C(:,1) = ini_rho; N_A(1,1) = trace(Np*reshape(ini_rho,Nm+1,Nm+1)); tr_A(1,1) = 1;
%F_A = [0:Tsteps]*0.1+20;
%F_A = 3000*ones(size([0:Tsteps]));

DeltaF = gamma; F0 = 0*gamma;
ts = DeltaF * 600/gamma^2/4 / 100/2/4;
t_mid = Tmax/2;
t_1 = t_mid-ts; t_2 = t_mid+ts;

for it = 1:Tsteps+1
    if (tsA(1,it) < t_1)
    u1 = 0; u2 = 0;
    elseif (tsA(1,it)> t_1 && tsA(1,it) < t_mid )
    u1 = 1; u2 = 0;
    elseif (tsA(1,it)> t_mid && tsA(1,it) < t_2 )
    u1 = 1; u2 = 1;
    elseif (tsA(1,it)> t_2)
    u1 = 0; u2 = 0;
    end
F_A(1,it) = F0 + u1*(tsA(1,it)-t_1)*(15/20*gamma/ts) + u2*(tsA(1,it)-t_mid)*(-30/20*gamma/ts);
        
        
end

%figure(20);plot(tsA,F_A,'b','LineWidth',2);


%F_A = 3*gamma+2*exp(-(tsA-t_M).^2/2/Tsig^2);
%F_A = 0*ones(size(ts));
%figure(8); plot(ts,F_A,'r',ts,F_dz,'g');
%figure(9); plot(ts,F_dz,'g');


roH = zeros(Nm+1,1);
coH = zeros(Nm+1,1);
vaH = zeros(Nm+1,1);
rowH = zeros(Nm+1+2*Nm,1);
colH = zeros(Nm+1+2*Nm,1);
valH = zeros(Nm+1+2*Nm,1);

for it=1:Nm+1
    roH(it,1) = it; coH(it,1) = it;
    vaH(it,1) = -Delta*(it-1) + (U/2)*(it-1)*(it-2);    
    
    rowH(it,1) = it; colH(it,1) = it;
    valH(it,1) = -Delta*(it-1) + (U/2)*(it-1)*(it-2);        
end
tA(1,1) = 0;
for t_i = 1:Tsteps/1 
    t_i
    rowH(1:Nm+1,1) = roH;    colH(1:Nm+1,1) = coH;
    
    for it=1:Nm
    rowH(Nm+1+  2*(it-1)+1,1) = it; colH(Nm+1+  2*(it-1)+1,1) = it+1; valH(Nm+1+  2*(it-1)+1,1) = sqrt(it)*F_A(1,t_i);    
    rowH(Nm+1+  2*(it-1)+2,1) = it+1; colH(Nm+1+  2*(it-1)+2,1) = it; valH(Nm+1+  2*(it-1)+2,1) = sqrt(it)*F_A(1,t_i);     
    end
    
    Hams = sparse(rowH,colH,valH,Nm+1,Nm+1);
    
%    Ham = diag( -Delta* [0:Nm] + (U/2)*[0:Nm].*([0:Nm]-1 )   )+F_A(1,t_i)*diag( sqrt( [1:Nm] ),1 )+F_A(1,t_i)*diag( sqrt( [1:Nm] ),-1);
%    df = find(Ham - full(Hams))
    
    
    Liu = -1i*( kron(speye(Nm+1),Hams)- kron(Hams',speye(Nm+1) )  ) + 0.5*gamma* ( 2*kron(ad',a) - kron(speye(Nm+1),Np) - kron(Np,speye(Nm+1))  );
%    Liu = ( kron(speye(Nm+1),Hams)- kron(Hams',speye(Nm+1) )  )  + 1i* gamma * ( kron(ad',a) - 0.5*kron(speye(Nm+1),N) - 0.5*kron(N',speye(Nm+1))   );

%    [VV,DD] = eig(Liu);
    rho_beg(:,:) = rho_C(:,t_i);
%    rho_end = (VV*exp(-1i*DD*delt)*VV' )*rho_beg;
%    rho_end = (VV*  diag(exp( -1i*diag(DD)*delt   ))  *VV' )*rho_beg;
%    rho_end = expm( Liu*delt   )*rho_beg;

%    Liup = sparse(Liu);
    [rho_end,err] = expv(delt,Liu,rho_beg);


    
    rho_C(:,t_i+1) = rho_end;
    roI = reshape(rho_end,Nm+1,Nm+1);
    tr_A(1,t_i+1) = trace(roI);
    N_A(1,t_i+1) = trace(Np*roI);    
%    rho_CN(:,:) = rho_C(:,t_i);
%    der_C(:,t_i) = -1i*( kron(eye(Nm+1),Ham)- kron(Ham',eye(Nm+1) )  )*rho_CN;
%    rho_C(:,t_i+1) = der_C(:,t_i) * delt + rho_C(:,t_i);    

tA(1,t_i+1) = t_i;
end
tsA = tA;


figure(5);plot(tsA,abs(rho_C(1,:)),'b','LineWidth',2);

figure(6);plot(tsA,abs(rho_C(7,:)),'b','LineWidth',2);
figure(7);plot(tsA,abs(rho_C(12,:)),'b','LineWidth',2);
figure(8);plot(tsA,abs(rho_C(19,:)),'b','LineWidth',2);
%figure(9);plot(ts,abs(rho_C(41,:)),'b','LineWidth',2);

%figure(20);plot(tsA,F_A,'b','LineWidth',2);

figure(40);plot(tsA,N_A,'r','LineWidth',2);
figure(45);plot(tsA,tr_A,'r','LineWidth',2);

%figure(50);plot(F_A,N_A,'r','LineWidth',2);

%figure(35);plot(tsA,abs(rho_C( (Nm+1)*Nm + (Nm+1) ,:)),'b','LineWidth',2);
