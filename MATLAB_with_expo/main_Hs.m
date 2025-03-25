clear all; close all; clc;

Nmax = 80;
gamma = 1;
Delta = 0*gamma;kappa = 1*gamma;lamda = 1*gamma;

U = 0.1*gamma;F0 = 0;F1 = 0;F2 = 20;


N_0 = basis(Nmax+1,1);
F_0 = basis(3,1);

psi_0 = kron(N_0,F_0);

b = kron(destroy(Nmax), speye(3) );

F1_F0 = sparse( [0,1,0; 0,0,0; 0,0,0;] );F0_F1 = sparse( [0,0,0; 1,0,0; 0,0,0;] );
F1_F2 = sparse( [0,0,0; 0,0,0; 0,1,0;] );F2_F1 = sparse( [0,0,0; 0,0,1; 0,0,0;] );

F1tF0 = kron( speye(Nmax+1), F1_F0 );F0tF1 = kron( speye(Nmax+1), F0_F1 );
F1tF2 = kron( speye(Nmax+1), F1_F2 );F2tF1 = kron( speye(Nmax+1), F2_F1 );

%bdb = kron(number(Nmax),speye(3) );

%bNp = kron(basis(Nmax+1,Nmax)*(basis(Nmax+1,Nmax))', speye(3)  );
%b0p = kron(basis(Nmax+1,1)*(basis(Nmax+1,1))', speye(3) );

F1_F1 = sparse( [0,0,0; 0,1,0; 0,0,0;] );F1tF1 = kron( speye(Nmax+1), F1_F1 );
FA = sparse( [F0,0,0; 0,F1,0; 0,0,F2;] );Fop = kron(speye(Nmax+1), FA );

H0 = kron( -Delta* number(Nmax) + U*( number(Nmax).^2 - number(Nmax)  )   , speye(3) );  
H1 = kron( ( destroy(Nmax) + create(Nmax) ), FA );  
H2 = kron( speye(Nmax+1), (F1_F0+F0_F1) );

Sgm = 1;t0 = 6;
T_pts = 4e2; T_fin = 12; T_ini = 0; delt = (T_fin-T_ini)/T_pts; 
tlist = [T_ini: delt: T_fin]; % Define time vector


Np = kron(number(Nmax),speye(3));
num_C = zeros(9*(Nmax+1)*(Nmax+1),T_pts+1);
cN_A = zeros(1,T_pts+1);N_A = zeros(1,T_pts+1);tr_A = zeros(1,T_pts+1);rNN_A = zeros(1,T_pts+1);
sNN_A = zeros(1,T_pts+1);  tNN_A = zeros(1,T_pts+1);
N0_A = zeros(1,T_pts+1);NM_A = zeros(1,T_pts+1);ph_k_t = zeros(1,T_pts+1);

ini_num = reshape(Np,9*(Nmax+1)*(Nmax+1),1);

num_C(:,1) = ini_num;
sq_inN = reshape(ini_num,3*(Nmax+1),3*(Nmax+1) );
N_A(1,1) = trace(Np*sq_inN);
tr_A(1,1) = trace(Np) ;
An = kron(destroy(Nmax),speye(3)); An_d = kron(create(Nmax),speye(3));
ph_k_t(1,1)=0;

ini_rho = kron(basis(Nmax+1,1),basis(3,1));
rho0 = ini_rho*ini_rho';

cN_A(1,1) = trace(Np*sq_inN);    
 
rNN_A(1,1) = sq_inN(3*(Nmax+1),3*(Nmax+1));         %(    = ini_num(3*(Nmax+1)*3*(Nmax+1),1)   )
sNN_A(1,1) = sq_inN(3*(Nmax+1-20),3*(Nmax+1-20));
tNN_A(1,1) = sq_inN(3*(Nmax+1-40),3*(Nmax+1-40));


for t_i = 2:T_pts+1
    t_i
    tmm = tlist(1,t_i);
    tdc = H2_coeff(tmm, Sgm, t0, kappa);
    
    Hams = H0 + H1 + tdc*H2;
%    Hams = H0 + H1;
    
    Liu = -1i*( kron(Hams.',speye(3*(Nmax+1)) ) - kron(speye(3*(Nmax+1)),Hams)  ) ... 
 + 0.5*gamma* ( 2*kron(An.',An_d) - kron(speye(3*(Nmax+1)),Np) - kron(Np.',speye(3*(Nmax+1)) )   )...
 + 0.5*kappa* ( 2*kron(F1tF0.',F0tF1) - kron(F1tF1.',speye(3*(Nmax+1)) ) - kron(speye(3*(Nmax+1)),F1tF1) )...
 + 0.5*lamda* ( 2*kron(F1tF2.',F2tF1) - kron(F1tF1.',speye(3*(Nmax+1)) ) - kron(speye(3*(Nmax+1)),F1tF1) );

    num_beg(:,:) = num_C(:,t_i-1);
    num_beg = sparse(num_beg);

%    rho_end = expm( Liu*delt   )*rho_beg;
    tolf = 1e-8;
    [num_end,err] = expv(delt,Liu,num_beg,tolf,30);


    
    num_C(:,t_i) = num_end;
    roI = reshape(num_end,3*(Nmax+1),3*(Nmax+1));
    rNN_A(1,t_i) = roI(3*(Nmax+1),3*(Nmax+1));
    sNN_A(1,t_i) = roI(3*(Nmax+1-20),3*(Nmax+1-20));
    tNN_A(1,t_i) = roI(3*(Nmax+1-40),3*(Nmax+1-40));
    
    tr_A(1,t_i) = trace(roI);
    cN_A(1,t_i) = trace(Np*roI);    
    ph_k_t(1,t_i) = tdc;
  
end

figure(4);plot(tlist,abs(rNN_A),'b','LineWidth',2);
figure(5);plot(tlist,abs(sNN_A),'b','LineWidth',2);
figure(6);plot(tlist,abs(tNN_A),'b','LineWidth',2);

figure(7);plot(tlist,abs(rNN_A),'b',tlist,abs(sNN_A),'g',tlist,abs(tNN_A),'r','LineWidth',2);

%figure(5);plot(tlist,abs(rho_C(1,:)),'b','LineWidth',2);

%figure(6);plot(tlist,abs(rho_C(7,:)),'b','LineWidth',2);
%figure(7);plot(tlist,abs(rho_C(12,:)),'b','LineWidth',2);

figure(8);plot(tlist,tr_A,'b','LineWidth',2);title('The trace is not 1 anymre!!')
figure(9);plot(tlist,cN_A,'b','LineWidth',2);






for t_i = 1:T_pts+1
    t_i
    tmm = tlist(1,t_i);
    
    num_end = num_C(:,t_i);
    numI = reshape(num_end,3*(Nmax+1),3*(Nmax+1));
    N_A(1,t_i)=trace(numI*rho0);
    
end

figure(10);plot(tlist,N_A,'b','LineWidth',2);

save('./N80_Np_0_20_I.mat','num_C','tlist','Sgm');


