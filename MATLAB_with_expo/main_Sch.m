clear all; close all; clc;

Nmax = 8;
gamma = 1;
Delta = 0*gamma;kappa = 1*gamma;lamda = 1*gamma;

U = 0.1*gamma;F0 = 0;F1 = 0;F2 = 44;


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
T_pts = 150; T_fin = 15; T_ini = 0; delt = (T_fin-T_ini)/T_pts; 
tlist = [T_ini: delt: T_fin]; % Define time vector


Np = kron(number(Nmax),speye(3));
Np_2 = kron(number(Nmax)*number(Nmax),speye(3));

rho_C = zeros(9*(Nmax+1)*(Nmax+1),T_pts+1);
N_A = zeros(1,T_pts+1);N2_A = zeros(1,T_pts+1);var_A = zeros(1,T_pts+1);
Fop_A = zeros(1,T_pts+1);tr_A = zeros(1,T_pts+1);rNN_A = zeros(1,T_pts+1);
N0_A = zeros(1,T_pts+1);NM_A = zeros(1,T_pts+1);ph_k_t = zeros(1,T_pts+1);


ini_rho = reshape(psi_0* psi_0.',3*(Nmax+1)*3*(Nmax+1),1);
%zeros(9*(Nmax+1)*(Nmax+1),1); ini_rho(1,1) = 1;
rho_C(:,1) = ini_rho;
N_A(1,1) = trace(Np*reshape(ini_rho,3*(Nmax+1),3*(Nmax+1)));
N2_A(1,1) = trace(Np_2*reshape(ini_rho,3*(Nmax+1),3*(Nmax+1)));
var_A(1,1) = N2_A(1,1) - N_A(1,1)*N_A(1,1);
tr_A(1,1) = 1;
An = kron(destroy(Nmax),speye(3)); An_d = kron(create(Nmax),speye(3));
ph_k_t(1,1)=0;


for t_i = 2:T_pts+1
    t_i
    tmm = tlist(1,t_i);
    tdc = H2_coeff(tmm, Sgm, t0, kappa);
    
    Hams = H0 + H1 + tdc*H2;
%    Hams = H0 + H1;
    
    Liu = -1i*( kron(speye(3*(Nmax+1)),Hams)- kron(Hams.',speye(3*(Nmax+1)) )  ) ... 
         + 0.5*gamma* ( 2*kron(An_d.',An) - kron(Np.',speye(3*(Nmax+1)) ) - kron(speye(3*(Nmax+1)),Np) )...
         + 0.5*kappa* ( 2*kron(F0tF1.',F1tF0) - kron(F1tF1.',speye(3*(Nmax+1)) ) - kron(speye(3*(Nmax+1)),F1tF1) )...
         + 0.5*lamda* ( 2*kron(F2tF1.',F1tF2) - kron(F1tF1.',speye(3*(Nmax+1)) ) - kron(speye(3*(Nmax+1)),F1tF1) );

    rho_beg(:,:) = rho_C(:,t_i-1);
    rho_beg = sparse(rho_beg);

%    rho_end = expm( Liu*delt   )*rho_beg;
    tolf = 1e-7;
    [rho_end,err] = expv(delt,Liu,rho_beg,tolf,30);


    
    rho_C(:,t_i) = rho_end;
    roI = reshape(rho_end,3*(Nmax+1),3*(Nmax+1));
    rNN_A(1,t_i) = roI(3*(Nmax+1),3*(Nmax+1));
    tr_A(1,t_i) = trace(roI);
    N_A(1,t_i) = trace(Np*roI);    
    N2_A(1,t_i) = trace(Np_2*roI);     
    var_A(1,t_i) = N2_A(1,t_i) -  N_A(1,t_i)^2;
    Fop_A(1,t_i) = trace(Fop*roI);    
    ph_k_t(1,t_i) = tdc;
end

figure(4);plot(tlist,abs(rNN_A),'b','LineWidth',2);title('\rho_{last,last}')
%figure(5);plot(tlist,abs(rho_C(1,:)),'b','LineWidth',2);

%figure(6);plot(tlist,abs(rho_C(7,:)),'b','LineWidth',2);
%figure(7);plot(tlist,abs(rho_C(12,:)),'b','LineWidth',2);

figure(8);plot(tlist,tr_A,'b','LineWidth',2);
figure(9);plot(tlist,N_A,'b','LineWidth',2);

%for t_i = 1:T_pts+1
%    t_i
%    tmm = tlist(1,t_i);
%    rho_L = rho_C(:,t_i);
%    roI = reshape(rho_L,3*(Nmax+1),3*(Nmax+1));
%    N_A(1,t_i)=trace(Np*roI);   
%end

figure(10);plot(tlist,Fop_A,'r','LineWidth',2);title('<F> for F2=20')
figure(11);plot(tlist,var_A,'b','LineWidth',2);
%save('./Sc_N60_20.mat','rho_C','tlist','Sgm');
save('./Sc_N80_F44.mat','N_A','N2_A','var_A','Fop_A','tlist','Sgm');


