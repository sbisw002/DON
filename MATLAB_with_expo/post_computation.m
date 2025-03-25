clear all; close all; clc;

load('./N120_50.mat');
aN2_A = zeros(1,T_pts+1);bN2_A = zeros(1,T_pts+1);cN2_A = zeros(1,T_pts+1);
dN2_A = zeros(1,T_pts+1);eN2_A = zeros(1,T_pts+1);fN2_A = zeros(1,T_pts+1);
gN2_A = zeros(1,T_pts+1);hN2_A = zeros(1,T_pts+1);iN2_A = zeros(1,T_pts+1);

aN1_A = zeros(1,T_pts+1);bN1_A = zeros(1,T_pts+1);cN1_A = zeros(1,T_pts+1);
dN1_A = zeros(1,T_pts+1);eN1_A = zeros(1,T_pts+1);fN1_A = zeros(1,T_pts+1);
gN1_A = zeros(1,T_pts+1);hN1_A = zeros(1,T_pts+1);iN1_A = zeros(1,T_pts+1);

aN0_A = zeros(1,T_pts+1);bN0_A = zeros(1,T_pts+1);cN0_A = zeros(1,T_pts+1);
dN0_A = zeros(1,T_pts+1);eN0_A = zeros(1,T_pts+1);fN0_A = zeros(1,T_pts+1);
gN0_A = zeros(1,T_pts+1);hN0_A = zeros(1,T_pts+1);iN0_A = zeros(1,T_pts+1);




cN_A = zeros(1,T_pts+1);N_A = zeros(1,T_pts+1);tr_A = zeros(1,T_pts+1);
N0_A = zeros(1,T_pts+1);NM_A = zeros(1,T_pts+1);ph_k_t = zeros(1,T_pts+1);


aN2_A(1,1) = sq_inN(3*(01),3*(01));     
bN2_A(1,1) = sq_inN(3*(21),3*(21));
cN2_A(1,1) = sq_inN(3*(41),3*(41));

dN2_A(1,1) = sq_inN(3*(61),3*(61));     
eN2_A(1,1) = sq_inN(3*(81),3*(81));
fN2_A(1,1) = sq_inN(3*(101),3*(101));

gN2_A(1,1) = sq_inN(3*(121),3*(121));

N_A(1,1) = trace(Np*sq_inN);
tr_A(1,1) = trace(Np) ;
ph_k_t(1,1)=0;


for t_i = 1:T_pts+1
    t_i
    tmm = tlist(1,t_i);
    
    num_end = num_C(:,t_i);
    roI = reshape(num_end,3*(Nmax+1),3*(Nmax+1));
    
    figure(1);
%    spy(roI);
    toI = roI - diag(diag(roI),0);
    imagesc(abs(toI))

    N_A(1,t_i)=trace(roI*rho0);
    
    aN2_A(1,t_i) = roI(3*(01),3*(01));bN2_A(1,t_i) = roI(3*(21),3*(21));cN2_A(1,t_i) = roI(3*(41),3*(41));
    dN2_A(1,t_i) = roI(3*(61),3*(61));eN2_A(1,t_i) = roI(3*(81),3*(81));fN2_A(1,t_i) = roI(3*(101),3*(101));     
    gN2_A(1,t_i) = roI(3*(121),3*(121));
    
    aN1_A(1,t_i) = roI(3*(01)-1,3*(01)-1);bN1_A(1,t_i) = roI(3*(21)-1,3*(21)-1);cN1_A(1,t_i) = roI(3*(41)-1,3*(41)-1);
    dN1_A(1,t_i) = roI(3*(61)-1,3*(61)-1);eN1_A(1,t_i) = roI(3*(81)-1,3*(81)-1);fN1_A(1,t_i) = roI(3*(101)-1,3*(101)-1);     
    gN1_A(1,t_i) = roI(3*(121)-1,3*(121)-1);
    
    aN0_A(1,t_i) = roI(3*(01)-2,3*(01)-2);bN0_A(1,t_i) = roI(3*(21)-2,3*(21)-2);cN0_A(1,t_i) = roI(3*(41)-2,3*(41)-2);
    dN0_A(1,t_i) = roI(3*(61)-2,3*(61)-2);eN0_A(1,t_i) = roI(3*(81)-2,3*(81)-2);fN0_A(1,t_i) = roI(3*(101)-2,3*(101)-2);     
    gN0_A(1,t_i) = roI(3*(121)-2,3*(121)-2);
    
        
    
    
    
    tr_A(1,t_i) = trace(roI);
    cN_A(1,t_i) = trace(Np*roI);    
    ph_k_t(1,t_i) = tdc;
      
    
    
end


figure(10);plot(tlist,N_A,'b','LineWidth',2);

%figure(12);plot(tlist,abs(aN2_A),'b',tlist,abs(bN2_A),'g',tlist,abs(cN2_A),'r','LineWidth',2);
%figure(13);plot(tlist,abs(dN2_A),'k',tlist,abs(eN2_A),'y',tlist,abs(fN2_A),'c',tlist,abs(gN2_A),'m','LineWidth',2);
figure(14);plot(tlist,abs(aN2_A),'b',tlist,abs(bN2_A),'g',tlist,abs(cN2_A),'r',tlist,abs(dN2_A),'k',tlist,abs(eN2_A),'y',tlist,abs(fN2_A),'c',tlist,abs(gN2_A),'m','LineWidth',2);


%figure(22);plot(tlist,abs(aN1_A),'b',tlist,abs(bN1_A),'g',tlist,abs(cN1_A),'r','LineWidth',2);
%figure(23);plot(tlist,abs(dN1_A),'k',tlist,abs(eN1_A),'y',tlist,abs(fN1_A),'c',tlist,abs(gN1_A),'m','LineWidth',2);
figure(24);plot(tlist,abs(aN1_A),'b',tlist,abs(bN1_A),'g',tlist,abs(cN1_A),...
tlist,abs(dN1_A),'k',tlist,abs(eN1_A),'y',tlist,abs(fN1_A),'c',tlist,abs(gN1_A),'m','LineWidth',2);



%figure(32);plot(tlist,abs(aN0_A),'b',tlist,abs(bN0_A),'g',tlist,abs(cN0_A),'r','LineWidth',2);
%figure(33);plot(tlist,abs(dN0_A),'k',tlist,abs(eN0_A),'y',tlist,abs(fN0_A),'c',tlist,abs(gN0_A),'m','LineWidth',2);
figure(34);plot(tlist,abs(aN0_A),'b',tlist,abs(bN0_A),'g',tlist,abs(cN0_A),'r',...
tlist,abs(dN0_A),'k',tlist,abs(eN0_A),'y',tlist,abs(fN0_A),'c',tlist,abs(gN0_A),'m','LineWidth',2);


