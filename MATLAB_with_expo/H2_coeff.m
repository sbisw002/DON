function [phit] = H2_coeff(tt, Sgm, t0, kappa)
phit = sqrt(kappa)*exp(-( (tt-t0) /2/Sgm  )^2)/sqrt(sqrt(2*pi*Sgm*Sgm));