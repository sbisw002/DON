function [phit] = Omega_I(tt, Sgm, tp, kappa)
phit = exp(-( (tt-tp/2) /sqrt(2)/Sgm  )^2) - exp(-( (tp/2) /sqrt(2)/Sgm  )^2) ;