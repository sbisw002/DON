function [bb] = basis(Nsi, ind)

bb = zeros(Nsi,1);
bb(ind,1) = 1;