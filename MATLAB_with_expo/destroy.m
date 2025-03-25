function [Dess] = destroy(Nsi)

Ae = sqrt(1:Nsi);
Des = diag(Ae,1);

Dess = sparse(Des);
