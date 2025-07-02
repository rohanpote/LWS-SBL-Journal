function [Phi_2D,ugrid,vgrid,x_pos,y_pos]=create_2Ddictionary(m1,m2,n1,n2,type)
if strcmp(type,'random')
    Phi_2D=randn(m1*m2,n1*n2);
    ugrid=zeros(1,n1*n2); vgrid=zeros(1,n1*n2);
elseif strcmp(type,'ULA')
    u = linspace(-1,1,n1);
    v = linspace(-1,1,n2);
    [umat,vmat] = meshgrid(u,v);
    ugrid_t = umat(:)';
    vgrid_t = vmat(:)';
    ugrid = ugrid_t(ugrid_t.^2+vgrid_t.^2<=1);
    vgrid = vgrid_t(ugrid_t.^2+vgrid_t.^2<=1);
    [x_pos, y_pos] = meshgrid(0:m1-1, 0:m2-1);
    Phi_2D = exp(-1j*pi*(x_pos(:)*ugrid+y_pos(:)*vgrid));
end