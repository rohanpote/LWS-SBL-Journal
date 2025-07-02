for i=1:80
    m=20;
    n=200;
    Nres=n;%2000;
    u1=0;
    a1=exp(-1j*pi*(0:m-1)'*u1);
    a1=a1/vecnorm(a1);
    u2=i*2/n
    a2=exp(-1j*pi*(0:m-1)'*u2);
    a2=a2/vecnorm(a2);
    u3=-1:2/Nres:1-2/Nres;
    a3=exp(-1j*pi*(0:m-1)'*u3);
    a3=a3/diag(vecnorm(a3));
    %% OMP Failure Condition
    LHS=abs(a3'*a2-(a3'*a1)*(a1'*a2));
    RHS1=1-abs(a2'*a1)^2;
    figure(20); clf;
    plot(u3, LHS-RHS1,'Color',[0 0.4470 0.7410],'LineWidth',2)
    grid on
    hold on
    plot(u1,0,'xr','MarkerSize',16,'LineWidth',1)
    plot(u2*ones(1,10),linspace(0,0.2,10),'-.r','LineWidth',1)
    xlabel('Spatial angle in u-space')
    ylabel('OMP failure condition check')
    set(gca,'FontWeight','bold','FontSize',12)
    legend('OMP Failure Condition', 'Recovered DoA','DoA to be Recovered','Location','NorthEast')
    axis([-0.4 0.4 -0.04 0.2])

    %% Order-recursive MP Failure Condition
%     LHS_ORMP=(abs(a3'*a2-(a3'*a1)*(a1'*a2)).^2)./(1-abs(a3'*a1).^2);
%     RHS1_ORMP=(1-abs(a2'*a1)^2);
%     figure(21); clf
%     plot(u3, LHS_ORMP-RHS1_ORMP,'Color',[0 0.4470 0.7410],'LineWidth',2)
%     grid on
%     hold on
%     plot(u1,0,'xr','MarkerSize',16,'LineWidth',1)
%     plot(u2*ones(1,10),linspace(0,0.2,10),'-.r','LineWidth',1)
% %     title(['ORMP Cond.: Grid point index for u2=' num2str(i)])
%     xlabel('Spatial angle in u-space')
%     ylabel('LWS-SBL failure condition check')
%     set(gca,'FontWeight','bold','FontSize',12)
%     legend('LWS-SBL Failure Condition', 'Recovered DoA','DoA to be Recovered','Location','NorthEast')
%     axis([-0.4 0.4 -0.04 0.2])
end


