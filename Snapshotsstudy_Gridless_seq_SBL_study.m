% Array Processing Problem: Seq. SBL vs Matching Pursuit type algorithms
% Impemented: Seq. SBL 1.LS-OMP 2.OMP 3.MP 4.Weak-MP 5.Thresholding

clear all
rng(1)
addpath('G:\My Drive\Research\2_Code\Sparse-Bayesian-Learning\Gridless_SBL\plots_asilomar_draft\Simulation_study_ICASSP\Extension_work\NOMP\wcslspectralestimation-continuous-frequency-estimation-779a5e217053\')

m = 50;%[15 19 20:20:200];
n = 200;
L=[1:5 10 20:20:100 250 500 1000]; % single snapshot
suppsizemax = 10;
datagen = 0;
% epsilon = 1e-2;
% r = Inf; %suppsize;
ITER = 100;
SNR=30; % in dB
s_var=1;
w_var=s_var/10^(SNR/10);
sep=10; % make sure (max. sep X 2 X suppsizemax)<=n to ensure sep is always possible

% Grid Initialize
[A, u] = create_dictionary(m(end),n,'ULA');%randn(m,n);
% Anorm = A*diag(1./vecnorm(A));
% AtA = Anorm'*Anorm; AtA_tmp = abs(AtA);
% AtA_tmp(1:n+1:end)=0;
% mc = max(max(AtA_tmp));

% sp_l2error = zeros(ITER, length(L)); sp_suppdist = zeros(ITER, length(L));
% sp_orig_l2error = zeros(ITER, length(L)); sp_orig_suppdist = zeros(ITER, length(L));
% lsomp_l2error = zeros(ITER, length(L)); lsomp_suppdist = zeros(ITER, length(L));
seqsbl_l2error = zeros(ITER, length(L)); seqsbl_suppdist = zeros(ITER, length(L)); seqsbl_timecomp = zeros(ITER, length(L)); seqsbl_mse=zeros(ITER, length(L));
seqsbl_nvarNotgivenwvarby10_l2error = zeros(ITER, length(L)); seqsbl_nvarNotgivenwvarby10_suppdist = zeros(ITER, length(L)); seqsbl_nvarNotgivenwvarby10_timecomp = zeros(ITER, length(L)); seqsbl_nvarNotgivenwvarby10_mse=zeros(ITER, length(L));
seqsbl_nvarNotgivenwvar10_l2error = zeros(ITER, length(L)); seqsbl_nvarNotgivenwvar10_suppdist = zeros(ITER, length(L)); seqsbl_nvarNotgivenwvar10_timecomp = zeros(ITER, length(L)); seqsbl_nvarNotgivenwvar10_mse=zeros(ITER, length(L));
nomp_mse=zeros(ITER, length(L));
seqsbl_targetsj_l2error = zeros(ITER, length(L)); seqsbl_targetsj_suppdist = zeros(ITER, length(L)); seqsbl_targetsj_timecomp = zeros(ITER, length(L));
seqsbl_ximprov_l2error = zeros(ITER, length(L)); seqsbl_ximprov_suppdist = zeros(ITER, length(L)); seqsbl_ximprov_timecomp = zeros(ITER, length(L));
redcompomp_l2error = zeros(ITER, length(L)); redcompomp_suppdist = zeros(ITER, length(L)); redcompomp_timecomp = zeros(ITER, length(L));
omp_l2error = zeros(ITER, length(L)); omp_suppdist = zeros(ITER, length(L)); omp_timecomp = zeros(ITER, length(L));
% mp_l2error = zeros(ITER, length(L)); mp_suppdist = zeros(ITER, length(L));
% weakmp_l2error = zeros(ITER, length(L)); weakmp_suppdist = zeros(ITER, length(L));
% thresh_l2error = zeros(ITER, length(L)); thresh_suppdist = zeros(ITER, length(L));
% L1_l2error = zeros(ITER, length(L)); L1_suppdist = zeros(ITER, length(L));
% RL1_l2error = zeros(ITER, length(L)); RL1_suppdist = zeros(ITER, length(L));
% reg_IRLS_l2error = zeros(ITER, length(L)); reg_IRLS_suppdist = zeros(ITER, length(L));
% SBL_l2error = zeros(ITER, length(L)); SBL_suppdist = zeros(ITER, length(L));

for iter=1:ITER
    iter
    switch datagen
        case 0
            nonzero_x = randn(suppsizemax,L(end))+1j*randn(suppsizemax,L(end));
        case 5
            nonzero_x = sqrt(3)*rand(suppsizemax,1).*(randn(suppsizemax,1)+1j*randn(suppsizemax,1));
        case 1
            nonzero_x = (rand(suppsizemax,1)+1).*(2*(rand(suppsizemax,1)>0.5)-1);
        case 2
            nonzero_x = randn(suppsizemax,1);
        case 3
            nonzero_x = 2*(rand(suppsizemax,1)>0.5)-1;
        case 4
            nonzero_x = trnd(1,suppsizemax,1);
    end
    noise_vec=randn(m(end),L(end))+1j*randn(m(end),L(end));
    permute_n=randsample(n, n); % without replacement
    u_perturb=(2*rand([1,suppsizemax])-1)/n;% off-grid perturb
    for l_iter=1:length(L)
        L_test=L(l_iter);
        for m_iter=1:length(m)
            m_test=m(m_iter);
            A_test=A(1:m_test,:);
            for sep_iter=1:length(sep)
                [suppfull,loop_time] = min_separate(permute_n, suppsizemax, sep(sep_iter));%randsample(n, suppsizemax);
                %         loop_time
                xfull = zeros(n,L_test);
                xfull(suppfull,:)=nonzero_x(:,1:L_test);
                for isuppsize=suppsizemax:suppsizemax
                    if iter==359 && isuppsize==1
                        disp('hi')
                    end
                    suppsize = isuppsize;
                    supp = suppfull(1:suppsize);
                    x = xfull;
                    u_actual=u; u_actual(supp)=u_actual(supp)+u_perturb;
                    A_actual=exp(-1j*pi*(0:m-1)'*u_actual);
                    y = sqrt(s_var/2)*A_actual*x+sqrt(w_var/2)*noise_vec(1:m_test,1:L_test);
                    
                    %% NOMP
                    p_fa = 1e-2;
                    tau = w_var * ( log(m) - log( log(1/(1-p_fa)) ) );
                    [omegaList, gainList, residueList] = extractSpectrum(y, eye(m), tau, n/m);
                    estomega=omegaList/pi;
                    estomega(estomega>=1)=estomega(estomega>=1)-2;
                    estomega=-estomega;
                    err_mat = repmat(estomega, 1, suppsize)-repmat(u_actual(supp), length(estomega), 1);
                    [err_vec, ind_vec] = min(abs(err_mat));
                    nomp_mse(iter,l_iter)=mean(err_vec.^2);
                    
                    %% Sequential SBL (Computationally May Be Improved Further)
%                     % Initialization
%                     
%                     % Dimension Reduction
%                     [Q1, ~] = qr(y',0);
%                     yred=y*Q1; Lred = size(yred, 2);
%                     
%                     %                 candidateset=1:n;
%                     qj=zeros(n,Lred);
%                     qsj=zeros(n,1);
%                     sj=zeros(n,1);
%                     sigma_n=w_var; % noiseless case, can be adjusted
%                     gamma_est=zeros(n,1);
%                     kpset = zeros(suppsize,1);
%                     Cinv=eye(m_test)/sigma_n;
%                     CiA=Cinv*A_test;
%                     xhat = zeros(n,Lred);
%                     w_norm=zeros(suppsize,1);
%                     w_prev1mat=zeros(m_test,suppsize);
%                     % Main Loop
%                     tic
% %                     profile on
%                     for p=1:suppsize
%                         if p==1
%                             qj=A_test'*(yred/sigma_n);
%                             qsj=sum(conj(qj).*qj,2)/L_test;% abs(qj).^2;
%                             sj=(m_test/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
%                         end
%                         if p==1
%                             [val,kp]=max(qsj);
%                             val=sigma_n*val/m_test;
%                         else
%                             [val,kp]=max(qsj./sj);
%                         end
%                         if val>1
%                             %                         kp=l_prev;
%                             kpset(p)=kp;
%                             %                         candidateset=candidateset(candidateset~=kp);%[candidateset(1:l_prev-1) candidateset(l_prev+1:end)];%
%                             gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
%                             
%                             %                         w_norm=sqrt(1/gamma_est(kp)+sj(kp));
%                             %                         %                     Cinv_prev=Cinv;
%                             %                         %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
%                             %                         %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
%                             %                         w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*A_test(:,kp));
%                             %                         w_prev1=w_prev1/w_norm;
%                             %                         w_prev1mat(:,p)=w_prev1;
%                             %                         w_prev2=A_test'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
%                             %                         qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                             %                         qsj=conj(qj).*qj;% abs(qj).^2;
%                             %                         sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
%                             
%                             w_norm(p)=1/(1/gamma_est(kp)+sj(kp));
%                             w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*((w_prev1mat(:,1:p-1)'*A_test(:,kp)).*w_norm(1:p-1,1));
%                             w_prev1mat(:,p)=w_prev1;
%                             w_prev2tmp=(A_test'*w_prev1);
%                             w_prev2=w_prev2tmp*(w_norm(p)*qj(kp,:)); %(qj(kp)/w_norm)*w_prev1;
%                             qj=qj-w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                             qsj=sum(conj(qj).*qj,2)/L_test;% abs(qj).^2;
%                             %                         sj(candidateset)=sj(candidateset)-w_norm(p)*(conj(A_test(:,candidateset)'*w_prev1).*((A_test(:,candidateset)'*w_prev1)));
%                             sj=sj-w_norm(p)*(conj(w_prev2tmp).*w_prev2tmp);%abs(w_prev2).^2;
%                         else
%                             warning('Seq. SBL did not add new column')
%                         end
%                     end
% %                     u_grid_updated=u;
%                     [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,suppsize,u,A_test,(0:m_test-1),L_test,m_test,yred*yred',sigma_n);
%                     
%                     xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,Lred).*qj(kpset(kpset>0),:);
% %                     profile off
%                     %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
%                     seqsbl_timecomp(iter,l_iter)=toc;
%                     err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
%                     [err_vec, ind_vec] = min(abs(err_mat));
%                     seqsbl_mse(iter,l_iter)=mean(err_vec.^2);
%                     seqsbl_l2error(iter,l_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
%                     seqsbl_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% Sequential SBL (Lambda set to w_var/10)
                    
%                     % Dimension Reduction
%                     [Q1, ~] = qr(y',0);
%                     yred=y*Q1; Lred = size(yred, 2);
%                     
%                     %                 candidateset=1:n;
%                     qj=zeros(n,Lred);
%                     qsj=zeros(n,1);
%                     sj=zeros(n,1);
%                     sigma_n=w_var/10; % noiseless case, can be adjusted
%                     gamma_est=zeros(n,1);
%                     kpset = zeros(suppsize,1);
%                     Cinv=eye(m_test)/sigma_n;
%                     CiA=Cinv*A_test;
%                     xhat = zeros(n,Lred);
%                     w_norm=zeros(suppsize,1);
%                     w_prev1mat=zeros(m_test,suppsize);
%                     % Main Loop
%                     tic
%                     %             profile on
%                     for p=1:suppsize
%                         if p==1
%                             qj=A_test'*(yred/sigma_n);
%                             qsj=sum(conj(qj).*qj,2)/L_test;% abs(qj).^2;
%                             sj=(m_test/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
%                         end
%                         if p==1
%                             [val,kp]=max(qsj);
%                             val=sigma_n*val/m_test;
%                         else
%                             [val,kp]=max(qsj./sj);
%                         end
%                         if val>1
%                             %                         kp=l_prev;
%                             kpset(p)=kp;
%                             %                         candidateset=candidateset(candidateset~=kp);%[candidateset(1:l_prev-1) candidateset(l_prev+1:end)];%
%                             gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
%                             
%                             %                         w_norm=sqrt(1/gamma_est(kp)+sj(kp));
%                             %                         %                     Cinv_prev=Cinv;
%                             %                         %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
%                             %                         %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
%                             %                         w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*A_test(:,kp));
%                             %                         w_prev1=w_prev1/w_norm;
%                             %                         w_prev1mat(:,p)=w_prev1;
%                             %                         w_prev2=A_test'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
%                             %                         qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                             %                         qsj=conj(qj).*qj;% abs(qj).^2;
%                             %                         sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
%                             
%                             w_norm(p)=1/(1/gamma_est(kp)+sj(kp));
%                             w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*((w_prev1mat(:,1:p-1)'*A_test(:,kp)).*w_norm(1:p-1,1));
%                             w_prev1mat(:,p)=w_prev1;
%                             w_prev2tmp=(A_test'*w_prev1);
%                             w_prev2=w_prev2tmp*(w_norm(p)*qj(kp,:)); %(qj(kp)/w_norm)*w_prev1;
%                             qj=qj-w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                             qsj=sum(conj(qj).*qj,2)/L_test;% abs(qj).^2;
%                             %                         sj(candidateset)=sj(candidateset)-w_norm(p)*(conj(A_test(:,candidateset)'*w_prev1).*((A_test(:,candidateset)'*w_prev1)));
%                             sj=sj-w_norm(p)*(conj(w_prev2tmp).*w_prev2tmp);%abs(w_prev2).^2;
%                         else
%                             warning('Seq. SBL did not add new column')
%                         end
%                     end
% %                     u_grid_updated=u;
%                     [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,suppsize,u,A_test,(0:m_test-1),L_test,m_test,yred*yred',sigma_n);
%                     
%                     xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,Lred).*qj(kpset(kpset>0),:);
%                     %             xhat(kpset(1:p))=gamma_est(kpset(1:p)).*qj(kpset(1:p));
%                     %             profile off
%                     %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
%                     seqsbl_nvarNotgivenwvarby10_timecomp(iter,l_iter)=toc;
%                     err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
%                     [err_vec, ind_vec] = min(abs(err_mat));
%                     seqsbl_nvarNotgivenwvarby10_mse(iter,l_iter)=mean(err_vec.^2);
%                     seqsbl_nvarNotgivenwvarby10_l2error(iter,l_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
%                     seqsbl_nvarNotgivenwvarby10_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% Sequential SBL (Lambda set to w_var*10)
                    
%                     % Dimension Reduction
%                     [Q1, ~] = qr(y',0);
%                     yred=y*Q1; Lred = size(yred, 2);
%                     
%                     %                 candidateset=1:n;
%                     qj=zeros(n,Lred);
%                     qsj=zeros(n,1);
%                     sj=zeros(n,1);
%                     sigma_n=10*w_var; % noiseless case, can be adjusted
%                     gamma_est=zeros(n,1);
%                     kpset = zeros(suppsize,1);
%                     Cinv=eye(m_test)/sigma_n;
%                     CiA=Cinv*A_test;
%                     xhat = zeros(n,Lred);
%                     w_norm=zeros(suppsize,1);
%                     w_prev1mat=zeros(m_test,suppsize);
%                     % Main Loop
%                     tic
%                     %             profile on
%                     for p=1:suppsize
%                         if p==1
%                             qj=A_test'*(yred/sigma_n);
%                             qsj=sum(conj(qj).*qj,2)/L_test;% abs(qj).^2;
%                             sj=(m_test/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
%                         end
%                         if p==1
%                             [val,kp]=max(qsj);
%                             val=sigma_n*val/m_test;
%                         else
%                             [val,kp]=max(qsj./sj);
%                         end
%                         if val>1
%                             %                         kp=l_prev;
%                             kpset(p)=kp;
%                             %                         candidateset=candidateset(candidateset~=kp);%[candidateset(1:l_prev-1) candidateset(l_prev+1:end)];%
%                             gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
%                             
%                             %                         w_norm=sqrt(1/gamma_est(kp)+sj(kp));
%                             %                         %                     Cinv_prev=Cinv;
%                             %                         %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
%                             %                         %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
%                             %                         w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*A_test(:,kp));
%                             %                         w_prev1=w_prev1/w_norm;
%                             %                         w_prev1mat(:,p)=w_prev1;
%                             %                         w_prev2=A_test'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
%                             %                         qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                             %                         qsj=conj(qj).*qj;% abs(qj).^2;
%                             %                         sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
%                             
%                             w_norm(p)=1/(1/gamma_est(kp)+sj(kp));
%                             w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*((w_prev1mat(:,1:p-1)'*A_test(:,kp)).*w_norm(1:p-1,1));
%                             w_prev1mat(:,p)=w_prev1;
%                             w_prev2tmp=(A_test'*w_prev1);
%                             w_prev2=w_prev2tmp*(w_norm(p)*qj(kp,:)); %(qj(kp)/w_norm)*w_prev1;
%                             qj=qj-w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                             qsj=sum(conj(qj).*qj,2)/L_test;% abs(qj).^2;
%                             %                         sj(candidateset)=sj(candidateset)-w_norm(p)*(conj(A_test(:,candidateset)'*w_prev1).*((A_test(:,candidateset)'*w_prev1)));
%                             sj=sj-w_norm(p)*(conj(w_prev2tmp).*w_prev2tmp);%abs(w_prev2).^2;
%                         else
%                             warning('Seq. SBL did not add new column')
%                         end
%                     end
% %                     u_grid_updated=u;
%                     [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,suppsize,u,A_test,(0:m_test-1),L_test,m_test,yred*yred',sigma_n);
%                     
%                     xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,Lred).*qj(kpset(kpset>0),:);
%                     %             xhat(kpset(1:p))=gamma_est(kpset(1:p)).*qj(kpset(1:p));
%                     %             profile off
%                     %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
%                     seqsbl_nvarNotgivenwvar10_timecomp(iter,l_iter)=toc;
%                     err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
%                     [err_vec, ind_vec] = min(abs(err_mat));
%                     seqsbl_nvarNotgivenwvar10_mse(iter,l_iter)=mean(err_vec.^2);
%                     seqsbl_nvarNotgivenwvar10_l2error(iter,l_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
%                     seqsbl_nvarNotgivenwvar10_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% 2.OMP
                    %                 %         p=0;
                    %                 bp = y;
                    %                 %         bpsqnorm = vecnorm(y)^2;
                    %                 qkp = zeros(m_test,0);
                    %                 %         Atbp = A'*bp;
                    %                 Qp = qkp;
                    %                 kpset = zeros(1,suppsize);
                    %                 xhat = zeros(n,1);
                    %                 % OMP Support Recovery
                    %                 tic
                    %                 for p=1:suppsize
                    %                     %             tic
                    %                     %         while bpsqnorm>epsilon^2
                    %                     Atbp = A_test'*bp;
                    %                     Atbpabs = abs(Atbp);
                    %                     [~, kp] = max(Atbpabs);
                    %                     kpset(p) = kp; %kpset = union(kpset,kp);
                    %                     qkp = A_test(:,kp)-Qp*(Qp'*A_test(:,kp));
                    %                     qkp = qkp/vecnorm(qkp);
                    %                     Qp = [Qp qkp];
                    %                     %             bpsqnorm = bpsqnorm-(abs(qkp'*bp))^2;
                    %                     bp = bp-qkp*(qkp'*bp);
                    %                     %             toc
                    %                     %             p = p+1;
                    %                 end
                    %                 xhat(kpset) = pinv(A_test(:,kpset))*y;
                    %                 omp_timecomp(iter,l_iter)=toc;
                    %                 omp_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %                 omp_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% 2a.OMP (Reduced Complexity)
                    %                 % Initialization
                    %                 bp = y;
                    %                 Qp = zeros(m_test,suppsize);
                    %                 kpset = zeros(1,suppsize);
                    %                 ynorm=zeros(suppsize,1);
                    %                 Rmat=zeros(suppsize);
                    %                 xhat = zeros(n,1);
                    %                 %             candidateset=1:n;
                    %                 kp=0;
                    %                 % OMP Support Recovery
                    %                 tic
                    %                 for p=1:suppsize
                    %                     Atbp = A_test'*bp;
                    %                     Atbpabs = abs(Atbp);
                    %                     [~, kp] = max(Atbpabs); %kp=candidateset(kp);
                    %                     kpset(p) = kp;
                    %                     if p==1
                    %                         qkp = A_test(:,kp);
                    %                     else
                    %                         Rmat(1:p-1,p)=Qp(:,1:p-1)'*A_test(:,kp);
                    %                         qkp = A_test(:,kp)-Qp(:,1:p-1)*Rmat(1:p-1,p);
                    %                     end
                    %                     Rmat(p,p)=vecnorm(qkp);
                    %                     qkp = qkp/Rmat(p,p);
                    %                     Qp(:,p) = qkp;
                    %                     ynorm(p)=(qkp'*bp); % used later to solve Normal equations
                    %                     bp = bp-qkp*ynorm(p);
                    %                     %             toc
                    %                 end
                    %
                    %                 % Computing Sparse Vector x
                    %                 opts.UT=true;
                    %                 xhat(kpset) = linsolve(Rmat,ynorm,opts);
                    %                 redcompomp_timecomp(iter,l_iter)=toc;
                    %                 redcompomp_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %                 redcompomp_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    
                    %% 1.LS-OMP
                    %         p=0;
                    %         bp = y;
                    %         bpsqnorm = vecnorm(y)^2;
                    %         qkp = zeros(m,0);
                    %         Qp = qkp;
                    %         kpset = [];
                    %         while bpsqnorm>epsilon^2
                    %             nonkpset = setdiff(1:n,kpset);
                    %             qkpmat_tmp = zeros(m,length(nonkpset));
                    %             bpmat_tmp = zeros(m,length(nonkpset));
                    %             for i=1:length(nonkpset)
                    %                 qkp_tmp = A(:,nonkpset(i))-Qp*Qp'*A(:,nonkpset(i));
                    %                 qkp_tmp = qkp_tmp/vecnorm(qkp_tmp);
                    %                 qkpmat_tmp(:,i) = qkp_tmp;
                    %                 bpmat_tmp(:,i) = bp-qkp_tmp*(qkp_tmp'*bp);
                    %             end
                    %             bpmatnorm_tmp = vecnorm(bpmat_tmp);
                    %             [~, kp] = min(bpmatnorm_tmp);
                    %             kpset = union(kpset,nonkpset(kp));
                    %             Qp = [Qp qkpmat_tmp(:,kp)];
                    %             bpsqnorm = bpmatnorm_tmp(kp)^2;
                    %             bp = bpmat_tmp(:,kp);
                    %             p = p+1;
                    %         end
                    %         xhat = zeros(n,1);
                    %         xhat(kpset) = pinv(A(:,kpset))*y;
                    %         lsomp_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         lsomp_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    
                    %% 3.MP
                    %         p=0;
                    %         bp = y;
                    %         bpsqnorm = vecnorm(y)^2;
                    %         kpset = [];
                    %         while bpsqnorm>epsilon^2
                    %             if p==0
                    %                 Atbp = A'*bp;
                    %             else
                    %                 Atbp = Atbp-AtA(:,kp)*Atbp(kp);
                    %             end
                    %             Atbpabs = abs(Atbp);
                    %             [~, kp] = max(Atbpabs);
                    %             kpset = union(kpset,kp);
                    %             bp = bp-A(:,kp)*Atbp(kp);
                    %             bpsqnorm = bpsqnorm-Atbpabs(kp)^2;
                    %             p = p+1;
                    %         end
                    %         xhat = zeros(m,1);
                    %         xhat(kpset) = pinv(A(:,kpset))*y;
                    %         mp_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         mp_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% 4.Weak-MP
                    %         p=0;
                    %         bp = y;
                    %         bpsqnorm = vecnorm(y)^2;
                    %         kpset = [];
                    %         t = 0.5;
                    %         while bpsqnorm>epsilon^2
                    %             i = 1;
                    %             while i<=m && abs(A(:,i)'*bp)<=t*sqrt(bpsqnorm)
                    %                 i = i+1;
                    %             end
                    %             if i>m
                    %                 [~,i] = max(abs(A'*bp));
                    %             end
                    %             kp = i; Atbp = A(:,kp)'*bp; Atbpabs = abs(Atbp);
                    %             kpset = union(kpset,kp);
                    %             bp = bp-A(:,kp)*Atbp;
                    %             bpsqnorm = bpsqnorm-Atbpabs^2;
                    %             p = p+1;
                    %         end
                    %         xhat = zeros(m,1);
                    %         xhat(kpset) = pinv(A(:,kpset))*y;
                    %         weakmp_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         weakmp_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% Subspace Pursuit
                    %         p=0;
                    %         s=suppsizemax;
                    %         bp = y;
                    %         bpsqnorm = vecnorm(y)^2;
                    %         Ip = []; kpset = [];
                    %         while bpsqnorm>epsilon^2
                    %             Atbp = A'*bp;
                    %             Atbpabs = abs(Atbp);
                    %             [~, Ip_tilde] = maxk(Atbpabs,s);
                    %             Ip_tilde = union(Ip_tilde,Ip);
                    %             xp_tilde = pinv(A(:,Ip_tilde))*y;
                    %             [~,Ip] = maxk(abs(xp_tilde),s); Ip = Ip_tilde(Ip);
                    %             kpset = Ip;
                    %             xp = pinv(A(:,Ip))*y;
                    %             bp = y-A(:,Ip)*xp;
                    % %             bpsqnorm = vecnorm(bp)^2;
                    %             p = p+1;
                    %             clf
                    %             xhat = zeros(n,1);
                    %             xhat(kpset) = xp;
                    %             plot(abs(xhat)); hold on
                    %             plot(abs(x))
                    %             xlabel('Component Index i')
                    %             ylabel('Absolute value |x_i|')
                    %             title({['Iteration no.=' num2str(iter) 'Support size=' num2str(isuppsize)],...
                    %                 ['Ip={' num2str(sort(Ip)') '},  |Ip\cap Supp|/|Supp|=' num2str(length(intersect(supp,Ip))/isuppsize)],...
                    %                 ['Ip\_tilde={' num2str(sort(Ip_tilde)') '},  |Ip\_tilde\cap Supp|/|Supp|=' num2str(length(intersect(supp,Ip_tilde))/isuppsize)],...
                    %                 ['Suppport={' num2str(sort(supp)') '}']})
                    %             legend('estimated x', 'actual x')
                    %             drawnow
                    %             pause(0.15)
                    %             if vecnorm(bp)^2/bpsqnorm>=1
                    %                 break
                    %             else
                    %                 bpsqnorm = vecnorm(bp)^2;
                    %             end
                    %         end
                    %         xhat = zeros(n,1);
                    %         xhat(kpset) = pinv(A(:,kpset))*y;
                    %         sp_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         sp_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% Subspace Pursuit Original Code
                    %         s=suppsizemax;
                    %         Rec = CSRec_SP(s,A,y);
                    %         xhat = Rec.x_hat; [~,kpset] = maxk(abs(xhat),s);
                    %         sp_orig_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         sp_orig_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% 5.Thresholding
                    %         p=0;
                    %         bp = y;
                    %         bpsqnorm = vecnorm(y)^2;
                    %         kpset = [];
                    %         % xp = zeros(m,1);
                    %         xp = pinv(A)*y;
                    %         %         while bpsqnorm>epsilon^2
                    %         [~, ind] = maxk(abs(xp+A'*bp),suppsize);
                    %         xp = zeros(m,1);
                    %         xp(ind) = pinv(A(:,ind))*y;
                    %         bp = y-A*xp;
                    %         bpsqnorm = vecnorm(bp)^2;
                    %         p = p+1;
                    % %         end
                    %         xhat = xp;
                    %         kpset = ind;
                    %         thresh_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         thresh_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    %% 6. L1-minimization (Using MATLAB linprog)
                    %         xhat = linprog(ones(2*m,1),[],[],[A -A],y,zeros(2*m,1));
                    %         xhat = xhat(1:m)-xhat(m+1:end);
                    %         kpset = find(abs(xhat)>1e-4);
                    %         L1_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         L1_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% 7. Reweighted-L1 (Candes, Wakin and Boyd)
                    
                    %         reg_eps = 0.1;
                    %         w = ones(m,1);
                    %         for i=1:4
                    %             W = diag(w);
                    %             zhat = linprog(ones(2*m,1),[],[],[A/W -A/W],y,zeros(2*m,1));
                    %             zhat = zhat(1:m)-zhat(m+1:end);
                    %             xhat = W\zhat;
                    %             w = 1./(abs(xhat)+reg_eps);
                    %         end
                    %         kpset = find(abs(xhat)>1e-4);
                    %         RL1_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         RL1_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% 8. Reweighted-L2 (Chartrand and Yin)
                    %         reg_eps = 1;
                    %         p = 0;
                    %         u_new = pinv(A)*y;
                    %         while reg_eps>1e-8
                    %             u_old = u_new;
                    %             w = (u_old.^2+reg_eps).^(p/2-1);
                    %             Q = diag(1./w);
                    %             u_new = Q*A'*((A*Q*A')\y);
                    %             if vecnorm(u_new-u_old)/vecnorm(u_old)<sqrt(reg_eps)/100
                    %                 reg_eps = reg_eps/10;
                    %             end
                    %         end
                    %         xhat = u_new; kpset = find(abs(xhat)>1e-4);
                    %         reg_IRLS_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         reg_IRLS_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% 9. Sparse Bayesian Learning (EM-SBL)
                    %         ssigma = 1e-4;
                    %         gamma = ones(m,1);
                    %         for i=1:20
                    %             weightd_A = (repmat(gamma,1,n).*(A'));
                    %             W_i = weightd_A/(A*weightd_A+ssigma*eye(n));
                    %             mu_x = W_i*y;
                    %             Sigma_x_diag = gamma-sum(W_i.*weightd_A,2);
                    %             gamma = abs(mu_x).^2+Sigma_x_diag;
                    % %             plot(gamma)
                    % %             title(['i=' num2str(i)])
                    % %             drawnow
                    % %             pause(0.1)
                    %         end
                    %         xhat = mu_x; kpset = find(abs(xhat)>1e-4);
                    %         SBL_l2error(iter,l_iter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                    %         SBL_suppdist(iter,l_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                end
            end
        end
    end
end
% save('mpVssp.mat')
% unique_bound = 0.5*(1+1/mc);
figure
load('Gridless_seq_SBL_ITER100inner10GtwoITER.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '-ob', 'LineWidth', 2, 'MarkerSize',15);
hold on
clear all
load('Gridless_seq_SBL_ITER100inner10GoneITER.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '-xb', 'LineWidth', 2, 'MarkerSize',10)
clear all
load('Gridless_seq_SBL_ITER100inner10GtenITER.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '-sb', 'LineWidth', 1, 'MarkerSize',10)
clear all
load('Grid_seq_SBL_ITER100n200.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '--dr', 'LineWidth', 2, 'MarkerSize',10)
clear all
load('Grid_seq_SBL_ITER100n2000sep100gridpts.mat')
loglog(L, sqrt(mean(seqsbl_mse)), '--pr', 'LineWidth', 2, 'MarkerSize',10)


% Stochastic CRB equal power sources assumed
Sigma_s=diag(sqrt(s_var))*eye(suppsizemax)*diag(sqrt(s_var))';
num_factor=w_var./(2*L);
Acrb = exp(-1j*pi*(0:m-1)'*u_actual(supp));
Dcrb = -1j*(0:m-1)'.*Acrb;
Ry_inv = eye(m)/((Acrb*Sigma_s*Acrb')+w_var*eye(m));
wonumfactor_crb_psi = eye(suppsizemax)/(real(Dcrb'*(eye(m)-((Acrb/(Acrb'*Acrb))*Acrb'))*Dcrb).*((Sigma_s*Acrb'*Ry_inv*Acrb*Sigma_s).'));
%         % in theta degrees space
%         sqrcrb_theta(SNRiter)=sqrt(mean(diag(diag(180./(pi^2*sqrt(1-sind(theta).^2)))*crb_psi*(diag(180./(pi^2*sqrt(1-sind(theta).^2))).'))));
% in u space
sqrcrb_u_theta=real(sqrt(num_factor)*sqrt(mean(diag((1/pi)*wonumfactor_crb_psi*((1/pi).')))));

ax=loglog(L,sqrcrb_u_theta, '--k', 'LineWidth', 1);
legend({'Gridless LWS-SBL: ITER=2', 'Gridless LWS-SBL: ITER=1', 'Gridless LWS-SBL: ITER=10','LWS-SBL: n=200','LWS-SBL: n=2000','CRB'}, 'NumColumns',2)
ylabel('RMSE in u-space')
xlabel('Number of Snapshots (L)')
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',16)
xticks([1:5 10 20:20:60 100 250 500 1000])
grid on

% figure
% subplot(211)
% histogram(mc)
% title('Histogram')
% xlabel('Mutual Coherence (\mu(A))')
% ylabel('Frequency')
% grid on
% subplot(212)
% plot(0.5*(1+1./mc))
% xlabel('Iteration number')
% ylabel('0.5(1+1/\mu(A))')
% grid on

%% Functions

function [suppfull,loop_time]=min_separate(permute_n, suppsizemax, sep)
supp_list=zeros(suppsizemax,1);
supp_list(1)=permute_n(1);
list_len=1;
loop_time=1;
while list_len<suppsizemax
    loop_time=loop_time+1;
    supp_tmp=permute_n(loop_time);
    if min(abs(supp_list-supp_tmp))>=sep
        list_len=list_len+1;
        supp_list(list_len)=supp_tmp;
    end
end
suppfull=supp_list;
end

function [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,L,M,unRyoyo,lambda)
% Sigma=Agrid_updated*diag(gamma_est)*Agrid_updated'+lambda*eye(M); % Sigma minus grid point i
% iSigma=eye(M)/Sigma;
for iterGdPtAdPks=1:2 % grid point adjustment around peaks iteration
    G=length(gamma_est); G_inner=10*G;
    [pks,locs]=findpeaks1(gamma_est);
    [mpks,mlocs]=maxk(pks, K);
    K_est=length(mlocs);
    m_candidates=sort(locs(mlocs));% one source
    u_est=u_grid_updated(m_candidates);
    for iterK=1:K_est
        m_iterK=m_candidates(iterK);
        if m_iterK>1; left_delta=u_grid_updated(m_iterK)-u_grid_updated(m_iterK-1); else; left_delta=u_grid_updated(m_iterK)+1; end
        if m_iterK<G; right_delta=u_grid_updated(m_iterK+1)-u_grid_updated(m_iterK); else; left_delta=1-u_grid_updated(m_iterK); end
        delta=left_delta/2+right_delta/2; resSeqSBL=delta/G_inner;
        u_candidates=linspace(u_est(iterK)-left_delta/2,u_est(iterK),floor(left_delta/2/resSeqSBL+1));
        u_candidates=union(u_candidates,linspace(u_est(iterK),u_est(iterK)+right_delta/2,floor(right_delta/2/resSeqSBL+1)));
        Gp=length(u_candidates);
        gamma_updated=zeros(1,Gp);
        I_gamma_opt=zeros(1,Gp);
        
        % Rank one update
%         wi_vec_remove=iSigma*Agrid_updated(:,m_iterK);
%         iSigma_mi=iSigma-wi_vec_remove*wi_vec_remove'/(-1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_remove);
        cmpgdind=setdiff(m_candidates,m_iterK); %setdiff(1:G,m_iterK); % complement set of grid index
        Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(M); % Sigma minus grid point i
        iSigma_mi=eye(M)/Sigma_mi;
        
        Aadpt_grid=exp(-1j*pi*spos'*u_candidates);
        q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(((unRyoyo/L)*iSigma_mi)*Aadpt_grid)));
        s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
        q_sq_by_s_i=q_i_sq./s_i;
        gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
        I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
        [valneigh,indneigh]=min(I_gamma_opt);
        if valneigh<0
            gamma_est(m_iterK)=gamma_updated(indneigh);
            u_grid_updated(m_iterK)=u_candidates(indneigh);
            Agrid_updated(:,m_iterK)=exp(-1j*pi*spos'*u_candidates(indneigh));
%             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
%             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
        end
    end
end
end

function [PKS,LOCS]=findpeaks1(Y) % assumes Y is a non-negative vector
diff1=diff([0 reshape(Y,1,[]) 0]);
LOCS=find(diff1(1:end-1)>0 & diff1(2:end)<0);
PKS=Y(LOCS);
end