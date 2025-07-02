% Array Processing Problem: Seq. SBL vs Matching Pursuit type algorithms
% Impemented: Seq. SBL 1.LS-OMP 2.OMP 3.MP 4.Weak-MP 5.Thresholding

clear all
rng(1)
addpath('G:\My Drive\Research\2_Code\Sparse-Bayesian-Learning\Gridless_SBL\plots_asilomar_draft\Simulation_study_ICASSP\Extension_work\NOMP\wcslspectralestimation-continuous-frequency-estimation-779a5e217053\')
addpath atomsbl
m = [5:5:50 75:25:100]; %5:5:30; %
n = 500;
L=500; % single snapshot
suppsizemax = 2;
datagen = 0;
% epsilon = 1e-2;
% r = Inf; %suppsize;
ITER = 100;
NewtonSteps=10;
SNR=30;%[-40:5:20]; % in dB
s_var=1;
if datagen==0; Amp_vec=ones(suppsizemax,1); else; Amp_vec=sqrt(3)*sort(rand(suppsizemax,1),'descend'); end% amplitude vector
sep=1; % make sure (max. sep X 2 X suppsizemax)<=n to ensure sep is always possible

% Grid Initialize
[A, u] = create_dictionary(m(end),n,'ULA');%randn(m,n);
permute_n=[n/2+1 n/2+11];%randsample(n, n); % without replacement
u_perturb=(4*rand([1,suppsizemax])-2)/n;% off-grid perturb
% Anorm = A*diag(1./vecnorm(A));
% AtA = Anorm'*Anorm; AtA_tmp = abs(AtA);
% AtA_tmp(1:n+1:end)=0;
% mc = max(max(AtA_tmp));

% sp_l2error = zeros(ITER, length(m)); sp_suppdist = zeros(ITER, length(m));
% sp_orig_l2error = zeros(ITER, length(m)); sp_orig_suppdist = zeros(ITER, length(m));
% lsomp_l2error = zeros(ITER, length(m)); lsomp_suppdist = zeros(ITER, length(m));
seqsbl_l2error = zeros(ITER, length(m)); seqsbl_suppdist = zeros(ITER, length(m)); seqsbl_timecomp = zeros(ITER, length(m)); seqsbl_mse=zeros(ITER, length(m));
newton_seqsbl_suppdist = zeros(ITER, length(m)); newton_seqsbl_timecomp = zeros(ITER, length(m)); newton_seqsbl_mse=zeros(ITER, length(m));
newton_prerefine_seqsbl_suppdist = zeros(ITER, length(m)); newton_prerefine_seqsbl_timecomp = zeros(ITER, length(m)); newton_prerefine_seqsbl_mse=zeros(ITER, length(m));
seqsbl_nvarNotgivenwvarby10_l2error = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvarby10_suppdist = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvarby10_timecomp = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvarby10_mse=zeros(ITER, length(m));
seqsbl_nvarNotgivenwvar10_l2error = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvar10_suppdist = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvar10_timecomp = zeros(ITER, length(m)); seqsbl_nvarNotgivenwvar10_mse=zeros(ITER, length(m));
nomp_mse=zeros(ITER, length(m)); nomp_iterrun=zeros(ITER, length(m)); nomp_timecomp = zeros(ITER, length(m));
seqsbl_targetsj_l2error = zeros(ITER, length(m)); seqsbl_targetsj_suppdist = zeros(ITER, length(m)); seqsbl_targetsj_timecomp = zeros(ITER, length(m));
seqsbl_ximprov_l2error = zeros(ITER, length(m)); seqsbl_ximprov_suppdist = zeros(ITER, length(m)); seqsbl_ximprov_timecomp = zeros(ITER, length(m));
redcompomp_l2error = zeros(ITER, length(m)); redcompomp_suppdist = zeros(ITER, length(m)); redcompomp_timecomp = zeros(ITER, length(m));
omp_l2error = zeros(ITER, length(m)); omp_suppdist = zeros(ITER, length(m)); omp_timecomp = zeros(ITER, length(m));
atomsbl_mse = zeros(ITER, length(m));
% mp_l2error = zeros(ITER, length(m)); mp_suppdist = zeros(ITER, length(m));
% weakmp_l2error = zeros(ITER, length(m)); weakmp_suppdist = zeros(ITER, length(m));
% thresh_l2error = zeros(ITER, length(m)); thresh_suppdist = zeros(ITER, length(m));
% L1_l2error = zeros(ITER, length(m)); L1_suppdist = zeros(ITER, length(m));
% RL1_l2error = zeros(ITER, length(m)); RL1_suppdist = zeros(ITER, length(m));
% reg_IRLS_l2error = zeros(ITER, length(m)); reg_IRLS_suppdist = zeros(ITER, length(m));
% SBL_l2error = zeros(ITER, length(m)); SBL_suppdist = zeros(ITER, length(m));

for iter=1:ITER
    iter
    switch datagen
        case 0
            nonzero_x = randn(suppsizemax,L(end))+1j*randn(suppsizemax,L(end));
        case 5
            nonzero_x = Amp_vec.*(randn(suppsizemax,L(end))+1j*randn(suppsizemax,L(end)));
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
    for snr_iter=1:length(SNR)
        snr_test=SNR(snr_iter);
        w_var=s_var/10^(snr_test/10);
        for m_iter=1:length(m)
            disp(['iter=' num2str(iter) ', m=' num2str(m(m_iter))])
            m_test=m(m_iter);
            A_test=A(1:m_test,:);
            for sep_iter=1:length(sep)
                [suppfull,loop_time] = min_separate(permute_n, suppsizemax, sep(sep_iter));%randsample(n, suppsizemax);
                %         loop_time
                xfull = zeros(n,L);
                xfull(suppfull,:)=nonzero_x;
                for isuppsize=suppsizemax:suppsizemax
                    if iter==12 && m_iter==4
                        disp('hi')
                    end
                    suppsize = isuppsize;
                    supp = suppfull(1:suppsize);
                    x = xfull;
                    u_actual=u; u_actual(supp)=[-0.004000407656871 0.036001424899390];%u_actual(supp)+u_perturb;
                    A_actual=exp(-1j*pi*(0:m_test-1)'*u_actual);
                    y = sqrt(s_var/2)*A_actual*x+sqrt(w_var/2)*noise_vec(1:m_test,1:L);
                    
                    %% NOMP
                    tic
                    p_fa = 1e-2;
                    tau = L*w_var * ( log(m_test) - log( log(1/(1-p_fa)) ) );
                    [omegaList, gainList, residueList] = extractSpectrum_MMV(y, eye(m_test), tau, n/m_test,1,NewtonSteps,suppsizemax);
                    nomp_timecomp(iter,m_iter) = toc;
                    estomega=omegaList/pi;
                    estomega(estomega>=1)=estomega(estomega>=1)-2;
                    estomega=-estomega;
                    if isempty(estomega)
                        estomega=0;
                    end
                    err_mat = repmat(estomega, 1, suppsize)-repmat(u_actual(supp), length(estomega), 1);
                    [err_vec, ind_vec] = min(abs(err_mat),[],1); % ind_vec pick estimated source
                    % Special error calculation
                    [~,ind_vec2]=min(err_vec); % ind_vec2 pick ground truth source
                    nomp_mse(iter,m_iter)=mean([err_vec(ind_vec2)^2 abs(err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
                    
%                     nomp_mse(iter,m_iter)=mean(err_vec.^2);
                    nomp_iterrun(iter,m_iter)=length(estomega);
                    
                    %% Sequential SBL (Computationally May Be Improved Further)
                    tic
                    % Initialization
                    algo_iter_match_nomp=length(estomega);
                    % Dimension Reduction
                    [Q1, ~] = qr(y',0);
                    yred=y*Q1; Lred = size(yred, 2);
                    
                    %                 candidateset=1:n;
                    qj=zeros(n,Lred);
                    qsj=zeros(n,1);
                    sj=zeros(n,1);
                    sigma_n=w_var; % noiseless case, can be adjusted
                    gamma_est=zeros(n,1);
                    kpset = zeros(algo_iter_match_nomp,1);
                    Cinv=eye(m_test)/sigma_n;
                    CiA=Cinv*A_test;
                    xhat = zeros(n,Lred);
                    w_norm=zeros(algo_iter_match_nomp,1);
                    w_prev1mat=zeros(m_test,algo_iter_match_nomp);
                    % Main Loop
%                     tic
                    %                     profile on
                    for p=1:algo_iter_match_nomp
                        if p==1
                            qj=A_test'*(yred/sigma_n);
                            qsj=sum(conj(qj).*qj,2)/L;% abs(qj).^2;
                            sj=(m_test/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
                        end
                        if p==1
                            [val,kp]=max(qsj);
                            val=sigma_n*val/m_test;
                        else
                            [val,kp]=max(qsj./sj);
                        end
                        if val>1
                            %                         kp=l_prev;
                            kpset(p)=kp;
                            %                         candidateset=candidateset(candidateset~=kp);%[candidateset(1:l_prev-1) candidateset(l_prev+1:end)];%
                            gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
                            
                            %                         w_norm=sqrt(1/gamma_est(kp)+sj(kp));
                            %                         %                     Cinv_prev=Cinv;
                            %                         %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
                            %                         %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
                            %                         w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*A_test(:,kp));
                            %                         w_prev1=w_prev1/w_norm;
                            %                         w_prev1mat(:,p)=w_prev1;
                            %                         w_prev2=A_test'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
                            %                         qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
                            %                         qsj=conj(qj).*qj;% abs(qj).^2;
                            %                         sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
                            
                            w_norm(p)=1/(1/gamma_est(kp)+sj(kp));
                            w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*((w_prev1mat(:,1:p-1)'*A_test(:,kp)).*w_norm(1:p-1,1));
                            w_prev1mat(:,p)=w_prev1;
                            w_prev2tmp=(A_test'*w_prev1);
                            w_prev2=w_prev2tmp*(w_norm(p)*qj(kp,:)); %(qj(kp)/w_norm)*w_prev1;
                            qj=qj-w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
                            qsj=sum(conj(qj).*qj,2)/L;% abs(qj).^2;
                            %                         sj(candidateset)=sj(candidateset)-w_norm(p)*(conj(A_test(:,candidateset)'*w_prev1).*((A_test(:,candidateset)'*w_prev1)));
                            sj=sj-w_norm(p)*(conj(w_prev2tmp).*w_prev2tmp);%abs(w_prev2).^2;
                        else
                            warning('Seq. SBL did not add new column')
                        end
                    end
%                     u_grid_updated=u;
                    [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,algo_iter_match_nomp,u,A_test,(0:m_test-1),L,m_test,yred*yred',sigma_n,NewtonSteps);                    
%                     xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,Lred).*qj(kpset(kpset>0),:);% qj needs to be updated; poor xhat estimation
%                     %                     profile off
%                     %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
                    seqsbl_timecomp(iter,m_iter)=toc;
                    err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
                    [err_vec, ind_vec] = min(abs(err_mat),[],1); % ind_vec pick estimated source
                    % Special error calculation
                    [~,ind_vec2]=min(err_vec); % ind_vec2 pick ground truth source
                    % err_vec(ind_vec2) is the best error for the best source; ind_vec(ind_vec2) is the best estimate index for best source
                    % 3-ind_vec2 is the second best source; 3-ind_vec(ind_vec2) ensures different estimate index for second best source
                    seqsbl_mse(iter,m_iter)=mean([err_vec(ind_vec2)^2 abs(err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
                    
%                     seqsbl_mse(iter,m_iter)=mean(err_vec.^2);
%                     seqsbl_l2error(iter,m_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
                    seqsbl_suppdist(iter,m_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% Atom SBL
%                     res = atomsbl_custom(y, 1000, []); %100 worked best for now
%                     param_estim = -2*res.theta';
%                     for i_param_estim=1:length(param_estim)
%                         if mod(ceil(param_estim(i_param_estim)),2)==0
%                             param_estim(i_param_estim)=param_estim(i_param_estim)-ceil(param_estim(i_param_estim));
%                         else
%                             param_estim(i_param_estim)=param_estim(i_param_estim)-floor(param_estim(i_param_estim));
%                         end
% %                         if param_estim(i_param_estim)<-1
% %                             param_estim(i_param_estim) = param_estim(i_param_estim)+2;
% %                         end
%                     end
%                     err_mat = repmat(param_estim', 1, suppsize)-repmat(u_actual(supp), length(param_estim), 1);
%                     [err_vec, ind_vec] = min(abs(err_mat),[],1); % ind_vec pick estimated source
%                     % Special error calculation
%                     [~,ind_vec2]=min(err_vec); % ind_vec2 pick ground truth source
%                     atomsbl_mse(iter,m_iter)=mean([err_vec(ind_vec2)^2 abs(err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
                    
%                     newton_seqsbl_mse(iter,m_iter)=mean(err_vec.^2);
                    %% Sequential SBL (Newton-Steps)
                    tic
                    % Initialization
                    algo_iter_match_nomp=length(estomega);
                    % Dimension Reduction
                    [Q1, ~] = qr(y',0);
                    yred=y*Q1; Lred = size(yred, 2);
                    
                    %                 candidateset=1:n;
                    qj=zeros(n,Lred);
                    qsj=zeros(n,1);
                    sj=zeros(n,1);
                    sigma_n=w_var; % noiseless case, can be adjusted
                    gamma_est=zeros(n,1);
                    kpset = zeros(algo_iter_match_nomp,1);
                    Cinv=eye(m_test)/sigma_n;
                    CiA=Cinv*A_test;
                    xhat = zeros(n,Lred);
                    w_norm=zeros(algo_iter_match_nomp,1);
                    w_prev1mat=zeros(m_test,algo_iter_match_nomp);
                    % Main Loop
%                     tic
                    %                     profile on
                    for p=1:algo_iter_match_nomp
                        if p==1
                            qj=A_test'*(yred/sigma_n);
                            qsj=sum(conj(qj).*qj,2)/L;% abs(qj).^2;
                            sj=(m_test/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
                        end
                        if p==1
                            [val,kp]=max(qsj);
                            val=sigma_n*val/m_test;
                        else
                            [val,kp]=max(qsj./sj);
                        end
                        if val>1
                            %                         kp=l_prev;
                            kpset(p)=kp;
                            %                         candidateset=candidateset(candidateset~=kp);%[candidateset(1:l_prev-1) candidateset(l_prev+1:end)];%
                            gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
                            
                            %                         w_norm=sqrt(1/gamma_est(kp)+sj(kp));
                            %                         %                     Cinv_prev=Cinv;
                            %                         %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
                            %                         %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
                            %                         w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*A_test(:,kp));
                            %                         w_prev1=w_prev1/w_norm;
                            %                         w_prev1mat(:,p)=w_prev1;
                            %                         w_prev2=A_test'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
                            %                         qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
                            %                         qsj=conj(qj).*qj;% abs(qj).^2;
                            %                         sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
                            
                            w_norm(p)=1/(1/gamma_est(kp)+sj(kp));
                            w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*((w_prev1mat(:,1:p-1)'*A_test(:,kp)).*w_norm(1:p-1,1));
                            w_prev1mat(:,p)=w_prev1;
                            w_prev2tmp=(A_test'*w_prev1);
                            w_prev2=w_prev2tmp*(w_norm(p)*qj(kp,:)); %(qj(kp)/w_norm)*w_prev1;
                            qj=qj-w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
                            qsj=sum(conj(qj).*qj,2)/L;% abs(qj).^2;
                            %                         sj(candidateset)=sj(candidateset)-w_norm(p)*(conj(A_test(:,candidateset)'*w_prev1).*((A_test(:,candidateset)'*w_prev1)));
                            sj=sj-w_norm(p)*(conj(w_prev2tmp).*w_prev2tmp);%abs(w_prev2).^2;
                        else
                            warning('Seq. SBL did not add new column')
                        end
                    end
                    kplist(iter,m_iter,1)=kpset(1); kplist(iter,m_iter,2)=kpset(2);
                    %                     u_grid_updated=u;
                    [gamma_est,u_grid_updated,Agrid_updated]=Newton_gridPtAdjPks(gamma_est,algo_iter_match_nomp,u,A_test,(0:m_test-1),L,m_test,yred*yred',sigma_n,NewtonSteps);
                    
%                     xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,Lred).*qj(kpset(kpset>0),:);
%                     %                     profile off
%                     %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
                    newton_seqsbl_timecomp(iter,m_iter)=toc;
                    err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
                    [err_vec, ind_vec] = min(abs(err_mat),[],1); % ind_vec pick estimated source
                    % Special error calculation
                    [~,ind_vec2]=min(err_vec); % ind_vec2 pick ground truth source
                    newton_seqsbl_mse(iter,m_iter)=mean([err_vec(ind_vec2)^2 abs(err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
                    
%                     newton_seqsbl_mse(iter,m_iter)=mean(err_vec.^2);
%                     seqsbl_l2error(iter,m_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
                    newton_seqsbl_suppdist(iter,m_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% Sequential SBL (Newton-Steps Refining Prematurely)-Has Errors
%                     % Initialization
%                     algo_iter_match_nomp=length(estomega);
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
%                     kpset = zeros(algo_iter_match_nomp,1);
%                     Cinv=eye(m_test)/sigma_n;
%                     CiA=Cinv*A_test;
%                     xhat = zeros(n,Lred);
%                     w_norm=zeros(algo_iter_match_nomp,1);
%                     w_prev1mat=zeros(m_test,algo_iter_match_nomp);
%                     u_grid_updated=u;
%                     Agrid_updated=A_test;
%                     
%                     % Main Loop
%                     tic
%                     %                     profile on
%                     for p=1:algo_iter_match_nomp
%                         if p==1
%                             qj=A_test'*(yred/sigma_n);
%                             qsj=sum(conj(qj).*qj,2)/L;% abs(qj).^2;
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
%                             [gamma_est,u_grid_updated,Agrid_updated]=Newton_single_gridPtAdjPks(kp,gamma_est,algo_iter_match_nomp,u_grid_updated,Agrid_updated,(0:m_test-1),L,m_test,yred*yred',sigma_n);
%                             [gamma_est,u_grid_updated,Agrid_updated]=Newton_gridPtAdjPks(gamma_est,algo_iter_match_nomp,u_grid_updated,Agrid_updated,(0:m_test-1),L,m_test,yred*yred',sigma_n);
%                             % Following may be made computationally
%                             % efficient
%                             Sigma_mi=(Agrid_updated.*repmat(gamma_est',m_test,1))*Agrid_updated'+sigma_n*eye(m_test); % Sigma minus grid point i
%                             iSigma_mi=eye(m_test)/Sigma_mi;
%                             qsj=real(sum(conj(iSigma_mi*Agrid_updated).*(((yred*yred'/L)*iSigma_mi)*Agrid_updated)));
%                             sj=real(sum(conj(Agrid_updated).*(iSigma_mi*Agrid_updated)));
% %                             w_norm(p)=1/(1/gamma_est(kp)+sj(kp));
% %                             w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*((w_prev1mat(:,1:p-1)'*A_test(:,kp)).*w_norm(1:p-1,1));
% %                             w_prev1mat(:,p)=w_prev1;
% %                             w_prev2tmp=(A_test'*w_prev1);
% %                             w_prev2=w_prev2tmp*(w_norm(p)*qj(kp,:)); %(qj(kp)/w_norm)*w_prev1;
% %                             qj=qj-w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
% %                             qsj=sum(conj(qj).*qj,2)/L;% abs(qj).^2;
% %                             %                         sj(candidateset)=sj(candidateset)-w_norm(p)*(conj(A_test(:,candidateset)'*w_prev1).*((A_test(:,candidateset)'*w_prev1)));
% %                             sj=sj-w_norm(p)*(conj(w_prev2tmp).*w_prev2tmp);%abs(w_prev2).^2;
%                         else
%                             warning('Seq. SBL did not add new column')
%                         end
%                     end
%                     %                     u_grid_updated=u;
% %                     [gamma_est,u_grid_updated,Agrid_updated]=Newton_gridPtAdjPks(gamma_est,algo_iter_match_nomp,u,A_test,(0:m_test-1),L,m_test,yred*yred',sigma_n);
%                     
% %                     xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,Lred).*qj(kpset(kpset>0),:);
% %                     %                     profile off
% %                     %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
%                     newton_prerefine_seqsbl_timecomp(iter,m_iter)=toc;
%                     err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
%                     [err_vec, ind_vec] = min(abs(err_mat),[],1); % ind_vec pick estimated source
%                     % Special error calculation
%                     [~,ind_vec2]=min(err_vec); % ind_vec2 pick ground truth source
%                     newton_prerefine_seqsbl_mse(iter,m_iter)=mean([err_vec(ind_vec2)^2 abs(err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
%                     
% %                     newton_prerefine_seqsbl_mse(iter,m_iter)=mean(err_vec.^2);
% %                     seqsbl_l2error(iter,m_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
%                     newton_prerefine_seqsbl_suppdist(iter,m_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                end
            end
        end
    end
end
% save('mpVssp.mat')
% unique_bound = 0.5*(1+1/mc);
save('mSizestudy_NOMPRs1Rc10_Newton10Steps_rankoneupdateGridless.mat')
load('mSizestudy_NOMPRs1Rc3_Newton3Steps_nobacktrack.mat')
figure
semilogy(m, sqrt(mean(seqsbl_mse)),'d', 'LineWidth', 2)
hold on
semilogy(m, sqrt(mean(nomp_mse)),'x', 'LineWidth', 2)
semilogy(m, sqrt(mean(newton_seqsbl_mse)), 'o', 'LineWidth', 2)
grid on
legend('Gridless LWS-SBL', 'NOMP', 'Newtonized Gridless LWS-SBL')
xlabel('Measurement size, m')
ylabel('RMSE in u-space')
for m_iter=1:length(m)
m_test=m(m_iter);
w_var=s_var/10^(SNR/10);
Sigma_s=diag(Amp_vec)*eye(suppsizemax)*diag(Amp_vec)';
num_factor=w_var./(2*L);
Acrb = exp(-1j*pi*(0:m_test-1)'*u_actual(supp));
Dcrb = -1j*(0:m_test-1)'.*Acrb;
Ry_inv = eye(m_test)/((Acrb*Sigma_s*Acrb')+w_var*eye(m_test));
wonumfactor_crb_psi = eye(suppsizemax)/(real(Dcrb'*(eye(m_test)-((Acrb/(Acrb'*Acrb))*Acrb'))*Dcrb).*((Sigma_s*Acrb'*Ry_inv*Acrb*Sigma_s).'));
%         % in theta degrees space
%         sqrcrb_theta(m_iter)=sqrt(mean(diag(diag(180./(pi^2*sqrt(1-sind(theta).^2)))*crb_psi*(diag(180./(pi^2*sqrt(1-sind(theta).^2))).'))));
% in u space
sqrcrb_u_theta(m_iter)=real(sqrt(num_factor)*sqrt(mean(diag((1/pi)*wonumfactor_crb_psi*((1/pi).')))));
end
semilogy(m,sqrcrb_u_theta, '--k', 'LineWidth', 1);
clear all
load('mSizestudy_NOMPRs1Rc3_Newton3Steps_backtrack.mat')
semilogy(m, sqrt(mean(newton_seqsbl_mse)), 'o', 'LineWidth', 2)
legend('Gridless LWS-SBL', 'NOMP', 'Newtonized Gridless LWS-SBL', 'CRB', 'with-Backtracking')


load('mSizestudy_NOMPRs1Rc3_Newton3Steps_nobacktrack.mat')
figure; kplist_sorted=sort(kplist,3);
subplot(313)
plot(m,diff(u_actual(supp))*ones(1,length(m)))
hold on
plot(m,1.782./m)
legend('sep. b/w sources in u-space', 'HPBW=1.782/m for Rectangular window')
xlabel('m')
ylabel('sep. in u-space')
grid on
for m_iter=1:12
subplot(311)
histogram(kplist_sorted(:,m_iter,1))
title(['m=' num2str(m(m_iter))])
axis([240 270 0 100])
subplot(312)
histogram(kplist_sorted(:,m_iter,2))
axis([240 270 0 100])
drawnow
pause(2)
end

% 
% % Stochastic CRB equal power sources assumed
% for m_iter=1:length(m)
%     m_test=m(m_iter);
%     w_var=s_var/10^(SNR/10);
%     Sigma_s=diag(Amp_vec)*eye(suppsizemax)*diag(Amp_vec)';
%     num_factor=w_var./(2*L);
%     Acrb = exp(-1j*pi*(0:m_test-1)'*u_actual(supp));
%     Dcrb = -1j*(0:m_test-1)'.*Acrb;
%     Ry_inv = eye(m_test)/((Acrb*Sigma_s*Acrb')+w_var*eye(m_test));
%     wonumfactor_crb_psi = eye(suppsizemax)/(real(Dcrb'*(eye(m_test)-((Acrb/(Acrb'*Acrb))*Acrb'))*Dcrb).*((Sigma_s*Acrb'*Ry_inv*Acrb*Sigma_s).'));
%     %         % in theta degrees space
%     %         sqrcrb_theta(m_iter)=sqrt(mean(diag(diag(180./(pi^2*sqrt(1-sind(theta).^2)))*crb_psi*(diag(180./(pi^2*sqrt(1-sind(theta).^2))).'))));
%     % in u space
%     sqrcrb_u_theta(m_iter)=real(sqrt(num_factor)*sqrt(mean(diag((1/pi)*wonumfactor_crb_psi*((1/pi).')))));
% end
% ax=loglog(m,sqrcrb_u_theta, '--k', 'LineWidth', 1);
% legend({'Gridless LWS-SBL: ITER=2', 'Gridless LWS-SBL: ITER=1', 'Gridless LWS-SBL: ITER=10','LWS-SBL: n=200','LWS-SBL: n=2000','CRB'}, 'NumColumns',2)
% ylabel('RMSE in u-space')
% xlabel('Number of Snapshots (L)')
% ax=ax.Parent;
% set(ax, 'FontWeight', 'bold','FontSize',16)
% xticks([1:5 10 20:20:60 100 250 500 1000])
% grid on

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

function [gamma_est,u_grid_updated,Agrid_updated]=gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,L,M,unRyoyo,lambda,NewtonSteps)
Sigma=Agrid_updated*diag(gamma_est)*Agrid_updated'+lambda*eye(M); % Sigma minus grid point i
iSigma=eye(M)/Sigma;
for iterGdPtAdPks=1:NewtonSteps % grid point adjustment around peaks iteration
    G=length(gamma_est); G_inner=10*G;
    m_candidates=find(gamma_est>0);
%     [pks,locs]=findpeaks1(gamma_est);
%     [mpks,mlocs]=maxk(pks, K);
    K_est=length(m_candidates);
%     m_candidates=sort(locs(mlocs));% one source
    u_est=u_grid_updated(m_candidates);
    for iterK=1:K_est
        m_iterK=m_candidates(iterK);
        if m_iterK>1; left_delta=u_grid_updated(m_iterK)-u_grid_updated(m_iterK-1); else; left_delta=u_grid_updated(m_iterK)+1; end
        if m_iterK<G; right_delta=u_grid_updated(m_iterK+1)-u_grid_updated(m_iterK); else; right_delta=1-u_grid_updated(m_iterK); end
        delta=left_delta/2+right_delta/2; resSeqSBL=delta/G_inner;
        u_candidates=linspace(u_est(iterK)-left_delta/2,u_est(iterK),floor(left_delta/2/resSeqSBL+1));
        u_candidates=union(u_candidates,linspace(u_est(iterK),u_est(iterK)+right_delta/2,floor(right_delta/2/resSeqSBL+1)));
        Gp=length(u_candidates);
        gamma_updated=zeros(1,Gp);
        I_gamma_opt=zeros(1,Gp);
        
        % Rank one update
        wi_vec_remove=iSigma*Agrid_updated(:,m_iterK);
        iSigma_mi=iSigma-(wi_vec_remove*wi_vec_remove')/(-1/gamma_est(m_iterK)+real(Agrid_updated(:,m_iterK)'*wi_vec_remove));
%         cmpgdind=setdiff(m_candidates,m_iterK); %setdiff(1:G,m_iterK); % complement set of grid index
%         Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(M); % Sigma minus grid point i
%         iSigma_mi=eye(M)/Sigma_mi;
        
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
            wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
            iSigma=iSigma_mi-(wi_vec_add*wi_vec_add')/(1/gamma_est(m_iterK)+real(Agrid_updated(:,m_iterK)'*wi_vec_add));
        end
    end
end
end

function [gamma_est,u_grid_updated,Agrid_updated]=Newton_single_gridPtAdjPks(kp,gamma_est,K,u_grid_updated,Agrid_updated,spos,L,M,unRyoyo,lambda)
% Sigma=Agrid_updated*diag(gamma_est)*Agrid_updated'+lambda*eye(M); % Sigma minus grid point i
% iSigma=eye(M)/Sigma;
% figure
for iterGdPtAdPks=1:1 % grid point adjustment around peaks iteration
    G=length(gamma_est); G_inner=10*G;
    %     m_candidates=find(gamma_est>0);
    %     [pks,locs]=findpeaks1(gamma_est);
    %     [mpks,mlocs]=maxk(pks, K);
    %     K_est=length(m_candidates);
    %     m_candidates=sort(locs(mlocs));% one source
    %     u_est=u_grid_updated(m_candidates);
    %     for iterK=1:K_est
    m_iterK=kp;
    u_old=u_grid_updated(m_iterK);
    % Rank one update
    %         wi_vec_remove=iSigma*Agrid_updated(:,m_iterK);
    %         iSigma_mi=iSigma-wi_vec_remove*wi_vec_remove'/(-1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_remove);
    cmpgdind=setdiff(1:G,m_iterK); %setdiff(1:G,m_iterK); % complement set of grid index
    Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(M); % Sigma minus grid point i
    iSigma_mi=eye(M)/Sigma_mi;
    %% Newton-Based Update
    der_q_u_sq=2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
    der_s_u=2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK));
    q_u_sq=real((Agrid_updated(:,m_iterK)'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
    s_u=real((Agrid_updated(:,m_iterK)'*iSigma_mi)*Agrid_updated(:,m_iterK));
    Rtildec=q_u_sq/s_u;
    der_Rtildec=(der_q_u_sq-Rtildec*der_s_u)/s_u; % 'R tilde c' as defined in ICASSP submission
    secder_q_u_sq=2*real(((((-1j*spos'*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)))...
        +2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*((-1j*spos'*pi).*Agrid_updated(:,m_iterK))));
    secder_s_u=2*real(((((-1j*spos'*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK))...
        +2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*((-1j*spos'*pi).*Agrid_updated(:,m_iterK)));
    secder_Rtildec=(secder_q_u_sq-Rtildec*secder_s_u)/s_u-2*der_s_u*der_Rtildec/s_u;
    if secder_Rtildec<0
        u_candidate=u_grid_updated(m_iterK)-der_Rtildec/secder_Rtildec;
        
        Aadpt_grid=exp(-1j*pi*spos'*[u_old u_candidate]);
        q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(((unRyoyo/L)*iSigma_mi)*Aadpt_grid)));
        s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
        q_sq_by_s_i=q_i_sq./s_i;
        gamma_updated=zeros(1,2);
        I_gamma_opt=zeros(1,2);
        gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
        I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
        %         [valneigh,indneigh]=min(I_gamma_opt);
        if I_gamma_opt(2)<I_gamma_opt(1)
            gamma_est(m_iterK)=gamma_updated(2);
            u_grid_updated(m_iterK)=u_candidate;
            Agrid_updated(:,m_iterK)=exp(-1j*pi*spos'*u_candidate);
            %             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
            %             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
        end
        %% Likelihood Evaluation-Based Update
        if m_iterK>1; left_delta=u_grid_updated(m_iterK)-u_grid_updated(m_iterK-1); else; left_delta=u_grid_updated(m_iterK)+1; end
        if m_iterK<G; right_delta=u_grid_updated(m_iterK+1)-u_grid_updated(m_iterK); else; right_delta=1-u_grid_updated(m_iterK); end
        delta=left_delta/2+right_delta/2; resSeqSBL=delta/G_inner;
        u_candidates=linspace(u_old-left_delta/2,u_old,floor(left_delta/2/resSeqSBL+1));
        u_candidates=union(u_candidates,linspace(u_old,u_old+right_delta/2,floor(right_delta/2/resSeqSBL+1)));
        Gp=length(u_candidates);
        gamma_updated=zeros(1,Gp);
        I_gamma_opt=zeros(1,Gp);
        
        Aadpt_grid=exp(-1j*pi*spos'*u_candidates);
        q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(((unRyoyo/L)*iSigma_mi)*Aadpt_grid)));
        s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
        q_sq_by_s_i=q_i_sq./s_i;
        gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
        I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
        %         [valneigh,indneigh]=min(I_gamma_opt);
        %         if valneigh<0
        %             gamma_est(m_iterK)=gamma_updated(indneigh);
        %             u_grid_updated(m_iterK)=u_candidates(indneigh);
        %             Agrid_updated(:,m_iterK)=exp(-1j*pi*spos'*u_candidates(indneigh));
        % %             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
        % %             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
        %         end
        %             clf
        %             plot(u_candidates, I_gamma_opt,'-b'); hold on
        %             plot(u_old*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--b')
        %             plot(u_candidate*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--r')
        %             plot(u_grid_updated(m_iterK)*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--k')
    end
end
end

function [gamma_est,u_grid_updated,Agrid_updated]=Newton_gridPtAdjPks(gamma_est,K,u_grid_updated,Agrid_updated,spos,L,M,unRyoyo,lambda,NewtonSteps)
Sigma=Agrid_updated*diag(gamma_est)*Agrid_updated'+lambda*eye(M); % Sigma minus grid point i
iSigma=eye(M)/Sigma;
% figure
for iterGdPtAdPks=1:NewtonSteps % grid point adjustment around peaks iteration
    G=length(gamma_est); G_inner=10*G;
    m_candidates=find(gamma_est>0);
%     [pks,locs]=findpeaks1(gamma_est);
%     [mpks,mlocs]=maxk(pks, K);
    K_est=length(m_candidates);
%     m_candidates=sort(locs(mlocs));% one source
    u_est=u_grid_updated(m_candidates);
    for iterK=1:K_est
        m_iterK=m_candidates(iterK);
        u_old=u_grid_updated(m_iterK);
        % Rank one update
        wi_vec_remove=iSigma*Agrid_updated(:,m_iterK);
        iSigma_mi=iSigma-(wi_vec_remove*wi_vec_remove')/(-1/gamma_est(m_iterK)+real(Agrid_updated(:,m_iterK)'*wi_vec_remove));
%         cmpgdind=setdiff(m_candidates,m_iterK); %setdiff(1:G,m_iterK); % complement set of grid index
%         Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(M); % Sigma minus grid point i
%         iSigma_mi=eye(M)/Sigma_mi;
        %% Newton-Based Update
        der_q_u_sq=2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
        der_s_u=2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK));
        q_u_sq=real((Agrid_updated(:,m_iterK)'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
        s_u=real((Agrid_updated(:,m_iterK)'*iSigma_mi)*Agrid_updated(:,m_iterK));
        Rtildec=q_u_sq/s_u;
        der_Rtildec=(der_q_u_sq-Rtildec*der_s_u)/s_u; % 'R tilde c' as defined in ICASSP submission
        secder_q_u_sq=2*real(((((-1j*spos'*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)))...
            +2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*((-1j*spos'*pi).*Agrid_updated(:,m_iterK))));
        secder_s_u=2*real(((((-1j*spos'*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK))...
            +2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*((-1j*spos'*pi).*Agrid_updated(:,m_iterK)));
        secder_Rtildec=(secder_q_u_sq-Rtildec*secder_s_u)/s_u-2*der_s_u*der_Rtildec/s_u;
        if secder_Rtildec<0
            alpha=1; contract_factor=0.5; step_fail=1; count=0;
            while step_fail
%                 step_fail=0;
                u_candidate=u_grid_updated(m_iterK)-alpha*der_Rtildec/secder_Rtildec;
                
                Aadpt_grid=exp(-1j*pi*spos'*[u_old u_candidate]);
                q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(((unRyoyo/L)*iSigma_mi)*Aadpt_grid)));
                s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
                q_sq_by_s_i=q_i_sq./s_i;
                gamma_updated=zeros(1,2);
                I_gamma_opt=zeros(1,2);
                gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
                I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
                %         [valneigh,indneigh]=min(I_gamma_opt);
                if I_gamma_opt(2)<I_gamma_opt(1)
                    step_fail=0;
                    gamma_est(m_iterK)=gamma_updated(2);
                    u_grid_updated(m_iterK)=u_candidate;
                    Agrid_updated(:,m_iterK)=exp(-1j*pi*spos'*u_candidate);
                    wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
                    iSigma=iSigma_mi-(wi_vec_add*wi_vec_add')/(1/gamma_est(m_iterK)+real(Agrid_updated(:,m_iterK)'*wi_vec_add));
                else
                    alpha=contract_factor*alpha;
                    count=count+1;
                    if count>10
%                         disp('Newton backtracked 10 times without success!')
                        step_fail=0;
                    end
                end
            end
%             %% Likelihood Evaluation-Based Update
%             if m_iterK>1; left_delta=u_grid_updated(m_iterK)-u_grid_updated(m_iterK-1); else; left_delta=u_grid_updated(m_iterK)+1; end
%             if m_iterK<G; right_delta=u_grid_updated(m_iterK+1)-u_grid_updated(m_iterK); else; right_delta=1-u_grid_updated(m_iterK); end
%             delta=left_delta/2+right_delta/2; resSeqSBL=delta/G_inner;
%             u_candidates=linspace(u_est(iterK)-left_delta/2,u_est(iterK),floor(left_delta/2/resSeqSBL+1));
%             u_candidates=union(u_candidates,linspace(u_est(iterK),u_est(iterK)+right_delta/2,floor(right_delta/2/resSeqSBL+1)));
%             Gp=length(u_candidates);
%             gamma_updated=zeros(1,Gp);
%             I_gamma_opt=zeros(1,Gp);
%             
%             Aadpt_grid=exp(-1j*pi*spos'*u_candidates);
%             q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(((unRyoyo/L)*iSigma_mi)*Aadpt_grid)));
%             s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
%             q_sq_by_s_i=q_i_sq./s_i;
%             gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
%             I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
%             %         [valneigh,indneigh]=min(I_gamma_opt);
%             %         if valneigh<0
%             %             gamma_est(m_iterK)=gamma_updated(indneigh);
%             %             u_grid_updated(m_iterK)=u_candidates(indneigh);
%             %             Agrid_updated(:,m_iterK)=exp(-1j*pi*spos'*u_candidates(indneigh));
%             % %             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
%             % %             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
%             %         end
% %             clf
% %             plot(u_candidates, I_gamma_opt,'-b'); hold on
% %             plot(u_old*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--b')
% %             plot(u_candidate*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--r')
% %             plot(u_grid_updated(m_iterK)*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--k')
        end
    end
end
end

function [PKS,LOCS]=findpeaks1(Y) % assumes Y is a non-negative vector
diff1=diff([0 reshape(Y,1,[]) 0]);
LOCS=find(diff1(1:end-1)>0 & diff1(2:end)<0);
PKS=Y(LOCS);
end