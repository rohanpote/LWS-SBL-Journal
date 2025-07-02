% Array Processing Problem: Seq. SBL vs Matching Pursuit type algorithms
% Impemented: Seq. SBL 1.LS-OMP 2.OMP 3.MP 4.Weak-MP 5.Thresholding

clear all
rng(1)
addpath('G:\My Drive\Research\2_Code\Sparse-Bayesian-Learning\Gridless_SBL\plots_asilomar_draft\Simulation_study_ICASSP\Extension_work\NOMP\wcslspectralestimation-continuous-frequency-estimation-779a5e217053\')
addpath('G:\My Drive\Research\2_Code\Sparse-Bayesian-Learning\Gridless_SBL\plots_asilomar_draft\Simulation_study_ICASSP\Extension_work\2DMNOMP-main\')

m1 = 5; m2=5;
n1 = 100; n2 = 100;
L=100; % single snapshot
suppsizemax = 2;
datagen = 0;
% epsilon = 1e-2;
% r = Inf; %suppsize;
ITER = 100;
SNR=-10:5:20;%[-40:5:20]; % in dB
s_var=1;
if datagen==0; Amp_vec=ones(suppsizemax,1); else; Amp_vec=sqrt(3)*sort(rand(suppsizemax,1),'descend'); end% amplitude vector
sep=1; % make sure (max. sep X 2 X suppsizemax)<=n to ensure sep is always possible

% Grid Initialize
[A, u, v, x_pos, y_pos] = create_2Ddictionary(m1,m2,n1,n2,'ULA');% number of grid points<n1*n2
n=size(A,2);


u_perturb=(4*rand([1,suppsizemax])-2)/n1;%/1e1;% off-grid perturb; [-1 1]/n1;%
v_perturb=(4*rand([1,suppsizemax])-2)/n2;%/1e1;% off-grid perturb; [-1 1]/n2;%
    
% Anorm = A*diag(1./vecnorm(A));
% AtA = Anorm'*Anorm; AtA_tmp = abs(AtA);
% AtA_tmp(1:n+1:end)=0;
% mc = max(max(AtA_tmp));

% sp_l2error = zeros(ITER, length(SNR)); sp_suppdist = zeros(ITER, length(SNR));
% sp_orig_l2error = zeros(ITER, length(SNR)); sp_orig_suppdist = zeros(ITER, length(SNR));
% lsomp_l2error = zeros(ITER, length(SNR)); lsomp_suppdist = zeros(ITER, length(SNR));
seqsbl_l2error = zeros(ITER, length(SNR)); seqsbl_suppdist = zeros(ITER, length(SNR)); seqsbl_timecomp = zeros(ITER, length(SNR)); seqsbl_mse=zeros(ITER, length(SNR));
newton_seqsbl_suppdist = zeros(ITER, length(SNR)); newton_seqsbl_timecomp = zeros(ITER, length(SNR)); newton_seqsbl_mse=zeros(ITER, length(SNR));
newton_prerefine_seqsbl_suppdist = zeros(ITER, length(SNR)); newton_prerefine_seqsbl_timecomp = zeros(ITER, length(SNR)); newton_prerefine_seqsbl_mse=zeros(ITER, length(SNR));
seqsbl_nvarNotgivenwvarby10_l2error = zeros(ITER, length(SNR)); seqsbl_nvarNotgivenwvarby10_suppdist = zeros(ITER, length(SNR)); seqsbl_nvarNotgivenwvarby10_timecomp = zeros(ITER, length(SNR)); seqsbl_nvarNotgivenwvarby10_mse=zeros(ITER, length(SNR));
seqsbl_nvarNotgivenwvar10_l2error = zeros(ITER, length(SNR)); seqsbl_nvarNotgivenwvar10_suppdist = zeros(ITER, length(SNR)); seqsbl_nvarNotgivenwvar10_timecomp = zeros(ITER, length(SNR)); seqsbl_nvarNotgivenwvar10_mse=zeros(ITER, length(SNR));
TwoDMNOMP_mse=zeros(ITER, length(SNR));
seqsbl_targetsj_l2error = zeros(ITER, length(SNR)); seqsbl_targetsj_suppdist = zeros(ITER, length(SNR)); seqsbl_targetsj_timecomp = zeros(ITER, length(SNR));
seqsbl_ximprov_l2error = zeros(ITER, length(SNR)); seqsbl_ximprov_suppdist = zeros(ITER, length(SNR)); seqsbl_ximprov_timecomp = zeros(ITER, length(SNR));
redcompomp_l2error = zeros(ITER, length(SNR)); redcompomp_suppdist = zeros(ITER, length(SNR)); redcompomp_timecomp = zeros(ITER, length(SNR));
omp_l2error = zeros(ITER, length(SNR)); omp_suppdist = zeros(ITER, length(SNR)); omp_timecomp = zeros(ITER, length(SNR));
% mp_l2error = zeros(ITER, length(SNR)); mp_suppdist = zeros(ITER, length(SNR));
% weakmp_l2error = zeros(ITER, length(SNR)); weakmp_suppdist = zeros(ITER, length(SNR));
% thresh_l2error = zeros(ITER, length(SNR)); thresh_suppdist = zeros(ITER, length(SNR));
% L1_l2error = zeros(ITER, length(SNR)); L1_suppdist = zeros(ITER, length(SNR));
% RL1_l2error = zeros(ITER, length(SNR)); RL1_suppdist = zeros(ITER, length(SNR));
% reg_IRLS_l2error = zeros(ITER, length(SNR)); reg_IRLS_suppdist = zeros(ITER, length(SNR));
% SBL_l2error = zeros(ITER, length(SNR)); SBL_suppdist = zeros(ITER, length(SNR));

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
    noise_vec=randn(m1*m2,L(end))+1j*randn(m1*m2,L(end));
    permute_n=[n2*n1/4 n2*n1/4+20*n2+30];%randsample(n, n); % without replacement
    for snr_iter=1:length(SNR)
        snr_test=SNR(snr_iter);
        w_var=s_var/10^(snr_test/10);
        for m_iter=1:length(m1)
            m1_test=m1(m_iter);
            A_test=A(1:m1_test*m2,:);
            for sep_iter=1:length(sep)
                [suppfull,loop_time] = min_separate(permute_n, suppsizemax, sep(sep_iter));%randsample(n, suppsizemax);
                %         loop_time
                xfull = zeros(n,L);
                xfull(suppfull,:)=nonzero_x;
                for isuppsize=suppsizemax:suppsizemax
                    if iter==30 && m_iter==8
                        disp('hi')
                    end
                    suppsize = isuppsize;
                    supp = suppfull(1:suppsize);
                    x = xfull;
                    u_actual=u; u_actual(supp)=u_actual(supp)+u_perturb;
                    v_actual=v; v_actual(supp)=v_actual(supp)+v_perturb;
                    A_actual=exp(-1j*pi*(x_pos(:)*u_actual+y_pos(:)*v_actual));
                    y = sqrt(s_var/2)*A_actual*x+sqrt(w_var/2)*noise_vec;
                    y=y(1:m1_test*m2,:);
                    
                    %% NOMP (Need to adapt to handle 2D geometry)
%                     p_fa = 1e-2;
%                     tau = L*w_var * ( log(m1_test*m2) - log( log(1/(1-p_fa)) ) );
%                     [omegaList, gainList, residueList] = extractSpectrum_MMV(y, eye(m1_test*m2), tau, n/m1_test/m2,1,3,suppsizemax);
%                     estomega=omegaList/pi;
%                     estomega(estomega>=1)=estomega(estomega>=1)-2;
%                     estomega=-estomega;
%                     if isempty(estomega)
%                         estomega=0;
%                     end
%                     err_mat = repmat(estomega, 1, suppsize)-repmat(u_actual(supp), length(estomega), 1);
%                     [err_vec, ind_vec] = min(abs(err_mat),[],1); % ind_vec pick estimated source
%                     % Special error calculation
%                     [~,ind_vec2]=min(err_vec); % ind_vec2 pick ground truth source
%                     nomp_mse(iter,snr_iter)=mean([err_vec(ind_vec2)^2 abs(err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
%                     
% %                     nomp_mse(iter,snr_iter)=mean(err_vec.^2);
%                     nomp_iterrun(iter,snr_iter)=length(estomega);
                    
                    %% 2D-MNOMP paper. Reference 'Two-dimensional multi-snapshot Newtonized orthogonal matching pursuit for DOA estimation' by Jiang Zhu et al.
                    y_matrix=reshape(permute(y,[1 3 2]),m2,m1_test,[]);
                    tau=Inf;
                    [omegaList_tau, gainList_tau, y_residue_matrix]=MNOMP2D(y_matrix, tau, [n2/m2 n1/m1],1,3,suppsizemax);
                    estomega=omegaList_tau/pi;
                    estomega(estomega>=1)=estomega(estomega>=1)-2;
                    estomega=-estomega;
                    if isempty(estomega)
                        estomega=0;
                    end
                    u_est_2DMNOMP=estomega(:,2); v_est_2DMNOMP=estomega(:,1);
                    u_err_mat = repmat(u_est_2DMNOMP, 1, suppsize)-repmat(u_actual(supp), length(u_est_2DMNOMP), 1);
                    v_err_mat = repmat(v_est_2DMNOMP, 1, suppsize)-repmat(v_actual(supp), length(v_est_2DMNOMP), 1);
                    [err_vec, ind_vec] = min(abs(u_err_mat).^2+abs(v_err_mat).^2,[],1); % each source picks one estimate
                    % Special error calculation
                    [~,ind_vec2]=min(err_vec); % pick the best source
                    % err_vec(ind_vec2) is the best error for best source; ind_vec(ind_vec2) is the best estimate index for best source
                    % 3-ind_vec2 is the second best source; 3-ind_vec(ind_vec2) ensures different estimate index for second best source
                    TwoDMNOMP_mse(iter,snr_iter)=mean([err_vec(ind_vec2) abs(u_err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2+abs(v_err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
                    %% Sequential SBL (Computationally May Be Improved Further)
                    % Initialization
                    algo_iter_match_nomp=2;%length(estomega);
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
                    Cinv=eye(m1_test*m2)/sigma_n;
                    CiA=Cinv*A_test;
                    xhat = zeros(n,Lred);
                    w_norm=zeros(algo_iter_match_nomp,1);
                    w_prev1mat=zeros(m1_test*m2,algo_iter_match_nomp);
                    % Main Loop
                    tic
                    %                     profile on
                    for p=1:algo_iter_match_nomp
                        if p==1
                            qj=A_test'*(yred/sigma_n);
                            qsj=sum(conj(qj).*qj,2)/L;% abs(qj).^2;
                            sj=(m1_test*m2/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
                        end
                        if p==1
                            [val,kp]=max(qsj);
                            val=sigma_n*val/m1_test/m2;
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
%                     %Plotting
%                     figure(1);clf
%                     plot(u_actual(supp(1)), v_actual(supp(1)), 'xr', 'LineWidth', 2, 'MarkerSize',10)
%                     hold on
%                     plot(u_actual(supp(2)), v_actual(supp(2)), 'xr', 'LineWidth', 2, 'MarkerSize',10)
%                     axis([-1 1 -1 1])
%                     grid on
%                     plot(u(kpset(1)), v(kpset(1)), 'og', 'LineWidth', 2, 'MarkerSize',10)
%                     plot(u(kpset(2)), v(kpset(2)), 'og', 'LineWidth', 2, 'MarkerSize',10)
%                     title('Gridless LWS-SBL')
%                     u_grid_updated=u;
%                     v_grid_updated=v;
                    [gamma_est,u_grid_updated,v_grid_updated,Agrid_updated]=TwoDim_gridPtAdjPks(gamma_est,algo_iter_match_nomp,u,v,A_test,x_pos,y_pos,L,m1_test,m2,yred*yred',sigma_n);                    
%                     xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,Lred).*qj(kpset(kpset>0),:);% qj needs to be updated; poor xhat estimation
%                     %                     profile off
%                     %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
                    seqsbl_timecomp(iter,snr_iter)=toc;
                    u_err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
                    v_err_mat = repmat(v_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(v_actual(supp), length(v_grid_updated(kpset(kpset>0))), 1);
                    [err_vec, ind_vec] = min(abs(u_err_mat).^2+abs(v_err_mat).^2,[],1); % each source picks one estimate
                    % Special error calculation
                    [~,ind_vec2]=min(err_vec); % pick the best source
                    % err_vec(ind_vec2) is the best error for best source; ind_vec(ind_vec2) is the best estimate index for best source
                    % 3-ind_vec2 is the second best source; 3-ind_vec(ind_vec2) ensures different estimate index for second best source
                    seqsbl_mse(iter,snr_iter)=mean([err_vec(ind_vec2) abs(u_err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2+abs(v_err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
                                        
%                     seqsbl_mse(iter,snr_iter)=mean(err_vec.^2);
%                     seqsbl_l2error(iter,snr_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
                    seqsbl_suppdist(iter,snr_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                    %% Sequential SBL (Newton-Steps)
                    % Initialization
                    algo_iter_match_nomp=2;%length(estomega);
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
                    Cinv=eye(m1_test*m2)/sigma_n;
                    CiA=Cinv*A_test;
                    xhat = zeros(n,Lred);
                    w_norm=zeros(algo_iter_match_nomp,1);
                    w_prev1mat=zeros(m1_test*m2,algo_iter_match_nomp);
                    % Main Loop
                    tic
                    %                     profile on
                    for p=1:algo_iter_match_nomp
                        if p==1
                            qj=A_test'*(yred/sigma_n);
                            qsj=sum(conj(qj).*qj,2)/L;% abs(qj).^2;
                            sj=(m1_test*m2/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
                        end
                        if p==1
                            [val,kp]=max(qsj);
                            val=sigma_n*val/m1_test/m2;
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
%                     %Plotting
%                     figure(2);clf
%                     plot(u_actual(supp(1)), v_actual(supp(1)), 'xr', 'LineWidth', 2, 'MarkerSize',10)
%                     hold on
%                     plot(u_actual(supp(2)), v_actual(supp(2)), 'xr', 'LineWidth', 2, 'MarkerSize',10)
%                     axis([-1 1 -1 1])
%                     grid on
%                     plot(u(kpset(1)), v(kpset(1)), 'og', 'LineWidth', 2, 'MarkerSize',10)
%                     plot(u(kpset(2)), v(kpset(2)), 'og', 'LineWidth', 2, 'MarkerSize',10)
%                     title('Newtonized Gridless LWS-SBL')
%                     u_grid_updated=u;
%                     v_grid_updated=v;
                    %                     u_grid_updated=u;
                    [gamma_est,u_grid_updated,v_grid_updated,Agrid_updated]=TwoDim_Newton_gridPtAdjPks(gamma_est,algo_iter_match_nomp,u,v,A_test,x_pos,y_pos,L,m1_test,m2,yred*yred',sigma_n);
                    
%                     xhat(kpset(kpset>0),:)=repmat(gamma_est(kpset(kpset>0)),1,Lred).*qj(kpset(kpset>0),:);
%                     %                     profile off
%                     %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
                    newton_seqsbl_timecomp(iter,snr_iter)=toc;
                    u_err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
                    v_err_mat = repmat(v_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(v_actual(supp), length(v_grid_updated(kpset(kpset>0))), 1);
                    [err_vec, ind_vec] = min(abs(u_err_mat).^2+abs(v_err_mat).^2,[],1); % each source picks one estimate
                    % Special error calculation
                    [~,ind_vec2]=min(err_vec); % pick the best source
                    % err_vec(ind_vec2) is the best error for best source; ind_vec(ind_vec2) is the best estimate index for best source
                    % 3-ind_vec2 is the second best source; 3-ind_vec(ind_vec2) ensures different estimate index for second best source
                    newton_seqsbl_mse(iter,snr_iter)=mean([err_vec(ind_vec2) abs(u_err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2+abs(v_err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
                    
%                     newton_seqsbl_mse(iter,snr_iter)=mean(err_vec.^2);
%                     seqsbl_l2error(iter,snr_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
                    newton_seqsbl_suppdist(iter,snr_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
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
%                     newton_prerefine_seqsbl_timecomp(iter,snr_iter)=toc;
%                     err_mat = repmat(u_grid_updated(kpset(kpset>0))', 1, suppsize)-repmat(u_actual(supp), length(u_grid_updated(kpset(kpset>0))), 1);
%                     [err_vec, ind_vec] = min(abs(err_mat),[],1); % ind_vec pick estimated source
%                     % Special error calculation
%                     [~,ind_vec2]=min(err_vec); % ind_vec2 pick ground truth source
%                     newton_prerefine_seqsbl_mse(iter,snr_iter)=mean([err_vec(ind_vec2)^2 abs(err_mat(3-ind_vec(ind_vec2),3-ind_vec2))^2]);
%                     
% %                     newton_prerefine_seqsbl_mse(iter,snr_iter)=mean(err_vec.^2);
% %                     seqsbl_l2error(iter,snr_iter) = norm(x-xhat*Q1','fro')^2/norm(x,'fro')^2;
%                     newton_prerefine_seqsbl_suppdist(iter,snr_iter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                    
                end
            end
        end
    end
end
% save('mpVssp.mat')
% unique_bound = 0.5*(1+1/mc);
%%
figure
load('SNRstudy_Gridbased_TwoDim_LWS_SBL.mat')
semilogy(SNR, sqrt(mean(seqsbl_mse)), '-.b', 'LineWidth', 2)
hold on
load('SNRstudy_TwoDim_LWSSBL_2DMNOMP.mat')
semilogy(SNR, sqrt(mean(newton_seqsbl_mse)), '-sb', 'LineWidth', 2,'MarkerSize',25)
semilogy(SNR, sqrt(mean(TwoDMNOMP_mse)), '-or', 'LineWidth', 2,'MarkerSize',15)
semilogy(SNR, sqrt(mean(seqsbl_mse)),'--pb', 'LineWidth', 1, 'MarkerSize',10);
grid on
xlabel('SNR in dB')
ylabel('RMSE in u-v space')
% Stochastic CRB equal power sources assumed
for snr_iter=1:length(SNR)
    snr_test=SNR(snr_iter);
    w_var=s_var/10^(snr_test/10);
    Sigma_s=diag(Amp_vec)*eye(suppsizemax)*diag(Amp_vec)';
    num_factor=w_var./(2*L);
%     Acrb = exp(-1j*pi*(0:m1_test-1)'*u_actual(supp));
    Acrb = exp(-1j*pi*((x_pos(:)*u_actual(supp)+y_pos(:)*v_actual(supp))));
    Dcrb = [-1j*x_pos(:).*Acrb -1j*y_pos(:).*Acrb];
    Ry_inv = eye(m1*m2)/((Acrb*Sigma_s*Acrb')+w_var*eye(m1*m2));
    wonumfactor_crb_psi = eye(2*suppsizemax)/(real(Dcrb'*(eye(m1*m2)-((Acrb/(Acrb'*Acrb))*Acrb'))*Dcrb).*(([Sigma_s;Sigma_s]*Acrb'*Ry_inv*Acrb*[Sigma_s Sigma_s]).'));
    %         % in theta degrees space
    %         sqrcrb_theta(m_iter)=sqrt(mean(diag(diag(180./(pi^2*sqrt(1-sind(theta).^2)))*crb_psi*(diag(180./(pi^2*sqrt(1-sind(theta).^2))).'))));
    % in u space
    sqrcrb_u_theta(snr_iter)=real(sqrt(num_factor)*sqrt(2*mean(diag((1/pi)*wonumfactor_crb_psi*((1/pi).'))))); % Note a '2' is multiplied in variance as u^2+v^2 is the squared error in simulation
end
ax=semilogy(SNR,sqrcrb_u_theta, '--k', 'LineWidth', 1);
legend('LWS-SBL (on-grid)','2D Newton(N) LWS-SBL','2D MNOMP','Grid refine LWS-SBL','CRB', 'Location', 'SouthWest')
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',14)
xticks(SNR)
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

function [gamma_est,u_grid_updated,v_grid_updated,Agrid_updated]=TwoDim_gridPtAdjPks(gamma_est,K,u_grid_updated,v_grid_updated,Agrid_updated,x_spos,y_spos,L,m1,m2,unRyoyo,lambda)
% Sigma=Agrid_updated*diag(gamma_est)*Agrid_updated'+lambda*eye(M); % Sigma minus grid point i
% iSigma=eye(M)/Sigma;
for iterGdPtAdPks=1:2 % grid point adjustment around peaks iteration
    G=length(gamma_est); G_inner=ceil(sqrt(G));
    [pks,locs]=findpeaks1(gamma_est);
    [mpks,mlocs]=maxk(pks, K);
    K_est=length(mlocs);
    m_candidates=sort(locs(mlocs));% one source
    u_est=u_grid_updated(m_candidates);
    v_est=v_grid_updated(m_candidates);
    for iterK=1:K_est
        m_iterK=m_candidates(iterK);
        % Keeping it simple
%         if m_iterK>1; left_delta=u_grid_updated(m_iterK)-u_grid_updated(m_iterK-1); else; left_delta=u_grid_updated(m_iterK)+1; end
%         if m_iterK<G; right_delta=u_grid_updated(m_iterK+1)-u_grid_updated(m_iterK); else; right_delta=1-u_grid_updated(m_iterK); end
        left_u_delta=u_grid_updated(m_iterK)-u_grid_updated(sum((u_grid_updated(m_iterK)-u_grid_updated(1:m_iterK-1))>1e-4)); 
        right_u_delta=u_grid_updated(find((u_grid_updated-u_grid_updated(m_iterK))>1e-4,1))-u_grid_updated(m_iterK);
        left_v_delta=v_grid_updated(m_iterK)-v_grid_updated(m_iterK-1); right_v_delta=v_grid_updated(m_iterK+1)-v_grid_updated(m_iterK);
        u_delta=left_u_delta/2+right_u_delta/2; u_resSeqSBL=u_delta/G_inner;
        v_delta=left_v_delta/2+right_v_delta/2; v_resSeqSBL=v_delta/G_inner;
        u_candidates=linspace(u_est(iterK)-left_u_delta/2,u_est(iterK),floor(left_u_delta/2/u_resSeqSBL+1));
        u_candidates=union(u_candidates,linspace(u_est(iterK),u_est(iterK)+right_u_delta/2,floor(right_u_delta/2/u_resSeqSBL+1)));
        v_candidates=linspace(v_est(iterK)-left_v_delta/2,v_est(iterK),floor(left_v_delta/2/v_resSeqSBL+1));
        v_candidates=union(v_candidates,linspace(v_est(iterK),v_est(iterK)+right_v_delta/2,floor(right_v_delta/2/v_resSeqSBL+1)));
        u_Gp=length(u_candidates); v_Gp=length(v_candidates); % skipping u^2+v^2<=1 condition as interior points assumed
        gamma_updated=zeros(1,u_Gp*v_Gp);
        I_gamma_opt=zeros(1,u_Gp*v_Gp);
        
        % Rank one update
        %         wi_vec_remove=iSigma*Agrid_updated(:,m_iterK);
        %         iSigma_mi=iSigma-wi_vec_remove*wi_vec_remove'/(-1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_remove);
        cmpgdind=setdiff(m_candidates,m_iterK); %setdiff(1:G,m_iterK); % complement set of grid index
        Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(m1*m2); % Sigma minus grid point i
        iSigma_mi=eye(m1*m2)/Sigma_mi;
        
        [umat,vmat] = meshgrid(u_candidates,v_candidates);
        ugrid_t = umat(:)';
        vgrid_t = vmat(:)';
        Aadpt_grid=exp(-1j*pi*(x_spos(:)*ugrid_t+y_spos(:)*vgrid_t));
        q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(((unRyoyo/L)*iSigma_mi)*Aadpt_grid)));
        s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
        q_sq_by_s_i=q_i_sq./s_i;
        gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
        I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
        [valneigh,indneigh]=min(I_gamma_opt);
        if valneigh<0
            gamma_est(m_iterK)=gamma_updated(indneigh);
            u_grid_updated(m_iterK)=ugrid_t(indneigh);
            v_grid_updated(m_iterK)=vgrid_t(indneigh);
            Agrid_updated(:,m_iterK)=exp(-1j*pi*(x_spos(:)*ugrid_t(indneigh)+y_spos(:)*vgrid_t(indneigh)));
            %             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
            %             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
        end
    end
%     %Plotting
%     figure(1)
%     plot(u_grid_updated(m_candidates(1)), v_grid_updated(m_candidates(1)), 'o','Color',1-[iterGdPtAdPks iterGdPtAdPks 0]/3, 'LineWidth', 2, 'MarkerSize',10)
%     plot(u_grid_updated(m_candidates(2)), v_grid_updated(m_candidates(2)), 'o','Color',1-[iterGdPtAdPks iterGdPtAdPks 0]/3, 'LineWidth', 2, 'MarkerSize',10)
end
end

% function [gamma_est,u_grid_updated,Agrid_updated]=Newton_single_gridPtAdjPks(kp,gamma_est,K,u_grid_updated,Agrid_updated,spos,L,M,unRyoyo,lambda)
% % Sigma=Agrid_updated*diag(gamma_est)*Agrid_updated'+lambda*eye(M); % Sigma minus grid point i
% % iSigma=eye(M)/Sigma;
% % figure
% for iterGdPtAdPks=1:1 % grid point adjustment around peaks iteration
%     G=length(gamma_est); G_inner=10*G;
%     %     [pks,locs]=findpeaks1(gamma_est);
%     %     [mpks,mlocs]=maxk(pks, K);
%     %     K_est=length(mlocs);
%     %     m_candidates=sort(locs(mlocs));% one source
%     %     u_est=u_grid_updated(m_candidates);
%     %     for iterK=1:K_est
%     m_iterK=kp;
%     u_old=u_grid_updated(m_iterK);
%     % Rank one update
%     %         wi_vec_remove=iSigma*Agrid_updated(:,m_iterK);
%     %         iSigma_mi=iSigma-wi_vec_remove*wi_vec_remove'/(-1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_remove);
%     cmpgdind=setdiff(1:G,m_iterK); %setdiff(1:G,m_iterK); % complement set of grid index
%     Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(M); % Sigma minus grid point i
%     iSigma_mi=eye(M)/Sigma_mi;
%     %% Newton-Based Update
%     der_q_u_sq=2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
%     der_s_u=2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK));
%     q_u_sq=real((Agrid_updated(:,m_iterK)'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
%     s_u=real((Agrid_updated(:,m_iterK)'*iSigma_mi)*Agrid_updated(:,m_iterK));
%     Rtildec=q_u_sq/s_u;
%     der_Rtildec=(der_q_u_sq-Rtildec*der_s_u)/s_u; % 'R tilde c' as defined in ICASSP submission
%     secder_q_u_sq=2*real(((((-1j*spos'*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)))...
%         +2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*((-1j*spos'*pi).*Agrid_updated(:,m_iterK))));
%     secder_s_u=2*real(((((-1j*spos'*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK))...
%         +2*real((((-1j*spos'*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*((-1j*spos'*pi).*Agrid_updated(:,m_iterK)));
%     secder_Rtildec=(secder_q_u_sq-Rtildec*secder_s_u)/s_u-2*der_s_u*der_Rtildec/s_u;
%     if secder_Rtildec<0
%         u_candidate=u_grid_updated(m_iterK)-der_Rtildec/secder_Rtildec;
%         
%         Aadpt_grid=exp(-1j*pi*spos'*[u_old u_candidate]);
%         q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(((unRyoyo/L)*iSigma_mi)*Aadpt_grid)));
%         s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
%         q_sq_by_s_i=q_i_sq./s_i;
%         gamma_updated=zeros(1,2);
%         I_gamma_opt=zeros(1,2);
%         gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
%         I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
%         %         [valneigh,indneigh]=min(I_gamma_opt);
%         if I_gamma_opt(2)<I_gamma_opt(1)
%             gamma_est(m_iterK)=gamma_updated(2);
%             u_grid_updated(m_iterK)=u_candidate;
%             Agrid_updated(:,m_iterK)=exp(-1j*pi*spos'*u_candidate);
%             %             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
%             %             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
%         end
%         %% Likelihood Evaluation-Based Update
%         if m_iterK>1; left_delta=u_grid_updated(m_iterK)-u_grid_updated(m_iterK-1); else; left_delta=u_grid_updated(m_iterK)+1; end
%         if m_iterK<G; right_delta=u_grid_updated(m_iterK+1)-u_grid_updated(m_iterK); else; right_delta=1-u_grid_updated(m_iterK); end
%         delta=left_delta/2+right_delta/2; resSeqSBL=delta/G_inner;
%         u_candidates=linspace(u_old-left_delta/2,u_old,floor(left_delta/2/resSeqSBL+1));
%         u_candidates=union(u_candidates,linspace(u_old,u_old+right_delta/2,floor(right_delta/2/resSeqSBL+1)));
%         Gp=length(u_candidates);
%         gamma_updated=zeros(1,Gp);
%         I_gamma_opt=zeros(1,Gp);
%         
%         Aadpt_grid=exp(-1j*pi*spos'*u_candidates);
%         q_i_sq=real(sum(conj(iSigma_mi*Aadpt_grid).*(((unRyoyo/L)*iSigma_mi)*Aadpt_grid)));
%         s_i=real(sum(conj(Aadpt_grid).*(iSigma_mi*Aadpt_grid)));
%         q_sq_by_s_i=q_i_sq./s_i;
%         gamma_updated(q_i_sq>s_i)=(q_sq_by_s_i(q_i_sq>s_i)-1)./s_i(q_i_sq>s_i);
%         I_gamma_opt(q_i_sq>s_i)=log(q_sq_by_s_i(q_i_sq>s_i))-q_sq_by_s_i(q_i_sq>s_i)+1;
%         %         [valneigh,indneigh]=min(I_gamma_opt);
%         %         if valneigh<0
%         %             gamma_est(m_iterK)=gamma_updated(indneigh);
%         %             u_grid_updated(m_iterK)=u_candidates(indneigh);
%         %             Agrid_updated(:,m_iterK)=exp(-1j*pi*spos'*u_candidates(indneigh));
%         % %             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
%         % %             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
%         %         end
%         %             clf
%         %             plot(u_candidates, I_gamma_opt,'-b'); hold on
%         %             plot(u_old*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--b')
%         %             plot(u_candidate*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--r')
%         %             plot(u_grid_updated(m_iterK)*ones(1,10),linspace(min(I_gamma_opt),max(I_gamma_opt),10),'--k')
%     end
% end
% end

function [gamma_est,u_grid_updated,v_grid_updated,Agrid_updated]=TwoDim_Newton_gridPtAdjPks(gamma_est,K,u_grid_updated,v_grid_updated,Agrid_updated,x_spos,y_spos,L,m1,m2,unRyoyo,lambda)
% Sigma=Agrid_updated*diag(gamma_est)*Agrid_updated'+lambda*eye(M); % Sigma minus grid point i
% iSigma=eye(M)/Sigma;
% figure
ITERGdPtAdPks=3;
for iterGdPtAdPks=1:ITERGdPtAdPks % grid point adjustment around peaks iteration
%     G=length(gamma_est); G_inner=ceil(sqrt(G));
    [pks,locs]=findpeaks1(gamma_est);
    [mpks,mlocs]=maxk(pks, K);
    K_est=length(mlocs);
    m_candidates=sort(locs(mlocs));% one source
    u_est=u_grid_updated(m_candidates);
    v_est=v_grid_updated(m_candidates);
    for iterK=1:K_est
        m_iterK=m_candidates(iterK);
        u_old=u_grid_updated(m_iterK);
        v_old=v_grid_updated(m_iterK);
        % Rank one update
        %         wi_vec_remove=iSigma*Agrid_updated(:,m_iterK);
        %         iSigma_mi=iSigma-wi_vec_remove*wi_vec_remove'/(-1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_remove);
        cmpgdind=setdiff(m_candidates,m_iterK); %setdiff(1:G,m_iterK); % complement set of grid index
        Sigma_mi=Agrid_updated(:,cmpgdind)*diag(gamma_est(cmpgdind))*Agrid_updated(:,cmpgdind)'+lambda*eye(m1*m2); % Sigma minus grid point i
        iSigma_mi=eye(m1*m2)/Sigma_mi;
        %% Newton-Based Update
        u_der_q_uv_sq=2*real((((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
        v_der_q_uv_sq=2*real((((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
        u_der_s_uv=2*real((((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK));
        v_der_s_uv=2*real((((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK));
        q_uv_sq=real((Agrid_updated(:,m_iterK)'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)));
        s_uv=real((Agrid_updated(:,m_iterK)'*iSigma_mi)*Agrid_updated(:,m_iterK));
        Rtildec=q_uv_sq/s_uv;
        u_der_Rtildec=(u_der_q_uv_sq-Rtildec*u_der_s_uv)/s_uv; % 'R tilde c' as defined in ICASSP submission
        v_der_Rtildec=(v_der_q_uv_sq-Rtildec*v_der_s_uv)/s_uv; % 'R tilde c' as defined in ICASSP submission
        grad_Rtildec=[u_der_Rtildec; v_der_Rtildec];
        u_secder_q_uv_sq=2*real(((((-1j*x_spos(:)*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)))...
            +2*real((((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK))));
        v_secder_q_uv_sq=2*real(((((-1j*y_spos(:)*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)))...
            +2*real((((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK))));
        u_secder_s_uv=2*real(((((-1j*x_spos(:)*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK))...
            +2*real((((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK)));
        v_secder_s_uv=2*real(((((-1j*y_spos(:)*pi).^2).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK))...
            +2*real((((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK)));
        uv_secder_q_uv_sq=2*real(((((-1j*y_spos(:)*pi).*(-1j*x_spos(:)*pi)).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)))...
            +2*real((((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK))));
        vu_secder_q_uv_sq=2*real(((((-1j*x_spos(:)*pi).*(-1j*y_spos(:)*pi)).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*Agrid_updated(:,m_iterK)))...
            +2*real((((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*(((unRyoyo/L)*iSigma_mi)*((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK))));
        uv_secder_s_uv=2*real(((((-1j*y_spos(:)*pi).*(-1j*x_spos(:)*pi)).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK))...
            +2*real((((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK)));
        vu_secder_s_uv=2*real(((((-1j*x_spos(:)*pi).*(-1j*y_spos(:)*pi)).*Agrid_updated(:,m_iterK))'*iSigma_mi)*Agrid_updated(:,m_iterK))...
            +2*real((((-1j*x_spos(:)*pi).*Agrid_updated(:,m_iterK))'*iSigma_mi)*((-1j*y_spos(:)*pi).*Agrid_updated(:,m_iterK)));
        u_secder_Rtildec=(u_secder_q_uv_sq-Rtildec*u_secder_s_uv)/s_uv-2*u_der_s_uv*u_der_Rtildec/s_uv;
        v_secder_Rtildec=(v_secder_q_uv_sq-Rtildec*v_secder_s_uv)/s_uv-2*v_der_s_uv*v_der_Rtildec/s_uv;
        uv_secder_Rtildec=(uv_secder_q_uv_sq-u_der_q_uv_sq*v_der_s_uv/s_uv-Rtildec*uv_secder_s_uv+Rtildec*u_der_s_uv*v_der_s_uv/s_uv)/s_uv-u_der_s_uv*v_der_Rtildec/s_uv;
        vu_secder_Rtildec=(vu_secder_q_uv_sq-v_der_q_uv_sq*u_der_s_uv/s_uv-Rtildec*vu_secder_s_uv+Rtildec*u_der_s_uv*v_der_s_uv/s_uv)/s_uv-v_der_s_uv*u_der_Rtildec/s_uv;
        Hessian_Rtildec=[u_secder_Rtildec uv_secder_Rtildec; vu_secder_Rtildec v_secder_Rtildec];
        if max(eig(Hessian_Rtildec))<0
            uv_candidate=[u_grid_updated(m_iterK); v_grid_updated(m_iterK)]-Hessian_Rtildec\grad_Rtildec;
            u_candidate=uv_candidate(1);v_candidate=uv_candidate(2);
            
            Aadpt_grid=exp(-1j*pi*(x_spos(:)*[u_old u_candidate]+y_spos(:)*[v_old v_candidate]));
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
                v_grid_updated(m_iterK)=v_candidate;
                Agrid_updated(:,m_iterK)=exp(-1j*pi*(x_spos(:)*u_candidate+y_spos(:)*v_candidate));
                %             wi_vec_add=iSigma_mi*Agrid_updated(:,m_iterK);
                %             iSigma=iSigma_mi-wi_vec_add*wi_vec_add'/(1/gamma_est(m_iterK)+Agrid_updated(:,m_iterK)'*wi_vec_add);
            end
            %% Likelihood Evaluation-Based Update
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
%     %Plotting
%     figure(2)
%     plot(u_grid_updated(m_candidates(1)), v_grid_updated(m_candidates(1)), 'o','Color',1-[iterGdPtAdPks iterGdPtAdPks 0]/ITERGdPtAdPks, 'LineWidth', 2, 'MarkerSize',10)
%     plot(u_grid_updated(m_candidates(2)), v_grid_updated(m_candidates(2)), 'o','Color',1-[iterGdPtAdPks iterGdPtAdPks 0]/ITERGdPtAdPks, 'LineWidth', 2, 'MarkerSize',10)
end
end

function [PKS,LOCS]=findpeaks1(Y) % assumes Y is a non-negative vector
diff1=diff([0 reshape(Y,1,[]) 0]);
LOCS=find(diff1(1:end-1)>0 & diff1(2:end)<0);
PKS=Y(LOCS);
end