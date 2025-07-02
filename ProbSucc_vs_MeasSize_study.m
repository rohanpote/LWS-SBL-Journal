% Array Processing Problem: Seq. SBL vs Matching Pursuit type algorithms
% Impemented: Seq. SBL 1.LS-OMP 2.OMP 3.MP 4.Weak-MP 5.Thresholding

clear all
rng(1)
m = [1:5 10:5:20 30:10:100];
n = 512; %(256 in journal draft)
suppsizelist = [1 5:5:20];
datagen = 2; % 0: complex (for array signal processing), 2: real (for Gaussian random measurement matrices)
epsilon = 1e-2;
% r = Inf; %suppsize;
ITER = 200;
SNR=20;
s_var=1;
w_var=s_var/10^(SNR/10);
sep=1; % make sure (max. sep X 2 X suppsizemax)<=n to ensure sep is always possible

% Grid Initialize
if datagen==0
    A = create_dictionary(m(end),n,'ULA');%randn(m,n);
elseif datagen==2
    A = create_dictionary(m(end),n,'random');%randn(m,n);
end


Anorm = A*diag(1./vecnorm(A));
AtA = Anorm'*Anorm; AtA_tmp = abs(AtA);
AtA_tmp(1:n+1:end)=0;
mc = max(max(AtA_tmp));

sp_l2error = zeros(ITER, length(suppsizelist),length(m)); sp_suppdist = zeros(ITER, length(suppsizelist),length(m));
% sp_orig_l2error = zeros(ITER, length(suppsizelist),length(m)); sp_orig_suppdist = zeros(ITER, length(suppsizelist),length(m));
% lsomp_l2error = zeros(ITER, length(suppsizelist),length(m)); lsomp_suppdist = zeros(ITER, length(suppsizelist),length(m));
seqsbl_l2error = zeros(ITER, length(suppsizelist),length(m)); seqsbl_suppdist = zeros(ITER, length(suppsizelist),length(m)); seqsbl_timecomp = zeros(ITER, length(suppsizelist),length(m));
seqsbl_Cinv_l2error = zeros(ITER, length(suppsizelist),length(m)); seqsbl_Cinv_suppdist = zeros(ITER, length(suppsizelist),length(m)); seqsbl_Cinv_timecomp = zeros(ITER, length(suppsizelist),length(m));
seqsbl_nvarNotgivenwvarby10_l2error = zeros(ITER, length(suppsizelist),length(m)); seqsbl_nvarNotgivenwvarby10_suppdist = zeros(ITER, length(suppsizelist),length(m)); seqsbl_nvarNotgivenwvarby10_timecomp = zeros(ITER, length(suppsizelist),length(m));
seqsbl_nvarNotgivenwvar10_l2error = zeros(ITER, length(suppsizelist),length(m)); seqsbl_nvarNotgivenwvar10_suppdist = zeros(ITER, length(suppsizelist),length(m)); seqsbl_nvarNotgivenwvar10_timecomp = zeros(ITER, length(suppsizelist),length(m));
seqsbl_targetsj_l2error = zeros(ITER, length(suppsizelist),length(m)); seqsbl_targetsj_suppdist = zeros(ITER, length(suppsizelist),length(m)); seqsbl_targetsj_timecomp = zeros(ITER, length(suppsizelist),length(m));
seqsbl_ximprov_l2error = zeros(ITER, length(suppsizelist),length(m)); seqsbl_ximprov_suppdist = zeros(ITER, length(suppsizelist),length(m)); seqsbl_ximprov_timecomp = zeros(ITER, length(suppsizelist),length(m));
redcompomp_l2error = zeros(ITER, length(suppsizelist),length(m)); redcompomp_suppdist = zeros(ITER, length(suppsizelist),length(m)); redcompomp_timecomp = zeros(ITER, length(suppsizelist),length(m));
omp_l2error = zeros(ITER, length(suppsizelist),length(m)); omp_suppdist = zeros(ITER, length(suppsizelist),length(m)); omp_timecomp = zeros(ITER, length(suppsizelist),length(m));
% mp_l2error = zeros(ITER, length(suppsizelist),length(m)); mp_suppdist = zeros(ITER, length(suppsizelist),length(m));
% weakmp_l2error = zeros(ITER, length(suppsizelist),length(m)); weakmp_suppdist = zeros(ITER, length(suppsizelist),length(m));
% thresh_l2error = zeros(ITER, length(suppsizelist),length(m)); thresh_suppdist = zeros(ITER, length(suppsizelist),length(m));
L1_l2error = zeros(ITER, length(suppsizelist),length(m)); L1_suppdist = zeros(ITER, length(suppsizelist),length(m));
% RL1_l2error = zeros(ITER, length(suppsizelist),length(m)); RL1_suppdist = zeros(ITER, length(suppsizelist),length(m));
% reg_IRLS_l2error = zeros(ITER, length(suppsizelist),length(m)); reg_IRLS_suppdist = zeros(ITER, length(suppsizelist),length(m));
% SBL_l2error = zeros(ITER, length(suppsizelist),length(m)); SBL_suppdist = zeros(ITER, length(suppsizelist),length(m));

for iter=1:ITER
    iter
    switch datagen
        case 0
            nonzero_x = randn(suppsizelist(end),1)+1j*randn(suppsizelist(end),1);
            noise_vec=randn(m(end),1)+1j*randn(m(end),1);
        case 5
            nonzero_x = sqrt(3)*rand(suppsizelist(end),1).*(randn(suppsizelist(end),1)+1j*randn(suppsizelist(end),1));
        case 1
            nonzero_x = (rand(suppsizelist(end),1)+1).*(2*(rand(suppsizelist(end),1)>0.5)-1);
        case 2
            nonzero_x = randn(suppsizelist(end),1);
            noise_vec=randn(m(end),1);
        case 3
            nonzero_x = 2*(rand(suppsizelist(end),1)>0.5)-1;
            noise_vec=randn(m(end),1);
        case 4
            nonzero_x = trnd(1,suppsizelist(end),1);
    end
    permute_n=randsample(n, n); % without replacement
    for sep_iter=1:length(sep)
        [suppfull,loop_time] = min_separate(permute_n, suppsizelist(end), sep(sep_iter));%randsample(n, suppsizelist(end));
        %         loop_time
        xfull = zeros(n,1);
        xfull(suppfull)=nonzero_x;
        for isuppsize=1:length(suppsizelist)
            if iter==359 && isuppsize==1
                disp('hi')
            end
            suppsize = suppsizelist(isuppsize);
            supp = suppfull(1:suppsize);
            x = zeros(n,1);
            x(supp) = xfull(supp);
            for miter=1:length(m)
                Atest=A(1:m(miter),:);
                if datagen==0 % complex case
                    y = sqrt(s_var/2)*Atest*x+sqrt(w_var/2)*noise_vec(1:m(miter),:);
                elseif datagen==2 % real case
                    y = sqrt(s_var)*Atest*x+sqrt(w_var)*noise_vec(1:m(miter),:);
                else
                    y = sqrt(s_var)*Atest*x+sqrt(w_var)*noise_vec(1:m(miter),:);
                end
                
                %% Sequential SBL (Computationally May Be Improved Further)
                % Initialization
                
                candidateset=1:n;
                qj=zeros(n,1);
                qsj=zeros(n,1);
                sj=zeros(n,1);
                sigma_n=w_var; % noiseless case, can be adjusted
                gamma_est=zeros(n,1);
                kpset = zeros(suppsize,1);
                Cinv=eye(m(miter))/sigma_n;
                CiA=Cinv*Atest;
                xhat = zeros(n,1);
                w_prev1mat=zeros(m(miter),suppsize);
                % Main Loop
                tic
                %             profile on
                for p=1:suppsize
                    if p==1
                        qj=Atest'*(y/sigma_n);
                        qsj=conj(qj).*qj;% abs(qj).^2;
                        sj=(sum(conj(Atest).*Atest,1).')/sigma_n; %(m(miter)/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
                    end
%                     if p==1
%                         [val,l_prev]=max(qsj(candidateset));
%                         val=sigma_n*val/m(miter);
%                     else
                    [val,l_prev]=max(qsj(candidateset)./sj(candidateset));
%                     end
                    if val>1
                        kp=candidateset(l_prev);
                        kpset(p)=kp;
                        candidateset=candidateset(candidateset~=kp);
                        gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
                        w_norm=sqrt(1/gamma_est(kp)+sj(kp));
                        
                        %                     Cinv_prev=Cinv;
                        %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
                        %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
                        
                        w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*Atest(:,kp));
                        w_prev1=w_prev1/w_norm;
                        w_prev1mat(:,p)=w_prev1;
                        
                        w_prev2=Atest'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
                        qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
                        qsj=conj(qj).*qj;% abs(qj).^2;
                        sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
                    else
                        warning('Seq. SBL did not add new column')
                    end
                end
%                 xhat(kpset(1:p))=gamma_est(kpset(1:p)).*qj(kpset(1:p));
                %             profile off
                %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
                seqsbl_timecomp(iter,isuppsize,miter)=toc;
%                 seqsbl_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                seqsbl_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% Sequential SBL (Uses Inverse)
%                 % Initialization
%                 
%                 candidateset=1:n;
%                 qj=zeros(n,1);
%                 qsj=zeros(n,1);
%                 sj=zeros(n,1);
%                 sigma_n=w_var; % noiseless case, can be adjusted
%                 gamma_est=zeros(n,1);
%                 kpset = zeros(suppsize,1);
%                 Cinv=eye(m(miter))/sigma_n;
%                 CiA=Cinv*Atest;
%                 xhat = zeros(n,1);
%                 w_prev1mat=zeros(m(miter),suppsize);
%                 % Main Loop
%                 tic
%                 %             profile on
%                 for p=1:suppsize
%                     if p==1
%                         qj=Atest'*(y/sigma_n);
%                         qsj=conj(qj).*qj;% abs(qj).^2;
%                         sj=(sum(conj(Atest).*Atest,1).')/sigma_n;% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
%                     end
% %                     if p==1
% %                         [val,l_prev]=max(qsj(candidateset));
% %                         val=sigma_n*val/m(miter);
% %                     else
%                     [val,l_prev]=max(qsj(candidateset)./sj(candidateset));
% %                     end
%                     if val>1
%                         kp=candidateset(l_prev);
%                         kpset(p)=kp;
%                         candidateset=candidateset(candidateset~=kp);
%                         gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
%                         Cinv=eye(m(miter))/((Atest(:,kpset(1:p)).*repmat(gamma_est(kpset(1:p))',m(miter),1))*Atest(:,kpset(1:p))'+sigma_n*eye(m(miter)));
%                         qj=Atest'*(Cinv*y);
% %                         w_norm=sqrt(1/gamma_est(kp)+sj(kp));
% %                         
% %                         %                     Cinv_prev=Cinv;
% %                         %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
% %                         %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
% %                         
% %                         w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*Atest(:,kp));
% %                         w_prev1=w_prev1/w_norm;
% %                         w_prev1mat(:,p)=w_prev1;
% %                         
% %                         w_prev2=Atest'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
% %                         qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                         qsj=conj(qj).*qj;% abs(qj).^2;
%                         sj=real(sum(conj(Atest).*(Cinv*Atest),1)).';
% %                         sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
%                     else
%                         warning('Seq. SBL did not add new column')
%                     end
%                 end
% %                 xhat(kpset(1:p))=gamma_est(kpset(1:p)).*qj(kpset(1:p));
%                 %             profile off
%                 %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
%                 seqsbl_Cinv_timecomp(iter,isuppsize,miter)=toc;
% %                 seqsbl_Cinv_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
%                 seqsbl_Cinv_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                %% Sequential SBL (Lambda set to w_var/10)
%                 % Initialization
%                 
%                 candidateset=1:n;
%                 qj=zeros(n,1);
%                 qsj=zeros(n,1);
%                 sj=zeros(n,1);
%                 sigma_n=w_var/10; % noiseless case, can be adjusted
%                 gamma_est=zeros(n,1);
%                 kpset = zeros(suppsize,1);
%                 Cinv=eye(m(miter))/sigma_n;
%                 CiA=Cinv*Atest;
%                 xhat = zeros(n,1);
%                 w_prev1mat=zeros(m(miter),suppsize);
%                 % Main Loop
%                 tic
%                 %             profile on
%                 for p=1:suppsize
%                     if p==1
%                         qj=Atest'*(y/sigma_n);
%                         qsj=conj(qj).*qj;% abs(qj).^2;
%                         sj=(sum(conj(Atest).*Atest,1).')/sigma_n; % exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
%                     end
% %                     if p==1
% %                         [val,l_prev]=max(qsj(candidateset));
% %                         val=sigma_n*val/m(miter);
% %                     else
%                     [val,l_prev]=max(qsj(candidateset)./sj(candidateset));
% %                     end
%                     if val>1
%                         kp=candidateset(l_prev);
%                         kpset(p)=kp;
%                         candidateset=candidateset(candidateset~=kp);
%                         gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
%                         w_norm=sqrt(1/gamma_est(kp)+sj(kp));
%                         
%                         %                     Cinv_prev=Cinv;
%                         %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
%                         %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
%                         
%                         w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*Atest(:,kp));
%                         w_prev1=w_prev1/w_norm;
%                         w_prev1mat(:,p)=w_prev1;
%                         
%                         w_prev2=Atest'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
%                         qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                         qsj=conj(qj).*qj;% abs(qj).^2;
%                         sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
%                     else
%                         warning('Seq. SBL did not add new column')
%                     end
%                 end
% %                 xhat(kpset(kpset>0))=gamma_est(kpset(kpset>0)).*qj(kpset(kpset>0));
%                 %             xhat(kpset(1:p))=gamma_est(kpset(1:p)).*qj(kpset(1:p));
%                 %             profile off
%                 %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
%                 seqsbl_nvarNotgivenwvarby10_timecomp(iter,isuppsize,miter)=toc;
% %                 seqsbl_nvarNotgivenwvarby10_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
%                 seqsbl_nvarNotgivenwvarby10_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% Sequential SBL (Lambda set to w_var*10)
%                 % Initialization
%                 
%                 candidateset=1:n;
%                 qj=zeros(n,1);
%                 qsj=zeros(n,1);
%                 sj=zeros(n,1);
%                 sigma_n=w_var*10; % noiseless case, can be adjusted
%                 gamma_est=zeros(n,1);
%                 kpset = zeros(suppsize,1);
%                 Cinv=eye(m(miter))/sigma_n;
%                 CiA=Cinv*Atest;
%                 xhat = zeros(n,1);
%                 w_prev1mat=zeros(m(miter),suppsize);
%                 % Main Loop
%                 tic
%                 %             profile on
%                 for p=1:suppsize
%                     if p==1
%                         qj=Atest'*(y/sigma_n);
%                         qsj=conj(qj).*qj;% abs(qj).^2;
%                         sj=(sum(conj(Atest).*Atest,1).')/sigma_n; % exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
%                     end
% %                     if p==1
% %                         [val,l_prev]=max(qsj(candidateset));
% %                         val=sigma_n*val/m(miter);
% %                     else
%                     [val,l_prev]=max(qsj(candidateset)./sj(candidateset));
% %                     end
%                     if val>1
%                         kp=candidateset(l_prev);
%                         kpset(p)=kp;
%                         candidateset=candidateset(candidateset~=kp);
%                         gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
%                         w_norm=sqrt(1/gamma_est(kp)+sj(kp));
%                         
%                         %                     Cinv_prev=Cinv;
%                         %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
%                         %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
%                         
%                         w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*Atest(:,kp));
%                         w_prev1=w_prev1/w_norm;
%                         w_prev1mat(:,p)=w_prev1;
%                         
%                         w_prev2=Atest'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
%                         qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
%                         qsj=conj(qj).*qj;% abs(qj).^2;
%                         sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
%                     else
%                         warning('Seq. SBL did not add new column')
%                     end
%                 end
% %                 xhat(kpset(kpset>0))=gamma_est(kpset(kpset>0)).*qj(kpset(kpset>0));
%                 %             xhat(kpset(1:p))=gamma_est(kpset(1:p)).*qj(kpset(1:p));
%                 %             profile off
%                 %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
%                 seqsbl_nvarNotgivenwvar10_timecomp(iter,isuppsize,miter)=toc;
% %                 seqsbl_nvarNotgivenwvar10_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
%                 seqsbl_nvarNotgivenwvar10_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% Sequential SBL (Targeted sj computations)
                %             % Initialization
                %
                %             candidateset=1:n;
                %             qj=zeros(n,1);
                %             qsj=zeros(n,1);
                %             sj=zeros(n,1);
                %             sigma_n=w_var; % noiseless case, can be adjusted
                %             gamma_est=zeros(n,1);
                %             kpset = zeros(suppsize,1);
                %             Cinv=eye(m)/sigma_n;
                %             CiA=Cinv*A;
                %             sj_init=(m/sigma_n)*ones(10,1);
                %             xhat = zeros(n,1);
                %             w_prev1mat=zeros(m,suppsize);
                %             w_prev2mat=zeros(n,suppsize);
                %             % Main Loop
                %             %             profile on
                %             tic
                %             for p=1:suppsize
                %                 if p==1
                %                     qj=A'*(y/sigma_n);
                %                     qsj=conj(qj).*qj;% abs(qj).^2;
                %                     sj=(m/sigma_n)*ones(n,1);% exploiting structure in (sum(conj(A).*A,1).')/sigma_n;
                %                 end
                %                 if p==1
                %                     [val,l_prev]=max(qsj(candidateset));
                %                     val=sigma_n*val/m;
                %                     kp=candidateset(l_prev);
                %                     %                 else
                %                     %                     [val,l_prev]=max(qsj(candidateset)./sj(candidateset));
                %                 end
                %                 if val>1
                %                     kpset(p)=kp;
                %                     candidateset=candidateset(candidateset~=kp);
                %                     gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
                %                     w_norm=sqrt(1/gamma_est(kp)+sj(kp));
                %
                %                     %                     Cinv_prev=Cinv;
                %                     %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
                %                     %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
                %
                %                     w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*A(:,kp));
                %                     w_prev1=w_prev1/w_norm;
                %                     w_prev1mat(:,p)=w_prev1;
                %
                %                     w_prev2=A'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
                %                     w_prev2mat(:,p)=w_prev2;
                %                     qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
                %                     qsj=conj(qj).*qj;% abs(qj).^2;
                %                     [~,creamset]=maxk(qsj(candidateset),10);
                %                     creamset=candidateset(creamset);
                %                     sj(creamset)=sj_init-sum(abs(w_prev2mat(creamset,1:p).^2),2);%sum(w_prev2mat(creamset,1:p).*conj(w_prev2mat(creamset,1:p)),2);
                %                     [val,l_prev]=max(qsj(creamset)./sj(creamset));
                %                     kp=creamset(l_prev);
                %                 else
                %                     warning('Seq. SBL did not add new column')
                %                 end
                %             end
                %             xhat(kpset(1:p))=gamma_est(kpset(1:p)).*qj(kpset(1:p));
                %             %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
                %             seqsbl_targetsj_timecomp(iter,isuppsize,miter)=toc;
                %             %             profile off
                %             seqsbl_targetsj_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %             seqsbl_targetsj_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% Sequential SBL (x-estim extra iterations)
                %             % Initialization
                %
                %             candidateset=1:n;
                %             qj=zeros(n,1);
                %             qsj=zeros(n,1);
                %             sj=zeros(n,1);
                %             sigma_n=w_var; % noiseless case, can be adjusted
                %             gamma_est=zeros(n,1);
                %             kpset = zeros(suppsize,1);
                %             Cinv=eye(m)/sigma_n;
                %             CiA=Cinv*A;
                %             xhat = zeros(n,1);
                %             w_prev1mat=zeros(m,suppsize);
                %             % Main Loop
                %             tic
                %             for p=1:suppsize
                %                 if p==1
                %                     qj=A'*(y/sigma_n);
                %                     qsj=conj(qj).*qj;% abs(qj).^2;
                %                     sj=(sum(conj(A).*A,1).')/sigma_n;
                %                 end
                %                 if p==1
                %                     [val,l_prev]=max(qsj(candidateset));
                %                     val=sigma_n*val/m;
                %                 else
                %                     [val,l_prev]=max(qsj(candidateset)./sj(candidateset));
                %                 end
                %                 if val>1
                %                     kp=candidateset(l_prev);
                %                     kpset(p)=kp;
                %                     candidateset=candidateset(candidateset~=kp);
                %                     gamma_est(kp)=(val-1)/sj(kp);%(qsj(kp)-sj(kp))/(sj(kp)^2);
                %                     w_norm=sqrt(1/gamma_est(kp)+sj(kp));
                %
                %                     Cinv_prev=Cinv;
                %                     w_prev1=Cinv_prev*(A(:,kp)/w_norm);
                %                     if suppsize>1; Cinv=Cinv_prev-w_prev1*w_prev1'; end % find way to avoid this computation!
                %
                % %                     w_prev1=CiA(:,kp)-w_prev1mat(:,1:p-1)*(w_prev1mat(:,1:p-1)'*A(:,kp));
                % %                     w_prev1=w_prev1/w_norm;
                % %                     w_prev1mat(:,p)=w_prev1;
                %
                %                     w_prev2=A'*w_prev1; %(qj(kp)/w_norm)*w_prev1;
                %                     qj=qj-(qj(kp)/w_norm)*w_prev2;% slight abuse of notation, for `j' in model qj (MVDR) is actually Qj (MPDR). For `j' in candidateset qj (MVDR) is correct
                %                     qsj=conj(qj).*qj;% abs(qj).^2;
                %                     sj=sj-conj(w_prev2).*w_prev2;%abs(w_prev2).^2;
                %                 else
                %                     warning('Seq. SBL did not add new column')
                %                 end
                %             end
                %
                %             for x_loop=1:5
                %                 for kploop=1:suppsize
                %                     kp=kpset(kploop);
                %                     gamma_est_kp=gamma_est(kp);
                %                     gamma_est_kp_new=(qsj(kp)-sj(kp))/(sj(kp)^2);
                %                     if gamma_est_kp_new>0
                %                         gamma_est(kp)=gamma_est_kp_new;
                %                         Cinv_prev=Cinv;
                %                         w_norm=sqrt(1/(gamma_est(kp)-gamma_est_kp)+sj(kp));
                %                         w_prev1=Cinv_prev*(A(:,kp)/w_norm);
                %                         Cinv=Cinv_prev-w_prev1*w_prev1';
                %                         w_prev2=(qj(kp)/w_norm)*w_prev1;
                %                         qj(kpset)=qj(kpset)-A(:,kpset)'*w_prev2;
                %                         qsj(kpset)=conj(qj(kpset)).*qj(kpset);% abs(qj).^2;
                %                         w_prev0=w_norm*w_prev1;
                %                         sj(kpset)=sj(kpset)-(abs(A(:,kpset)'*w_prev0).^2)/w_norm/w_norm;
                %                     end
                %                 end
                %             end
                %             xhat(kpset)=gamma_est(kpset).*qj(kpset);
                % %             xhat(kpset(1:p))=diag(gamma_est(kpset(1:p)))*A(:,kpset(1:p))'*((A*diag(gamma_est)*A'+sigma_n*eye(m))\y);%pinv(A(:,kpset(1:p)))*y;
                %             seqsbl_ximprov_timecomp(iter,isuppsize,miter)=toc;
                %             seqsbl_ximprov_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %             seqsbl_ximprov_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% 2.OMP
                %         p=0;
                bp = y;
                %         bpsqnorm = vecnorm(y)^2;
                qkp = zeros(m(miter),0);
                %         Atbp = A'*bp;
                Qp = qkp;
                kpset = zeros(1,suppsize);
                xhat = zeros(n,1);
                % OMP Support Recovery
                tic
                for p=1:suppsize
                    %             tic
                    %         while bpsqnorm>epsilon^2
                    Atbp = Atest'*bp;
                    Atbpabs = abs(Atbp);
                    [~, kp] = max(Atbpabs./vecnorm(Atest).');
                    kpset(p) = kp; %kpset = union(kpset,kp);
                    qkp = Atest(:,kp)-Qp*(Qp'*Atest(:,kp));
                    qkp = qkp/vecnorm(qkp);
                    Qp = [Qp qkp];
                    %             bpsqnorm = bpsqnorm-(abs(qkp'*bp))^2;
                    bp = bp-qkp*(qkp'*bp);
                    %             toc
                    %             p = p+1;
                end
%                 xhat(kpset) = pinv(Atest(:,kpset))*y;
                omp_timecomp(iter,isuppsize,miter)=toc;
%                 omp_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                omp_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% 2a.OMP (Reduced Complexity)
%                 % Initialization
%                 bp = y;
%                 Qp = zeros(m(miter),suppsize);
%                 kpset = zeros(1,suppsize);
%                 ynorm=zeros(suppsize,1);
%                 Rmat=zeros(suppsize);
%                 xhat = zeros(n,1);
%                 %             candidateset=1:n;
%                 kp=0;
%                 % OMP Support Recovery
%                 tic
%                 for p=1:suppsize
%                     Atbp = Atest'*bp;
%                     Atbpabs = abs(Atbp);
%                     [~, kp] = max(Atbpabs./vecnorm(Atest).'); %kp=candidateset(kp);
%                     kpset(p) = kp;
%                     if p==1
%                         qkp = Atest(:,kp);
%                     else
%                         Rmat(1:p-1,p)=Qp(:,1:p-1)'*Atest(:,kp);
%                         qkp = Atest(:,kp)-Qp(:,1:p-1)*Rmat(1:p-1,p);
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
% %                 xhat(kpset) = linsolve(Rmat,ynorm,opts);
%                 redcompomp_timecomp(iter,isuppsize,miter)=toc;
% %                 redcompomp_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
%                 redcompomp_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                
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
                %         lsomp_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %         lsomp_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                
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
                %             [~, kp] = max(Atbpabs./vecnorm(Atest));
                %             kpset = union(kpset,kp);
                %             bp = bp-A(:,kp)*Atbp(kp);
                %             bpsqnorm = bpsqnorm-Atbpabs(kp)^2;
                %             p = p+1;
                %         end
                %         xhat = zeros(m,1);
                %         xhat(kpset) = pinv(A(:,kpset))*y;
                %         mp_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %         mp_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
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
                %         weakmp_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %         weakmp_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% Subspace Pursuit
%                         p=0;
%                         s=suppsizelist(isuppsize);
%                         bp = y;
%                         bpsqnorm = vecnorm(y)^2;
%                         Ip = []; kpset = [];
%                         while bpsqnorm>epsilon^2
%                             Atbp = Atest'*bp;
%                             Atbpabs = abs(Atbp);
%                             [~, Ip_tilde] = maxk(Atbpabs,s);
%                             Ip_tilde = union(Ip_tilde,Ip);
%                             xp_tilde = pinv(Atest(:,Ip_tilde))*y;
%                             [~,Ip] = maxk(abs(xp_tilde),s); Ip = Ip_tilde(Ip);
%                             kpset = Ip;
%                             xp = pinv(Atest(:,Ip))*y;
%                             bp = y-Atest(:,Ip)*xp;
%                 %             bpsqnorm = vecnorm(bp)^2;
%                             p = p+1;
% %                             clf
%                             xhat = zeros(n,1);
%                             xhat(kpset) = xp;
% %                             plot(abs(xhat)); hold on
% %                             plot(abs(x))
% %                             xlabel('Component Index i')
% %                             ylabel('Absolute value |x_i|')
% %                             title({['Iteration no.=' num2str(iter) 'Support size=' num2str(suppsize)],...
% %                                 ['Ip={' num2str(sort(Ip)') '},  |Ip\cap Supp|/|Supp|=' num2str(length(intersect(supp,Ip))/suppsize)],...
% %                                 ['Ip\_tilde={' num2str(sort(Ip_tilde)') '},  |Ip\_tilde\cap Supp|/|Supp|=' num2str(length(intersect(supp,Ip_tilde))/suppsize)],...
% %                                 ['Suppport={' num2str(sort(supp)') '}']})
% %                             legend('estimated x', 'actual x')
% %                             drawnow
% %                             pause(0.15)
%                             if vecnorm(bp)^2/bpsqnorm>=1
%                                 break
%                             else
%                                 bpsqnorm = vecnorm(bp)^2;
%                             end
%                         end
%                         xhat = zeros(n,1);
%                         xhat(kpset) = pinv(Atest(:,kpset))*y;
%                         sp_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
%                         sp_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                %% Subspace Pursuit Original Code
                %         s=suppsizelist(isuppsize);
                %         Rec = CSRec_SP(s,A,y);
                %         xhat = Rec.x_hat; [~,kpset] = maxk(abs(xhat),s);
                %         sp_orig_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %         sp_orig_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
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
                %         thresh_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %         thresh_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                %% 6. L1-minimization (Using MATLAB linprog)
%                         xhat = linprog(ones(2*n,1),[],[],[Atest -Atest],y,zeros(2*n,1));
%                         xhat = xhat(1:n)-xhat(n+1:end);
%                         [~,kpset] = maxk(abs(xhat),suppsize);%find(abs(xhat)>1e-4);
%                         L1_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
%                         L1_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
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
                %         RL1_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %         RL1_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
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
                %         reg_IRLS_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %         reg_IRLS_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
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
                %         SBL_l2error(iter,isuppsize,miter) = vecnorm(x-xhat)^2/vecnorm(x)^2;
                %         SBL_suppdist(iter,isuppsize,miter) = 1-length(intersect(supp,kpset))/max(suppsize,length(kpset));
                
                
            end
        end
    end
end
% save('ProbSucc_vs_MeasSize_study_random.mat')
figure
% support size=1
sbl_supp_succ_rate=permute(sum(seqsbl_suppdist==0,1),[3 2 1]);%seqsbl_Cinv_suppdist
sbl_supp_succ_prob=sbl_supp_succ_rate/ITER;%meas. size X supp. size
plot(m,sbl_supp_succ_prob(:,1), '-sb', 'LineWidth', 2, 'MarkerSize', 15)
hold on
omp_supp_succ_rate=permute(sum(omp_suppdist==0,1),[3 2 1]);%seqsbl_Cinv_suppdist
omp_supp_succ_prob=omp_supp_succ_rate/ITER;%meas. size X supp. size
plot(m,omp_supp_succ_prob(:,1), '-or', 'LineWidth', 2, 'MarkerSize', 8)

% sp_supp_succ_rate=permute(sum(sp_suppdist==0,1),[3 2 1]);%seqsbl_Cinv_suppdist
% sp_supp_succ_prob=sp_supp_succ_rate/ITER;%meas. size X supp. size
% plot(m,sp_supp_succ_prob(:,1), '-*g', 'LineWidth', 2, 'MarkerSize', 15)

% L1_supp_succ_rate=permute(sum(L1_suppdist==0,1),[3 2 1]);%seqsbl_Cinv_suppdist
% L1_supp_succ_prob=L1_supp_succ_rate/ITER;%meas. size X supp. size
% plot(m,L1_supp_succ_prob(:,1), '-pc', 'LineWidth', 2, 'MarkerSize', 15)

grid on
xlabel('Number of measurements (m)')
ylabel('Success probability')
% legend('LWS-SBL', 'OMP','Subspace Pursuit','L1','Location', 'SouthEast')
legend('LWS-SBL', 'OMP','Location', 'SouthEast')
set(gca, 'FontWeight', 'bold', 'FontSize', 12)
xticks(0:10:100)
yticks(0:0.1:1)
% support size=suppsizelist(isuppsize)
% isuppsize=2; disp(['Support size=' num2str(suppsizelist(isuppsize))])
% plot(m,sbl_supp_succ_prob(:,isuppsize), '-sb', 'LineWidth', 2, 'MarkerSize', 15)
% plot(m,omp_supp_succ_prob(:,isuppsize), '-or', 'LineWidth', 2, 'MarkerSize', 8)
isuppsize=3; disp(['Support size=' num2str(suppsizelist(isuppsize))])
plot(m([1 5:end]),sbl_supp_succ_prob([1 5:end],isuppsize), '-sb', 'LineWidth', 2, 'MarkerSize', 15)
plot(m([1 5:end]),omp_supp_succ_prob([1 5:end],isuppsize), '-or', 'LineWidth', 2, 'MarkerSize', 8)
% plot(m([1 5:end]),sp_supp_succ_prob([1 5:end],isuppsize), '-*g', 'LineWidth', 2, 'MarkerSize', 8)
% plot(m([1 5:end]),L1_supp_succ_prob([1 5:end],isuppsize), '-pc', 'LineWidth', 2, 'MarkerSize', 8)
isuppsize=5; disp(['Support size=' num2str(suppsizelist(isuppsize))])
plot(m([1 5:end]),sbl_supp_succ_prob([1 5:end],isuppsize), '-sb', 'LineWidth', 2, 'MarkerSize', 15)
plot(m([1 5:end]),omp_supp_succ_prob([1 5:end],isuppsize), '-or', 'LineWidth', 2, 'MarkerSize', 8)
% plot(m([1 5:end]),sp_supp_succ_prob([1 5:end],isuppsize), '-*g', 'LineWidth', 2, 'MarkerSize', 8)
% plot(m([1 5:end]),L1_supp_succ_prob([1 5:end],isuppsize), '-pc', 'LineWidth', 2, 'MarkerSize', 8)
% legend('LWS-SBL', 'OMP','Subspace Pursuit','L1','Location', 'SouthEast')
legend('LWS-SBL', 'OMP','Location', 'SouthEast')
%%
unique_bound = 0.5*(1+1/mc);
%% Plotting
figure(7)
% subplot(211)
ax=plot(1:suppsizemax, mean(redcompomp_l2error), '--rx', 'LineWidth', 2, 'MarkerSize',8);
% semilogy(1:suppsizemax, mean(lsomp_l2error), 'LineWidth', 2)
hold on
plot(1:suppsizemax, mean(omp_l2error), '-.ro', 'LineWidth', 2, 'MarkerSize',10)
plot(1:suppsizemax, mean(seqsbl_l2error), '-bd', 'LineWidth', 2, 'MarkerSize',12)
plot(1:suppsizemax, mean(seqsbl_nvarNotgivenwvarby10_l2error), '-.bs', 'LineWidth', 1, 'MarkerSize',18)
plot(1:suppsizemax, mean(seqsbl_nvarNotgivenwvar10_l2error), '--bp', 'LineWidth', 1, 'MarkerSize',20)
% plot(1:suppsizemax, mean(seqsbl_targetsj_l2error), '-gd', 'LineWidth', 2, 'MarkerSize',10)
% semilogy(1:suppsizemax, mean(sp_l2error), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(sp_orig_l2error), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(mp_l2error), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(weakmp_l2error), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(thresh_l2error), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(L1_l2error), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(RL1_l2error), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(reg_IRLS_l2error), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(SBL_l2error), 'LineWidth', 2)
% semilogy(unique_bound*ones(1,length(0:1)), 0:1, '--k')
xlabel('Support size')
ylabel('Average relative L_2 error')
% legend('LS-OMP', 'OMP', 'MP', 'Weak-MP', 'Thresholding', 'L1', 'Re-L1', 'reg\_IRLS', 'SBL')
% legend('LS-OMP', 'OMP', 'L1', 'Re-L1', 'reg\_IRLS', 'SBL')
% legend('LS-OMP', 'OMP', 'SP', 'SP-original')
legend({'OMP-QR decomp.', 'OMP', 'LWS-SBL: \lambda=\sigma_n^2', 'LWS-SBL: \lambda=\sigma_n^2/10','LWS-SBL: \lambda=10*\sigma_n^2'}, 'Location', 'northwest')%, 'Seq. SBL-Target sj')
% l2error_mat=[mean(redcompomp_l2error); mean(omp_l2error); mean(seqsbl_l2error);...
%     mean(seqsbl_nvarNotgivenwvarby10_l2error);mean(seqsbl_nvarNotgivenwvar10_l2error)];% mean(seqsbl_targetsj_suppdist)];
% axis([min(1:suppsizemax) max(1:suppsizemax) min(min(l2error_mat)) max(max(l2error_mat))])
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',16)
xticks(1:suppsizemax)
grid on

figure(8)
% subplot(212)
ax=plot(1:suppsizemax, mean(redcompomp_suppdist), '--rx', 'LineWidth', 2, 'MarkerSize',8);
% semilogy(1:suppsizemax, mean(lsomp_suppdist), 'LineWidth', 2)
hold on
plot(1:suppsizemax, mean(omp_suppdist), '-.ro', 'LineWidth', 2, 'MarkerSize',10)
plot(1:suppsizemax, mean(seqsbl_suppdist), '-bd', 'LineWidth', 2, 'MarkerSize',12)
plot(1:suppsizemax, mean(seqsbl_nvarNotgivenwvarby10_suppdist), '-.bs', 'LineWidth', 1, 'MarkerSize',18)
plot(1:suppsizemax, mean(seqsbl_nvarNotgivenwvar10_suppdist), '--bp', 'LineWidth', 1, 'MarkerSize',20)
% plot(1:suppsizemax, mean(seqsbl_targetsj_suppdist), '-gd', 'LineWidth', 2, 'MarkerSize',10)
% semilogy(1:suppsizemax, mean(sp_suppdist), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(sp_orig_suppdist), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(mp_suppdist), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(weakmp_suppdist), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(thresh_suppdist), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(L1_suppdist), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(RL1_suppdist), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(reg_IRLS_suppdist), 'LineWidth', 2)
% semilogy(1:suppsizemax, mean(SBL_suppdist), 'LineWidth', 2)
xlabel('Support size')
ylabel('Probability of error in support')
% legend('LS-OMP', 'OMP', 'MP', 'Weak-MP', 'Thresholding', 'L1', 'Re-L1', 'reg\_IRLS', 'SBL')
% legend('LS-OMP', 'OMP', 'L1', 'Re-L1', 'reg\_IRLS', 'SBL')
% legend('LS-OMP', 'OMP', 'SP', 'SP-original')
legend({'OMP-QR decomp.', 'OMP', 'LWS-SBL: \lambda=\sigma_n^2', 'LWS-SBL: \lambda=\sigma_n^2/10','LWS-SBL: \lambda=10*\sigma_n^2'}, 'Location', 'northwest')%, 'Seq. SBL-Target sj')
% suppdist_mat=[mean(redcompomp_suppdist); mean(omp_suppdist); mean(seqsbl_suppdist);...
%     mean(seqsbl_nvarNotgivenwvarby10_suppdist); mean(seqsbl_nvarNotgivenwvar10_suppdist)];% mean(seqsbl_targetsj_suppdist)];
% axis([min(1:suppsizemax) max(1:suppsizemax) min(min(suppdist_mat)) max(max(suppdist_mat))])
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',16)
xticks(1:suppsizemax)
% plot(unique_bound*ones(1,length(0:1)), 0:1, '--k')
grid on

figure(9)
ax=plot(1:suppsizemax, 1e3*mean(redcompomp_timecomp), '--rx', 'LineWidth', 2, 'MarkerSize',8);
hold on
plot(1:suppsizemax, 1e3*mean(omp_timecomp), '-.ro', 'LineWidth', 2, 'MarkerSize',10)
plot(1:suppsizemax, 1e3*mean(seqsbl_timecomp), '-bd', 'LineWidth', 2, 'MarkerSize',12)
plot(1:suppsizemax, 1e3*mean(seqsbl_nvarNotgivenwvarby10_timecomp), '-.bs', 'LineWidth', 1, 'MarkerSize',18)
plot(1:suppsizemax, 1e3*mean(seqsbl_nvarNotgivenwvar10_timecomp), '--bp', 'LineWidth', 1, 'MarkerSize',20)
% plot(1:suppsizemax, 1e3*mean(seqsbl_targetsj_timecomp), '-gd', 'LineWidth', 2, 'MarkerSize',10)
xlabel('Support size')
ylabel('Time in milliseconds')
legend({'OMP-QR decomp.', 'OMP', 'LWS-SBL: \lambda=\sigma_n^2', 'LWS-SBL: \lambda=\sigma_n^2/10','LWS-SBL: \lambda=10*\sigma_n^2'}, 'Location', 'northwest')%, 'Seq. SBL-Target sj')
ax=ax.Parent;
set(ax, 'FontWeight', 'bold','FontSize',16)
xticks(1:suppsizemax)
grid on

figure
subplot(211)
histogram(mc)
title('Histogram')
xlabel('Mutual Coherence (\mu(A))')
ylabel('Frequency')
grid on
subplot(212)
plot(0.5*(1+1./mc))
xlabel('Iteration number')
ylabel('0.5(1+1/\mu(A))')
grid on

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