clear all;
close all;

%%
%Load data and create variables
load data.mat
ttl = vertcat(data(1:15).rec); 
ttl(:,6) = (ttl(:,3)==ttl(:,4) | ttl(:,3)-7==ttl(:,4));
for i=1:length(ttl)
    if ttl(i,3) >7
        ttl(i,7)= ttl(i,3)-7;
    else
        ttl(i,7)= ttl(i,3);
    end
end

Times = data(1).tim;

for e = 2:7
    for t = 1:length(Times)
        performance = ttl(:,6)==1 & ttl(:,7)==e & ttl(:,2)==Times(t);
        ttl_performance(e-1,t) = sum(performance) / sum(ttl(:,2)==Times(t) & ttl(:,7)==e);
    end
    
    plot(Times, ttl_performance(e-1,:));
    hold on;
    
    txt(e).legend = strcat('Emotion #', num2str(e));
    legend(txt(2:e).legend,'location','southwest');
    title('total Performance per Emotion');
    ylim([0, 1]);
end

%% SIMPLEX MLE, All subjects
figure;
for a = 1:6
    x = Times;
    y = ttl_performance(a,:);
    
    ga_fun = @(b) mle_logit(b,x,y);
    startParms = [0 0.1];
    [b_sim, finDiscrepancy] = fminsearch(ga_fun, startParms);
    
    subplot(3,2,a)
    hold on
    YHat = 1./(1+exp(-1*b_sim(1)*(x-b_sim(2))));
    plot(x, y,x, YHat);
    
    txt(a).title = strcat('Emotion ', num2str(a),' simplex');
    title(txt(a).title);
    
    b_simplex(a,:) = b_sim;
end

%% GA MLE, All Subjects
figure;
for a = 1:6
    
    x = Times;
    y = ttl_performance(a,:);
    
    ga_fun = @(b) mle_logit(b,x,y);
    options = gaoptimset('Display', 'off', 'Generations', 50);
    [b_ga,fval,exitflag,output] = ga(ga_fun, 2, [], [], [], [], [-10, -10], [1, 1], [], options);
      b = b_ga;
     YHat = 1./(1+exp(-1*b(1)*(x-b(2))));
    subplot(2,3,a)
    plot(Times, y, x,YHat);

    txt(a).title = ['emotion ', num2str(a),', GA'];
    title(txt(a).title)

    b_ga(a,:) = b_ga;
   
end

%% Bootstrap confidence interval, ttl performance (simplex method)
for a = 1:6
    
    x = Times;
    y = ttl_performance(a,:);
    
    for i = 1:100
        ind = randi(6, 1, 6);
        bts(i).ind = ind;
        bts(i).x = x(ind);
        bts(i).y = y(ind);
        
        ga_fun = @(b) mle_logit(b, bts(i).x, bts(i).y);
        
        startParms = [0 .1];
        
        [b_sim, finDiscrepancy] = fminsearch(ga_fun, startParms);
        
        b_bts(i).b1 = b_sim(1);
        b_bts(i).b2 = b_sim(2);
    end
    
    BTS(a).parms = b_bts;
    subplot(6, 2, (2*a-1))
    histfit([b_bts.b1])
    line([b_simplex(a,1) b_simplex(a,1)], ylim, 'color', 'r')
    title(['emotion ', num2str(a),', bts for b1'])
    
    subplot(6, 2, (2*a))
    histfit([b_bts.b2])
    line([b_simplex(a,2) b_simplex(a,2)], ylim, 'color', 'r')
    title(['emotion ', num2str(a), ', bts for b2'])
end



%% Individual subject logit fitting (MLE through simplex)

Times = data(1).tim;

for i = 1:15
    subjectData = data(i).rec;
    subject(i).data = subjectData;
    
    subject(i).data(:,6) = (subject(i).data(:,3) == subject(i).data(:,4)) | (subject(i).data(:,3)-7 == subject(i).data(:,4));
    for j= 1:length(subject(1).data)
        if subject(i).data(j,3) >7
            subject(i).data(j,7)= (subject(i).data(j,3))-7;
        else
            subject(i).data(j,7)= subject(i).data(j,3);
        end
    end
    
    for e = 2:7
        for t = 1:length(Times)
            trials = (subject(i).data(:,7) == e & subject(i).data(:,2) == Times(t));
            subject(i).performance(e-1,t) = sum(subject(i).data(trials,6)) / sum(trials);
        end
    end
end

for i=1:15
    figure;
    for a = 1:6
        x = Times;
        y = subject(i).performance(a,:);
        
        ga_fun = @(b) mle_logit(b,x,y);
        
        startParms = [0 0.1];
        [b_sim, finDiscrepancy] = fminsearch(ga_fun, startParms);
        
        subplot(2,3,a)
        YHat = 1./(1+exp(-1*b_sim(1)*(x-b_sim(2))));
        
        plot(x, y, x,YHat);
        txt(a).title = strcat('Emotion, ', num2str(a),', subject ,' , num2str(i));
        title(txt(a).title);
        
        b_sj(i).b(a,1) = b_sim(1);
        b_sj(i).b(a,2) = b_sim(2);
        
    end
end

%% parameter distributions

%b1 per emotion
figure;
for j=1:15
    for i=1:6
        b1_values(j,i)=b_sj(j).b(i,1);
        
    end
end

for ii= 1:6
    b1_ds(ii)= fitdist(b1_values(:,ii),'Normal');
    b1_sd(ii)= b1_ds(1,ii).sigma;
    b1_se(ii)=b1_sd(ii)/sqrt(15);
    subplot (2,3, ii)
    histfit(b1_values(ii,:))
    text(ii).titles= strcat('emotion # ' , num2str(ii), ', b1' );
    title (text(ii).titles)
end

%b2 per emotion
figure;
for j=1:15
    for i=1:6
        b2_values(j,i)=b_sj(j).b(i,2);
      
    end
end

for ii= 1:6
    subplot (2,3, ii)
    b2_ds(ii)= fitdist(b2_values(:,ii),'Normal');
        b2_sd(ii)= b2_ds(1,ii).sigma;
        b2_se(ii)=b2_sd(ii)/sqrt(15);
    histfit(b2_values(ii,:))
    text(ii).titles= strcat('emotion # ', num2str(ii) , ', b1');
    title (text(ii).titles)
    
end

%% Anova for parameters per emotions
%rows are subjects(15), columns(groups) are emotions(6),

[p_b1,tbl,stats] = anova1(b1_values); 

[p_b2,tbl,stats] = anova1(b2_values);

%% Standard Error confidence interval

figure;
subplot (211)
errorbar([2:7],b1_sd,b1_se,'-s','MarkerEdgeColor', 'red')
title( 'standard deviation and standard error for b1')
xlabel('emotion number')


subplot (212)
errorbar([2:7],b2_sd,b2_se,'-s','MarkerEdgeColor','red')
title( 'standard deviation and standard error for b2')
xlabel('emotion number')


