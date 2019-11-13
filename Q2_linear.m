clear all
close all
P1=0.35;
P2=0.65;
cov1=[1 0;0 1];
mu1=[0;0];
j=1;
k=1;
N1=0;
N2=0;
for i=1:1000
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       x1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else
        c=2;
        d=3;
        e=(d-c).*rand(1,1)+c;
        f=-pi;
        g=+pi;
        h=(g-f).*rand(1,1)+f;
        x2(1,k)= e*cos(h);
        x2(2,k)= e*sin(h);
        N2=N2+1;
        k=k+1;
       
    end
end
plot(x1(1,:),x1(2,:),'.')
hold on
plot(x2(1,:),x2(2,:),'.')
axis equal
title('Generated Data')
xlabel("Feature 1")
ylabel("Feature 2")
legend('Class 1','Class 2')
label1=zeros(N1,1);
label2=ones(N2,1);
label=[label1 ;label2];
%label=label';
x=[x1 x2];
x=x';
SVMModel = fitcsvm(x,label,'Standardize',true,'KernelFunction','linear','OptimizeHyperparameters','auto');
box_const= SVMModel.BoxConstraints;
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel)
N1=0;
N2=0;
k=1;
j=1;
for i=1:1000
    a=0;
    b=1;
    r=(b-a).*rand(1,1)+a;
    if r< P1
       X1(:,j)= mvnrnd(mu1,cov1,1);
       N1=N1+1;
       j=j+1;
    else
        c=2;
        d=3;
        e=(d-c).*rand(1,1)+c;
        f=-pi;
        g=+pi;
        h=(g-f).*rand(1,1)+f;
        X2(1,k)= e*cos(h);
        X2(2,k)= e*sin(h);
        N2=N2+1;
        k=k+1;
       
    end
end

label1_new=zeros(N1,1);
label2_new=ones(N2,1);
label_new=[label1_new ;label2_new];
%label=label';
X=[X1 X2];
X=X';
pred_label_new=predict(SVMModel,X);
Y=0;
Z=0;
for i=1:1000
    if pred_label_new(i)==label_new(i)
        Y=Y+1;
    else
        Z=Z+1;
    end
end
Error_percentage =(Z/1000)*100
hold off
figure(5)
plot(QW(:,1),QW(:,2),'r.');
hold on
plot(AS(:,1),AS(:,2),'b+');
axis equal
title('Classified Data')
xlabel("Feature 1")
ylabel("Feature 2")
legend('Correct','Misclassified')