clc,clear;
close all;
xite=0.20; % 学习速率
alfa=0.01; % 惯性因子
IN=4;H=5;Out=3; %NN Structure

wi=[-0.6394 -0.2696 -0.3756 -0.7023;
-0.8603 -0.2013 -0.5024 -0.2596;
-1.0749 0.5543 -1.6820 -0.5437;
-0.3625 -0.0724 -0.6463 -0.2859;
0.1425 0.0279 -0.5406 -0.7660];
%wi=0.50*rands(H,IN); % 隐含层加权系数 wi 初始化
wi_1=wi;wi_2=wi;wi_3=wi;

wo=[0.7576 0.2616 0.5820 -0.1416 -0.1325;
-0.1146 0.2949 0.8352 0.2205 0.4508;
0.7201 0.4566 0.7672 0.4962 0.3632];
%wo=0.50*rands(Out,H); % 输出层加权系数 wo初始化
wo_1=wo;wo_2=wo;wo_3=wo;

ts=20; % 采样周期取值
x=[0,0,0]; % 比例，积分，微分赋初值
u_1=0;
u_2=0;u_3=0;u_4=0;
u_5=0;
y_1=0;y_2=0;y_3=0;
Oh=zeros(H,1); %Output from NN middle layer 隐含层的输出
I=Oh; %Input to NN middle layer 隐含层输入
error_2=0;
error_1=0;
for k=1:1:500 % 仿真开始，共 500 步
time(k)=k*ts; %每20记录一次
rin(k)=1.0;%参照函数
%Delay plant
sys=tf(1.2,[208 1],'inputdelay',80); % 建立被控对象传递函数 ?
dsys=c2d(sys,ts,'zoh'); % 把传递函数离散化 ?
[num,den]=tfdata(dsys,'v'); % 离散化后提取分子、分母
yout(k)=-den(2)*y_1+num(2)*u_5;

error(k)=rin(k)-yout(k);%输出函数与参考函数的误差
xi=[rin(k),yout(k),error(k),1];%BP输入参数

x(1)=error(k)-error_1; % 比例输出
x(2)=error(k); % 积分输出
x(3)=error(k)-2*error_1+error_2; % 微分输出
epid=[x(1);x(2);x(3)];

I=xi*wi';% 隐含层的输入，即：输入层输入 * 权值
for j=1:1:H
Oh(j)=(exp(I(j))-exp(-I(j)))/(exp(I(j))+exp(-I(j))); %Middle Layer 在激活函数作用下隐含层的输出
end

K=wo*Oh; %Output Layer 输出层的输入，即：隐含层的输出 * 权值
for l=1:1:Out
K(l)=exp(K(l))/(exp(K(l))+exp(-K(l))); %Getting kp,ki,kd 输出层的输出，即三个 pid 控制器的参数
end

kp(k)=K(1);ki(k)=K(2);kd(k)=K(3);
Kpid=[kp(k),ki(k),kd(k)];
du(k)=Kpid*epid;
u(k)=u_1+du(k);%P.I.D控制输出函数

if u(k)>=10 % Restricting the output of controller 控制器饱和环节
u(k)=10;
end
if u(k)<=-10
u(k)=-10;
end

%反向传播
% 以下为权值 wi 、wo 的在线调整，参考 刘金琨的《先进 PID 控制》
dyu(k)=sign((yout(k)-y_1)/(u(k)-u_1+0.0000001));%被控对象输出函数对P.I.D控制输出函数求导
%Output layer 输出层
for j=1:1:Out
    dK(j)=2/(exp(K(j))+exp(-K(j)))^2;
end
for l=1:1:Out
    delta3(l)=error(k)*dyu(k)*epid(l)*dK(l);
end

for l=1:1:Out
    for i=1:1:H
        d_wo=xite*delta3(l)*Oh(i)+alfa*(wo_1-wo_2);
    end
end

wo=wo_1+d_wo+alfa*(wo_1-wo_2);
%Hidden layer
for i=1:1:H
dO(i)=4/(exp(I(i))+exp(-I(i)))^2;
end
segma=delta3*wo;
for i=1:1:H
delta2(i)=dO(i)*segma(i);
end
d_wi=xite*delta2'*xi;
wi=wi_1+d_wi+alfa*(wi_1-wi_2); 
%Parameters Update 参数更新
u_5=u_1;
u_4=u_3;u_3=u_2;u_2=u_1;
u_1=u(k);
y_2=y_1;y_1=yout(k);
wo_3=wo_2;
wo_2=wo_1;
wo_1=wo;
wi_3=wi_2;
wi_2=wi_1;
wi_1=wi;
error_2=error_1;
error_1=error(k);
end
% 仿真结束，绘图
figure(1);
plot(time,rin,'r',time,yout,'b');
xlabel('time(s)');ylabel('rin,yout');
figure(2);
plot(time,error,'r');
xlabel('time(s)');ylabel('error');
figure(3);
plot(time,u,'r');
xlabel('time(s)');ylabel('u');
figure(4);
subplot(311);
plot(time,kp,'r');
xlabel('time(s)');ylabel('kp');
subplot(312);
plot(time,ki,'g');
xlabel('time(s)');ylabel('ki');
subplot(313);
plot(time,kd,'b'); 
xlabel('time(s)');ylabel('kd'); 
