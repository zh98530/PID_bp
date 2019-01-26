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
    dK(j)=2/(exp(K(j))+exp(-K(j)))^2;clc,clear;

learning_rate=0.20;
inertia_factor=0.01;

INPUT_NODE = 4;
HIDDEN_NODE = 5;
OUTPUT_NODE = 3;

w_hid = 0.50*rands(HIDDEN_NODE,INPUT_NODE);
w_hid_1 = w_hid;
w_hid_2 = w_hid;
w_hid_3 = w_hid;

w_out = 0.50*rands(OUTPUT_NODE,HIDDEN_NODE);
w_out_1 = w_out;
w_out_2 = w_out;
w_out_3 = w_out;

TS = 20;%定义采样周期
x = [0,0,0];%定义P.I.D三个参数的初值


u_1 = 0;u_2 = 0;u_3 = 0;u_4 = 0;u_5 = 0;
yout_1 = 0;yout_2 = 0;yout_3 = 0;

HID_out = zeros(HIDDEN_NODE,1);
HID_in = HID_out;

error_1 = 0;
error_2 = 0;


for k = 1:1:500
    %% 前向传播
    time(k) = k*TS;%时间参考，每20s记录一次
    rin(k) = 1.0;%参考阶跃信号
    
    sys = tf(1.2,[208 1],'inputdelay',80); % 建立被控对象传递函数
    dsys = c2d(sys,TS,'zoh'); % 把传递函数离散化
    [num,den] = tfdata(dsys,'v'); % 离散化后提取分子、分母
    yout(k) = -den(2)*yout_1 + num(2)*u_5;%零阶保持器y(k)*den(1)+y(k-T)*den(2)=u(k)*num(1)+u(k-T)*num(2)
    
    error(k) = rin(k) - yout(k);
    IN_out=[rin(k),yout(k),error(k),1];%BP输入参数
    
    P_error = error(k)-error_1; % 比例输出
    I_error = error(k); % 积分输出
    D_error = error(k)-2*error_1+error_2; % 微分输出
    PID_error=[P_error;I_error;D_error];
    
    HID_in = IN_out*w_hid';
    for j = 1:1:HIDDEN_NODE
        HID_out(j) = ( exp(HID_in(j)) - exp(-HID_in(j)) )/( exp(HID_in(j)) + exp(-HID_in(j)) );
    end
    
    OUT_in = w_out*HID_out;
    for L = 1:1:OUTPUT_NODE
        OUT_out(L) =  exp(OUT_in(L))/( exp(OUT_in(L)) + exp(-OUT_in(L)) );
    end
    
    Kp(k) = OUT_out(1);
    Ki(k) = OUT_out(2);
    Kd(k) = OUT_out(3);
    Kpid = [Kp(k),Ki(k),Kd(k)];
    du(k) = Kpid*PID_error;
    u(k) = u_1 + du(k);%P.I.D控制输出函数
    
    if u(k)>=10 % 控制器饱和处理
        u(k)=10;
    end
    if u(k)<=-10
        u(k)=-10;
    end
    
    %% 反向传播
    %w_out
    dyout_u(k) = sign( ( yout(k)-yout_1 )/( u(k)-u_1+0.0000001 ) );%被控对象输出函数对P.I.D控制输出函数求导，只取符号不取数值
    for j=1:1:OUTPUT_NODE
        dOUT_out_OUT_in(j)=2/( exp(OUT_out(j)) + exp(-OUT_out(j)) )^2;%输出层输出函数对输入函数求导，即输出层激活函数求导
    end
    
    for L=1:1:OUTPUT_NODE
        dE_OUT_in(L)=error(k)*dyout_u(k)*PID_error(L)*dOUT_out_OUT_in(j);%损失函数(rin(k) - yout(k))^2/2对OUT_in求导，error为损失函数对yout求导，PID_error为pid控制输出函数对输出层输出函数求导
    end
    
    for L=1:1:OUTPUT_NODE
        for i=1:1:HIDDEN_NODE
            dE_w_out=learning_rate*dE_OUT_in(L)*HID_out(i)+inertia_factor*(w_out_1-w_out_2);
        end
    end
    w_out = w_out_1 + dE_w_out;%输出层参数更新
    
    %w_hide
    for i = 1:1:HIDDEN_NODE
        dHID_out_HID_in(i) = 4/( exp(HID_out(i)) + exp(-HID_out(i)) )^2;
    end
    dE_HID_out = dE_OUT_in*w_out;
    for i = 1:1:HIDDEN_NODE
        dE_HID_in(i) = dE_HID_out(i)*dHID_out_HID_in(i);
    end
    dE_w_hid=learning_rate*dE_HID_in'*IN_out;
    w_hid=w_hid_1+dE_w_hid+inertia_factor*(w_hid_1-w_hid_2); 
    
    u_5=u_1;
    u_4=u_3;u_3=u_2;u_2=u_1;
    u_1=u(k);
    yout_2=yout_1;yout_1=yout(k);
    w_out_3=w_out_2;
    w_out_2=w_out_1;
    w_out_1=w_out;
    w_hid_3=w_hid_2;
    w_hid_2=w_hid_1;
    w_hid_1=w_hid;
    error_2=error_1;
    error_1=error(k);
end
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
plot(time,Kp,'r');
xlabel('time(s)');ylabel('kp');
subplot(312);
plot(time,Ki,'g');
xlabel('time(s)');ylabel('ki');
subplot(313);
plot(time,Kd,'b'); 
xlabel('time(s)');ylabel('kd'); 
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
