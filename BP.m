clc,clear;


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
