COURSERA - Machine Learning - Octave/Recap
==========================================

## Help

help eye

## Basic operations

5+6
3-2
5*8
1/2

2^6

## Logical operations
1==2
1~=2
1&&0
1||0
xor(1,0)

## Variables

a=3;
b='hi';
c=(3>=1);
a=pi;

## Vectors & Matrices

A = [1 2; 3 4; 5 6]
v = [1 2 3]
v = [1; 2; 3]
v = 1:6
v = 1:0.1:2
ones(2,3)
C = 2*ones(2,3)
w = zeros(1,3)
r = rand(3,3)
n = randn(3,3)

w = -6 + sqrt(10)*(randn(1,10000));
hist(w)

I = eye(4)

## Data

A = [1 2; 3 4; 5 6]
size(A)
mn = size(A)
m = size(A,1)
n = size(A,2)

v = [1 2 3 4]
length(v)

pwd
cd
ls
load FILE.dat
who
whos
clear FILE
save FILE.dat FILE
save FILE.txt FILE -ascii

A = [1 2; 3 4; 5 6]
A(3,2)
A(2,:)
A(:,2)
A([1 3],:)

A(:,2) = [10; 11; 12]
A = [A, [100; 101; 102]]
A = [1 2; 3 4; 5 6]
B = [11 12; 13 14; 15 16]
C = [A B]
C = [A; B]

## Computing on data

A = [1 2; 3 4; 5 6]           # 3x2 matrix
B = [11 12; 13 14; 15 16]     # 3x2 matrix
C = [1 1; 2 2]                # 2x2 matrix
A*C                           # 3x2 matrix
A.*B                          # 3x2 matrix (elementwise moltiplication)
A.^2                          # 3x2 matrix
1./A                          # 3x2 matrix (elementwise inverse)

v = [1; 2; 3]                 # 3x1 vector
log(v)
exp(v)
abs(v)
v + ones(length(v),1)
v + 1

A'                            # 2x3 matrix (transpose of 3x2)

a = [1 15 2 0.5]
val = max(a)
[val, ind] = max(a)
a < 3
find(a < 3)
sum(a)
prod(a)
floor(a)
ceil(a)

A = magic(3)
max(A, [], 1)       # per column max
max(A. [], 2)       # per row max

A = magic(9)
sum(A,1)
sum(A,2)
sum(sum(A.*eye(9)))

pinv(A)

## Plotting data

t = [0:0.01:0.98];
y1 = sin(2*pi*4*t);
y2 = cos(2*pi*4*t):
plot(t,y1);
hold on:
plot(t,y2, 'r' );
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('plot')
print -dpng 'plot.png'
close

## Control statement: for, while, if

v = zeros(10,1)

for i=1:10,
  v(i) = 2^i;
end;

i = 1;
while i <= 5,
  v(i) = 100;
  i = i+1;
end;

i = 1;
while true,
  v(i) = 999;
  i = i+1;
  if i == 6,
    break;
  end;
end;

v(1) = 2;
if v(1) == 1,
  disp('the value is 1');
elseif v(1) == 2,
  disp('the value is 2');
else
  disp('the value is not 1 or 2');
end;

## Functions

function y = squareThisNumber(x)
  y = x^2;

function [y1,y2] = squareAndCubeThisNUmber(x)
  y1 = x^2;
  y2 = x^3;

### costFunctionJ 

X = [1 1; 1 2; 1 3]
y = [1; 2; 3]
theta = [0; 1]

function J = costFunctionJ(X, y, theta)
  m = size(X,1);
  pred = X*theta;
  sqrErrors = (pred-y).^2;
  
  J = 1/(2*m) * sum(sqrErrors);

## Vectorization!

