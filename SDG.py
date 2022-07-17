import matplotlib.pyplot as plt

# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0


# define the model linear model y = w*x
def forward(x):
    return x * w


# define the cost function MSE
def loss(x, y):
    y_pred = forward(x)
    loss = (y_pred - y) ** 2
    return loss


# define the gradient function  gd
def gradient(x, y):
    grad= 2 * x * (x * w - y)
    return grad


epoch_list = []
cost_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l=loss(x,y)
        grad_val=gradient(x,y)
        w-=0.01*grad_val
    print('epoch:', epoch, 'w=', w, 'loss=', l)
    epoch_list.append(epoch)
    cost_list.append(l)

print('predict (after training)', 4, forward(4))
plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()