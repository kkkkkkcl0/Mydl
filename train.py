def train(net, train_iter, loss, optimier, epochs):
    num_loss = 0
    num_y = 0
    #训练核心部分
    for epoch in epochs:
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimier.zero_grad()
            l.backward()
            optimier.step()
            # test1
            