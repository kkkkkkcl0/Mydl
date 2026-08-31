def train(net, train_iter, loss, optimizer, epochs):
    total_loss = 0
    total_num = 0
    #训练核心部分
    for epoch in range(epochs):
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        total_loss += l.item() * y.numel()
        total_num += y.numel()
        print(f'epoch {epoch + 1}, loss {total_loss / total_num:.4f}')
            