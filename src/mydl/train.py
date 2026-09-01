def train(net, train_iter, loss, optimizer, epochs):
        #训练核心部分
    for epoch in range(epochs):
        total_loss = 0
        total_num = 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            total_loss += l.item() * y.numel()
            total_num += y.numel()
        print(f'epoch {epoch + 1}, loss {total_loss / total_num:.4f}')
        # 我觉得这次的改动是值得被品鉴的，它涉及到缩进导致的循环问题，就是这个计数到底是该在哪一个循环里面？
            