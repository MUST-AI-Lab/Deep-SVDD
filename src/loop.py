

n_epochs = 0
for i in range(1000):
    n_epochs += 1
    if n_epochs % 100 == 0:
        print('../log/mnist_test/100_AUC.txt')
