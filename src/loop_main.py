import os

f_get_para = open('../log/mnist_test/get_param.txt', 'w')
f_get_para.write("class 1: update focal_parameter form 1 to 2   \r\n \r\n \r\n")
f_get_para.close()

f_get_para = open('../log/mnist_test/100_AUC.txt', 'w')
f_get_para.write("class 1: update focal_parameter form 1 to 2   \r\n \r\n \r\n")
f_get_para.close()

focal_parameter = 1

for i in range(10):
    f_get_para = open('../log/mnist_test/get_param.txt', 'a')
    f_get_para.write("\r\nSet focal_parameter to %.2f. ++++++++++ \r\n" % focal_parameter)
    f_get_para.close()

    f_get_para = open('../log/mnist_test/100_AUC.txt', 'a')
    f_get_para.write("\r\nSet focal_parameter to %.2f. ++++++++++ \r\n" % focal_parameter)
    f_get_para.close()

    os.system("python main.py --focal_parameter %.2f" % focal_parameter)
    focal_parameter = focal_parameter + 0.1

