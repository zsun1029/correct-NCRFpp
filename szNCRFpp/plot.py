# -*- coding: utf-8 -*-
# @Author: sz
# @Date:   2018-10-9
import matplotlib.pyplot as plt 
import numpy as np

def value_list(dev_all_list, label_num):
    acc_list, p_list, r_list, f_list = [], [], [], []
    for dev_all in dev_all_list:
        values = dev_all.split(';')[1].split(',') 
        label1 = values[0].split(':')[1]
        label2 = values[1].split(':')[1]
        label3 = values[2].split(':')[1]
        label4 = values[3].split(':')[1]
        acc_list.append(float(label1)) 
        p_list.append(float(label2))
        r_list.append(float(label3))
        f_list.append(float(label4))  
    return acc_list, p_list, r_list, f_list

def draw_plot(x_val, y_list, y_name_list, ii=0): 
    dev_best_index = [28,28,28,28]
    for i in range(len(y_list)):
        plt.plot(x_val, y_list[i], marker='*', label=y_name_list[i]) 
        if ii == 0: 
            y_max = max(y_list[i])
            y_max_index = y_list[i].index(y_max)
        else:
            y_max_index = dev_best_index[i]
            y_max = y_list[i][y_max_index]
        show_max='['+str(y_max_index)+', '+str(y_max)+']'
        plt.annotate(show_max, xytext=(y_max_index+1.5, y_max+0.006),xy=(y_max_index+0.1, y_max+0.0005),arrowprops = dict(facecolor = "r", headlength = 10, headwidth = 5, width = 2))
        
    plt.legend()  # 让图例生效 
    plt.xticks(x_val, x_val, rotation=45)
    # ynames = np.arange(0.8, 1.1, 0.005) 
    # plt.yticks(ynames, ynames, rotation=45)
    plt.margins(0.1)
    plt.subplots_adjust(bottom=0.1)
    plt.xlabel(u"epoch") #X轴标签
    plt.ylabel("value") #Y轴标签
    plt.title(u"dev/test") #标题 
    plt.show()



# Dev: time: 27.02s, speed: 1641.12st/s; acc: 0.9968, p: 0.9477, r: 0.9590, f: 0.9533
filename = "sz_2.log"
lines_list = open(filename, 'r').readlines()
lines_list = [line for line in lines_list if line[0]!='E' ]
epoch2 = len(lines_list)
print(epoch2)
# dev test 两个图
assert epoch2 % 2 == 0
dev_all_list = [lines_list[2*i] for i in range(int(len(lines_list)/2))]
test_all_list = [lines_list[2*i+1] for i in range(int(len(lines_list)/2))]
list_ = [dev_all_list, test_all_list]
for i in range(len(list_)):
    a_all_list = list_[i]
    acc_list, p_list, r_list, f_list = value_list(a_all_list, 4) 
    y_list = acc_list, p_list, r_list, f_list
    y_name_list = ["acc", "p", "r", "f"] 
    draw_plot(range(len(a_all_list)), y_list, y_name_list, i)
