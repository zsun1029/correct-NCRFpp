f_name = 'officialdoc/haidian_news_sz_train.txt'

with open(f_name, 'r') as f:
	line_str = f.read() 
	# print(line_str)
	line_str_list = line_str.replace('。\tO\n', '。\tO|\n').replace('，\tO\n', '，\tO|\n').replace('！\tO\n', '！\tO|\n').split('|')
	print(len(line_str_list))

	line_str_list_withwrongword = [s for s in line_str_list if 'S-1' in s]
	print(len(line_str_list_withwrongword))
good_filename = 'officialdoc/good_examples.bmes'
with open(good_filename, 'w') as f:
	f.writelines(line_str_list_withwrongword)

