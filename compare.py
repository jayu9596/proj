import os

##edited
# os.system("rm compare-Kmeans-prec.txt")
# os.system("rm compare-Kmeans-mprec.txt")
os.system("touch compare-Kmeansplusplus-prec.txt")
os.system("touch compare-Kmeansplusplus-mprec.txt")
##edited
for i in range(0,50):
	os.system("python3 eval.py")