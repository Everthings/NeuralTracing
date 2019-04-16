from subprocess import call
import pickle
from sys import argv
new_file = int(argv[1])
version = argv[2]
isbn = int(argv[3])
print(version)
if new_file > 0:
	with open('log.dict','wb') as f:
		accuracy = dict()
		pickle.dump(accuracy,f)

for angle_xy in range(30,181,30):
	for i in range(1):
		print(angle_xy)
		with open('log.dict','rb') as f:
			accuracy  = pickle.load(f)
		if ((angle_xy) not in accuracy) or (accuracy[(angle_xy)] < 0.0):
			if isbn > 0:
				call(["python","fbbn_train_script.py",version,str(angle_xy),"1","1","1e-3","1.0"])
			else:
				call(["python","fbcn_train_script.py",version,str(angle_xy),"1","1","1e-3","1.0"])
		else:
			break
