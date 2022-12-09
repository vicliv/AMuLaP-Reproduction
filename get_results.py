from os import listdir
from os.path import isfile, join
import collections
import numpy as np
import decimal

path = "outputs_large/logging/cola/16-"

total = []
for i, value in enumerate([13, 21, 42, 87, 100]):      
    mypath = path + str(value)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    d = {}
    for file in onlyfiles:
        if file[:3] == "no_" or 'pbs8' in file:
            continue
        s = file.split('_')[-3][4:]
        k = int(s)
        d[k] = file
        if i == 0:
            total.append([])
        
    od = collections.OrderedDict(sorted(d.items()))

    for k, file in enumerate(od.values()):
        with open(mypath+'/'+file, 'r') as f:
            line = f.readlines()
            print(file)
            try:
                v = float(line[-3].split(':')[-1][:-2])
            except:
                continue
            total[k].append(v)

for scores in total:
    a = np.array(scores)
    value = decimal.Decimal("12.34567")
    b = "{:.1f} ± {:.1f}".format(round(a.mean()*100, 1),round(a.std()*100, 1))
    print(b)

x = []
y = []
for scores in total:
    a = np.array(scores)
    value = decimal.Decimal("12.34567")
    b = round(a.mean()*100, 1)
    x.append(b)
    c = round(a.std()*100, 1)
    y.append(c)

print(x)
print(y)