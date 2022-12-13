from os import listdir
from os.path import isfile, join
import collections
import numpy as np
import decimal
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Get the results from amulap using the outputs path")
    parser.add_argument(
        "--path",
        type=str,
        default="outputs/logging/sst2",
        help="The path where the results are found",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        default=False,
        help="Whether the results used finetune."
    )
    parser.add_argument(
        "--dedup", 
        action="store_true",
        default=False,
        help="Whether to dedup label tokens."
    )
    parser.add_argument(
        "--shot_num", 
        type=int, 
        default=16, 
        help="The number of shots to used for training."    
    )
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    path = args.path + '/' + str(args.shot_num) + '-'

    total1 = []
    total2 = []
    for i, value in enumerate([13, 21, 42, 87, 100]):      
        mypath = path + str(value)
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        d = {}
        for file in onlyfiles:
            if file[:3] == "tra" and not args.finetune:
                continue
            if file[:3] == "no_" and args.finetune:
                continue
            if args.dedup:
                s = file.split('_')[-3][4:]
            else:
                s = file.split('_')[-2][4:]
            k = int(s)
            d[k] = file
            if i == 0:
                total1.append([])
                total2.append([])
            
        od = collections.OrderedDict(sorted(d.items()))

        for k, file in enumerate(od.values()):
            with open(mypath+'/'+file, 'r') as f:
                line = f.readlines()
                try:
                    num_line = -1
                    l = line[num_line]
                    while l == '\n':
                        num_line -= 1
                        l = line[num_line]
                    v1 = float(l.split(':')[-1][:-2])
                    l = line[num_line-1]
                    v2 = float(l.split(':')[-1][:-2])
                except:
                    continue
                total1[k].append(v1)
                total2[k].append(v2)

    # for scores in total1:
    #     a1 = np.array(scores)
    #     value = decimal.Decimal("12.34567")
    #     b1 = "{:.1f} ± {:.1f}".format(round(a.mean()*100, 1),round(a.std()*100, 1))
    
    # for scores in total2:
    #     a2 = np.array(scores)
    #     value = decimal.Decimal("12.34567")
    #     b2 = "{:.1f} ± {:.1f}".format(round(a.mean()*100, 1),round(a.std()*100, 1))

    x1 = []
    y1 = []
    for scores in total1:
        a = np.array(scores)
        value = decimal.Decimal("12.34567")
        b = round(a.mean()*100, 1)
        x1.append(b)
        c = round(a.std()*100, 1)
        y1.append(c)
    
    x2 = []
    y2 = []
    for scores in total2:
        a = np.array(scores)
        value = decimal.Decimal("12.34567")
        b = round(a.mean()*100, 1)
        x2.append(b)
        c = round(a.std()*100, 1)
        y2.append(c)

    print("Test mean accuracy: " + str(x1))
    print("Test std accuracy: " + str(y1))
    
    print("Validation mean accuracy: " + str(x2))
    print("Validation std accuracy: " + str(y2))