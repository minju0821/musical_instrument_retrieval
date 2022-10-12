import sys, os
sys.path.append(os.path.abspath('..'))

from deep_cnn import ConvNet_eval
model = ConvNet_eval(out_classes = 953)

print("1")


