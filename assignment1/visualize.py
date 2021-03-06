# Christoph Conrads (315565)
# Antje Relitz (327289)
# Benjamin Pietrowicz (332542)
# Mitja Richter (324680)

import classifiers
import numpy as np

# Visualize a classification boundary
#  input:
#   f: decision function
def visualize(f):
	# create a grid between -10 and 10 spaced by 1
	R = range(-10,11)

	# create an ascii art of the decision boundary on this range
	D = "\n".join(["".join(['1' if f([j,i]) else '0' for j in R]) for i in R[::-1]])
	print(D)


# -----------------------------------------------------------------------
# Create some decision boundaries (function+parameters) for visualization
# -----------------------------------------------------------------------
models = [
	{'f': classifiers.A, 'w': (1,)},
	{'f': classifiers.B, 'w': (np.array([1,1]),2.5)},
	{'f': classifiers.C, 'w': (np.array([0,2]),5,2)},
	{'f': classifiers.D, 'w': (np.array([[2,1],[1,1]]),np.array([0,0]),-16)},
	{'f': classifiers.ring, 'w': (np.array([0,2]),6,2)},
	{'f': classifiers.rect, 'w': (np.array([1,2]),np.array([3,4]))}
]


# -----------------------------------------------------------------------
# Visualize the decision function for some chosen functions and parameters
# -----------------------------------------------------------------------
for m in models:
	print('%32s | param=%s'%(m['f'],m['w']))
	visualize(lambda x: m['f'](x,m['w']))
