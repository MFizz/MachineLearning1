# Christoph Conrads (315565)
# Antje Relitz (327289)
# Benjamin Pietrowicz (332542)
# Mitja Richter (324680)

import numpy as N

# classifier A
def A(x,(a,)):
	return ( a**x[0] - x[1] ) >= 0


# classifier B
def B(x,(w,b)):
	return ( N.dot(w.T, x) + b ) >= 0


# classifier C
def C(x,(c,r,p)):
	assert(p == 1 or p == 2)
	assert(r > 0)
	return (N.abs(x-c)**p).sum() <= r**p


# classifier D
def D(x,(A,W,b)):
	x = N.array(x)
	return ( N.dot(x.T, N.dot(A,x)) + N.dot(W.T, x) + b ) >= 0


# classifier "ring"
#  x: data point
#  c: center of the ring
#  r: radius of the ring
#  w: width of the ring
def ring(x,(c,r,w)):
	assert(r > 0)
	assert(w < r)
	return C(x,(c,r+w,2)) and (not C(x,(c,r-w,2)))


# classifier "rectangle"
#  x: data point
#  c: center of the rectangle
#  s: size of the rectangle
def rect(x,(c,s)):
	assert( N.all(s > 0) )

	# Construct the rectangle using four planes
	w_x = N.array([1, 0]).reshape(2, 1)
	w_y = N.array([0, 1]).reshape(2, 1)

	# For each valid points it holds:
	# x_i - c_i >= -s_i/2 <=> +(x_i - c_i) + s_i/2 >= 0
	# x_i - c_i <= +s_i/2 <=> -(x_i - c_i) + s_i/2 >= 0

	return \
		B(x-c,(w_x,s[0]/2.0)) and B(x-c,(-w_x,s[0]/2.0)) and \
		B(x-c,(w_y,s[1]/2.0)) and B(x-c,(-w_y,s[1]/2.0))