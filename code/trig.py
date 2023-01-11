'''
imran iftikar
11/19/21
created 6 main trig functions from scratch by approximating using the taylor series
Also added degrees to rad, rad to degrees, and simplififcation of rads and degrees
All arguments that are in radians are in PI radians, so there is no need to specificy pi in the arg
log.txt separate
known bugs:
    does not work as intented if user puts arg that should in PI radians in radians as a whole
'''


######### Dependencies

######### GLOBALS

pi = 3.1415926535

######### Functions

deg_to_pi_rad = lambda deg: deg/180

pi_rad_to_deg = lambda pirad: pirad*180

simplest_pi_rad = lambda pirad: pirad%2
    
simplest_deg = lambda deg: deg%360

pirad_to_rad = lambda pirad: pirad*pi

rad_to_pirad = lambda rad: rad/pi

def factorial(number):

    if isinstance(number, int) and number >= 1:
        initial_product = 1
        for i in range(1, number+1):
            initial_product = initial_product*i
        
        return initial_product

    elif number == 0: return 1

    else:
        raise ValueError("Number cannot be factorial-ed")

def sin(x):

    if x >= 2:
        x = x%2

    x = x*pi

    h = x 

    for i in range(3, 102, 2):

        if i%4 == 1:

            h = h + ((x**i)/factorial(i))

        else:

            h = h - ((x**i)/factorial(i))

    return h

cos = lambda x: sin((1/2)-x) # cos(x) = sin(90-x) = sin(pi/2 radians - x)

tan = lambda x: sin(x)/cos(x)

csc = lambda x: 1/sin(x)

sec = lambda x: 1/cos(x)

cot = lambda x: cos(x)/sin(x)

def arcsin(x, out_pirad = True):

    if not(-1<x<1):
        raise TypeError(f"Arguemt x: {x} must be between -1 and 1")
    
    else:
        
        h = 0
        for n in range(0,100):
            t1 = factorial(2*n) / (((2**n) * (factorial(n) ))**2)
            t2 = (x**(2*n+1)) / (2*n+1)
            t3 = t1*t2
            h = h+t3
        
        if out_pirad == True:
            return h/pi
        
        else: return h

arccos = lambda x: (pi/2) - arcsin(x)

def arctan(x, bool_): # margin of error of abt 0.4 when passing large numbers. unsure why, will try to fix. although not *too* bad performance
    x = ((x)/((x**2+1)**(1/2))) # x over sqrt x*2 + 1
    return arcsin(x, bool_) # calls asin