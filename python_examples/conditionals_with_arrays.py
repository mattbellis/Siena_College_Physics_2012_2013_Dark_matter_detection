import numpy as np

# Let me create an array with 20 random numbers between 0 and 10
x = 10*np.random.random(20)

print "Original x-array"
print x

# Suppose I want to create a new array where I find all the numbers in this
# x array that are below 5. Normally, you would loop over all the numbers
# and check their value. But with numpy arrays, there is an easier way. 
# You can actually use that conditional as indexing!

xsubset = x[x<5]

print "\nOriginal x-array, but only the values < 5"
print xsubset

# But what if you want a slice of the numbers? Say, between 4 and 6? 
# You can create an index for each conditional statement and then
# multiply them! 
index0 = x>4
index1 = x<6
index = index0*index1

xslice = x[index]

print "\nOriginal x-array, but only the values where x>4 and x<6"
print xslice

print "\n"

