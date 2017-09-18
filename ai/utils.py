def difference(n):
   return abs(reg.predict([input[n]]) - output[n])

def avg_diff(n):
   sum = 0
   for i in range(n):
     sum = sum + difference(n)
   return sum /(n*1.0)
