#coding = utf-8
import sys
n = int(sys.stdin.readline().replace("/n",""))
terms = sys.stdin.readline().replace("/n","").split(" ")
array = [int(terms[i]) for i in range(n)]
judge_array = [0 for i in range(n)]
min_x = min(array)
for i in range(array.__len__()):
    array[i] = array[i]-min_x
min_x = 10000
for i in range(array.__len__()):
    if array[i] != 0 and array[i] < min_x:
        min_x = array[i]
for i in range(array.__len__()):
    array[i] = array[i] / min_x
for i in array:
    if i > n-1:
        print "Impossible"
        sys.exit()
    else:
        judge_array[i] = 1
for i in judge_array:
    if i != 1:
        print "Impossible"
        sys.exit()
print "Possible"