# Some code we used to make some of our graphs if you want to check it out...
# The commented outcode below are outputs from gridsearch.py

# Sorry, the variable names are pretty stupid...


'''
eta: 1, gamma: 0.6, epsilon: 0.001.
[1, 2, 3, 0, 0, 2, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 2, 1, 1, 1, 1, 0, 5, 6, 1, 8, 2, 2, 2, 2, 2, 3, 0, 3, 6, 0, 0, 12, 1, 7, 0, 0, 1, 1, 1, 2, 1, 0, 0, 2, 0, 10, 28, 0, 22, 1, 8, 81, 27, 0, 1, 0, 54, 40, 1, 0, 17, 0, 1, 18, 52, 0, 1, 1, 0, 0, 1, 1, 7, 0, 1, 0, 2, 0, 1, 2, 1, 1, 2, 6, 0, 1, 1, 1, 2, 1, 6, 13, 0, 0, 0, 0, 11, 26, 4, 0, 6, 25, 1, 2, 1, 0, 28, 0, 2, 57, 54, 0, 30, 1, 10, 24, 2, 38, 25, 14, 2, 12, 10, 1, 0, 1, 25, 25, 1, 15, 0, 58, 10, 0, 23, 0, 1, 3, 73, 5, 3, 0, 1, 0, 32, 2, 11, 8, 1, 11, 0, 48, 17, 23, 13, 17, 1, 1, 0, 0, 0, 0, 1, 2, 2, 5, 2, 1, 1, 2, 0, 0, 19, 0, 2, 2, 60, 1, 1, 9, 32, 1, 5, 0, 0, 6, 13, 9, 1, 0, 2, 0, 1, 17, 1, 3, 2, 0, 12, 1, 1, 5, 0, 1, 0, 2, 1, 1, 11, 1, 0, 17, 9, 1, 1, 63, 0, 11, 1, 11, 0, 10, 7, 10, 30, 1, 0, 0, 17, 0, 19, 1, 3, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 3, 0, 2, 44, 9, 5, 0, 1, 14, 1, 27, 4, 0, 60, 18, 1, 8, 19, 0, 0, 30, 14, 63, 2, 1, 6, 8, 0, 1, 2, 8, 1, 2, 3, 1, 0]
Average score over all epochs: 7.17
Max score over all epochs: 81


eta: 1, gamma: 0.7, epsilon: 0.001.
[0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 0, 0, 1, 3, 0, 1, 0, 2, 0, 0, 1, 0, 1, 0, 3, 1, 1, 0, 3, 0, 2, 0, 1, 1, 1, 12, 1, 3, 1, 4, 5, 6, 0, 1, 4, 2, 7, 3, 0, 0, 2, 1, 0, 3, 1, 0, 0, 2, 3, 1, 4, 6, 1, 1, 4, 1, 1, 8, 3, 1, 0, 1, 0, 1, 2, 10, 0, 4, 8, 2, 1, 0, 19, 0, 1, 6, 1, 0, 0, 12, 1, 1, 12, 1, 1, 1, 27, 1, 3, 17, 1, 30, 0, 13, 4, 14, 59, 1, 0, 1, 1, 0, 41, 31, 1, 17, 1, 0, 0, 22, 1, 40, 21, 1, 7, 2, 7, 1, 0, 1, 1, 2, 1, 7, 2, 1, 1, 0, 42, 1, 30, 2, 11, 32, 1, 17, 1, 23, 2, 1, 8, 9, 0, 1, 1, 1, 1, 23, 3, 0, 2, 2, 1, 1, 1, 1, 1, 18, 1, 0, 0, 9, 5, 1, 1, 10, 5, 1, 4, 1, 3, 16, 0, 1, 1, 2, 1, 7, 1, 1, 3, 3, 17, 12, 1, 1, 1, 0, 12, 1, 2, 0, 7, 3, 2, 1, 18, 1, 1, 1, 0, 1, 2, 22, 3, 1, 1, 2, 4, 1, 15, 0, 21, 33, 0, 0, 0, 3, 1, 3, 33, 3, 1, 1, 1, 4, 1, 0, 1, 1, 0, 30, 0, 0, 0, 2, 19, 1, 28, 3, 16, 20, 1, 1, 1, 0, 1, 1, 5, 1, 1, 0, 0, 2, 0, 0, 23, 2, 4, 1, 1, 22, 1, 1, 15, 0, 1, 1, 14, 24, 29, 1, 5, 21]
Average score over all epochs: 4.84
Max score over all epochs: 59


eta: 1, gamma: 0.8, epsilon: 0.001.
[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3, 1, 3, 0, 1, 0, 1, 1, 1, 10, 1, 4, 3, 9, 3, 1, 3, 3, 0, 0, 8, 1, 0, 12, 0, 1, 1, 0, 1, 17, 1, 20, 1, 3, 1, 2, 0, 0, 1, 1, 1, 0, 0, 5, 1, 0, 1, 5, 10, 26, 1, 6, 1, 26, 2, 1, 8, 0, 0, 0, 1, 0, 1, 3, 0, 4, 94, 22, 0, 1, 3, 0, 1, 1, 3, 14, 1, 1, 6, 1, 0, 20, 1, 0, 15, 0, 14, 13, 1, 2, 0, 0, 0, 9, 0, 0, 2, 6, 2, 0, 3, 3, 4, 4, 0, 0, 9, 0, 5, 5, 0, 0, 1, 1, 0, 3, 1, 1, 1, 0, 0, 9, 1, 0, 4, 1, 0, 0, 0, 1, 37, 0, 7, 5, 0, 9, 27, 9, 0, 1, 0, 15, 1, 1, 1, 4, 11, 3, 0, 0, 4, 7, 1, 1, 0, 1, 1, 0, 1, 15, 8, 1, 0, 2, 9, 2, 3, 1, 12, 23, 1, 1, 5, 1, 1, 6, 2, 1, 1, 18, 0, 1, 1, 0, 18, 2, 30, 7, 11, 1, 4, 33, 20, 0, 2, 3, 2, 1, 90, 20, 16, 0, 1, 3, 1, 1, 40, 1, 29, 23, 19, 1, 1, 1, 0, 12, 1, 15, 1, 0, 4, 1, 16, 28, 25, 0, 2, 38, 20, 27, 19, 15, 8, 1, 1, 1, 29, 2, 1, 13, 2, 1, 0, 6, 23, 44, 18, 2, 2, 37, 29, 1, 6, 7, 0, 0, 21, 1, 1, 1, 10, 2, 1, 1, 17, 1, 1, 6]
Average score over all epochs: 5.65
Max score over all epochs: 94


eta: 1, gamma: 0.9, epsilon: 0.001.
[0, 0, 0, 0, 0, 1, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 3, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 1, 2, 1, 1, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 2, 1, 0, 1, 1, 3, 2, 0, 2, 1, 0, 1, 2, 0, 0, 1, 1, 3, 3, 1, 0, 1, 15, 3, 0, 1, 3, 9, 7, 1, 22, 38, 22, 37, 0, 1, 2, 0, 1, 7, 0, 53, 1, 13, 1, 1, 18, 1, 2, 1, 1, 4, 0, 0, 24, 9, 36, 2, 18, 8, 38, 1, 1, 4, 0, 1, 1, 2, 22, 0, 54, 1, 21, 1, 1, 1, 2, 25, 7, 0, 9, 66, 1, 13, 67, 1, 3, 1, 2, 1, 1, 1, 49, 0, 0, 0, 3, 2, 49, 2, 1, 12, 18, 23, 1, 1, 1, 1, 1, 25, 15, 0, 3, 8, 1, 1, 15, 7, 43, 4, 0, 32, 1, 34, 2, 30, 2, 0, 26, 0, 1, 118, 3, 1, 2, 1, 110, 6, 2, 3, 1, 25, 0, 16, 0, 1, 2, 4, 2, 2, 13, 15, 1, 0, 1, 37, 64, 15, 2, 13, 77, 1, 1, 1, 1, 0, 52, 1, 98, 0, 1, 2, 23, 2, 19, 36, 5, 0, 45, 52, 40, 1, 0, 1, 57, 2, 2, 1, 4, 1, 1, 4, 1, 1, 1, 1, 10, 1, 0, 1, 15, 4, 2, 1, 1, 0, 66, 0, 5, 38, 12, 0, 1, 1, 55, 1, 27, 0, 36, 1, 22, 20, 1, 3, 3, 1, 3, 50, 5, 1, 0, 1, 70, 1, 2, 13, 1, 1, 38, 1, 78, 2, 6, 31, 1, 26, 17, 1, 3, 1, 1, 2, 48, 87, 17, 1, 1]
Average score over all epochs: 10.063333333333333
Max score over all epochs: 118


eta: 1, gamma: 1, epsilon: 0.001.
[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 4, 0, 0, 0, 2, 1, 2, 1, 0, 2, 1, 5, 0, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 3, 1, 2, 1, 0, 0, 4, 1, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 2, 0, 4, 3, 1, 1, 4, 3, 1, 2, 1, 1, 21, 8, 1, 0, 11, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 19, 1, 2, 6, 0, 3, 22, 6, 8, 20, 0, 1, 54, 98, 13, 0, 4, 1, 108, 1, 5, 1, 6, 0, 0, 0, 0, 1, 1, 5, 107, 6, 1, 128, 10, 18, 2, 23, 6, 1, 15, 2, 49, 62, 36, 82, 2, 1, 0, 0, 1, 0, 2, 0, 1, 14, 45, 2, 0, 1, 1, 34, 6, 60, 100, 95, 28, 0, 2, 1, 16, 2, 84, 37, 1, 1, 95, 11, 0, 32, 0, 0, 26, 0, 2, 36, 19, 1, 3, 2, 1, 54, 31, 1, 0, 0, 1, 150, 8, 1, 16, 39, 2, 1, 1, 1, 12, 0, 0, 88, 5, 54, 1, 54, 5, 1, 14, 26, 68, 51, 1, 17, 10, 12, 37, 1, 2, 7, 0, 64, 14, 11, 3, 5, 1, 2, 1, 10, 1, 8, 2, 105, 0, 140, 0, 178, 1, 1, 26, 1, 0, 17, 1, 49, 1, 4, 8, 0, 24, 4, 0, 1, 3, 34, 2, 162, 1, 1, 27, 49, 275, 11, 1, 0, 6, 44, 2, 1, 0, 1, 1, 20, 1, 1, 83, 18, 1, 1, 1, 3, 1, 1, 17, 1, 62, 0, 1, 10, 150, 19, 1, 1, 2, 16, 1, 25, 3, 1, 1, 41, 10, 73, 11, 0, 0, 1, 105, 1, 108, 1]
Average score over all epochs: 15.673333333333334
Max score over all epochs: 275
'''
six = [1, 2, 3, 0, 0, 2, 0, 1, 0, 1, 1, 1, 0, 1, 1, 2, 1, 0, 2, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 2, 1, 1, 1, 1, 0, 5, 6, 1, 8, 2, 2, 2, 2, 2, 3, 0, 3, 6, 0, 0, 12, 1, 7, 0, 0, 1, 1, 1, 2, 1, 0, 0, 2, 0, 10, 28, 0, 22, 1, 8, 81, 27, 0, 1, 0, 54, 40, 1, 0, 17, 0, 1, 18, 52, 0, 1, 1, 0, 0, 1, 1, 7, 0, 1, 0, 2, 0, 1, 2, 1, 1, 2, 6, 0, 1, 1, 1, 2, 1, 6, 13, 0, 0, 0, 0, 11, 26, 4, 0, 6, 25, 1, 2, 1, 0, 28, 0, 2, 57, 54, 0, 30, 1, 10, 24, 2, 38, 25, 14, 2, 12, 10, 1, 0, 1, 25, 25, 1, 15, 0, 58, 10, 0, 23, 0, 1, 3, 73, 5, 3, 0, 1, 0, 32, 2, 11, 8, 1, 11, 0, 48, 17, 23, 13, 17, 1, 1, 0, 0, 0, 0, 1, 2, 2, 5, 2, 1, 1, 2, 0, 0, 19, 0, 2, 2, 60, 1, 1, 9, 32, 1, 5, 0, 0, 6, 13, 9, 1, 0, 2, 0, 1, 17, 1, 3, 2, 0, 12, 1, 1, 5, 0, 1, 0, 2, 1, 1, 11, 1, 0, 17, 9, 1, 1, 63, 0, 11, 1, 11, 0, 10, 7, 10, 30, 1, 0, 0, 17, 0, 19, 1, 3, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 3, 0, 2, 44, 9, 5, 0, 1, 14, 1, 27, 4, 0, 60, 18, 1, 8, 19, 0, 0, 30, 14, 63, 2, 1, 6, 8, 0, 1, 2, 8, 1, 2, 3, 1, 0]
sixx = (0.6, 7.17, 81)


seven = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 0, 0, 1, 3, 0, 1, 0, 2, 0, 0, 1, 0, 1, 0, 3, 1, 1, 0, 3, 0, 2, 0, 1, 1, 1, 12, 1, 3, 1, 4, 5, 6, 0, 1, 4, 2, 7, 3, 0, 0, 2, 1, 0, 3, 1, 0, 0, 2, 3, 1, 4, 6, 1, 1, 4, 1, 1, 8, 3, 1, 0, 1, 0, 1, 2, 10, 0, 4, 8, 2, 1, 0, 19, 0, 1, 6, 1, 0, 0, 12, 1, 1, 12, 1, 1, 1, 27, 1, 3, 17, 1, 30, 0, 13, 4, 14, 59, 1, 0, 1, 1, 0, 41, 31, 1, 17, 1, 0, 0, 22, 1, 40, 21, 1, 7, 2, 7, 1, 0, 1, 1, 2, 1, 7, 2, 1, 1, 0, 42, 1, 30, 2, 11, 32, 1, 17, 1, 23, 2, 1, 8, 9, 0, 1, 1, 1, 1, 23, 3, 0, 2, 2, 1, 1, 1, 1, 1, 18, 1, 0, 0, 9, 5, 1, 1, 10, 5, 1, 4, 1, 3, 16, 0, 1, 1, 2, 1, 7, 1, 1, 3, 3, 17, 12, 1, 1, 1, 0, 12, 1, 2, 0, 7, 3, 2, 1, 18, 1, 1, 1, 0, 1, 2, 22, 3, 1, 1, 2, 4, 1, 15, 0, 21, 33, 0, 0, 0, 3, 1, 3, 33, 3, 1, 1, 1, 4, 1, 0, 1, 1, 0, 30, 0, 0, 0, 2, 19, 1, 28, 3, 16, 20, 1, 1, 1, 0, 1, 1, 5, 1, 1, 0, 0, 2, 0, 0, 23, 2, 4, 1, 1, 22, 1, 1, 15, 0, 1, 1, 14, 24, 29, 1, 5, 21]
sevenn = (0.7, 4.84, 59)


eight=[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 3, 1, 3, 0, 1, 0, 1, 1, 1, 10, 1, 4, 3, 9, 3, 1, 3, 3, 0, 0, 8, 1, 0, 12, 0, 1, 1, 0, 1, 17, 1, 20, 1, 3, 1, 2, 0, 0, 1, 1, 1, 0, 0, 5, 1, 0, 1, 5, 10, 26, 1, 6, 1, 26, 2, 1, 8, 0, 0, 0, 1, 0, 1, 3, 0, 4, 94, 22, 0, 1, 3, 0, 1, 1, 3, 14, 1, 1, 6, 1, 0, 20, 1, 0, 15, 0, 14, 13, 1, 2, 0, 0, 0, 9, 0, 0, 2, 6, 2, 0, 3, 3, 4, 4, 0, 0, 9, 0, 5, 5, 0, 0, 1, 1, 0, 3, 1, 1, 1, 0, 0, 9, 1, 0, 4, 1, 0, 0, 0, 1, 37, 0, 7, 5, 0, 9, 27, 9, 0, 1, 0, 15, 1, 1, 1, 4, 11, 3, 0, 0, 4, 7, 1, 1, 0, 1, 1, 0, 1, 15, 8, 1, 0, 2, 9, 2, 3, 1, 12, 23, 1, 1, 5, 1, 1, 6, 2, 1, 1, 18, 0, 1, 1, 0, 18, 2, 30, 7, 11, 1, 4, 33, 20, 0, 2, 3, 2, 1, 90, 20, 16, 0, 1, 3, 1, 1, 40, 1, 29, 23, 19, 1, 1, 1, 0, 12, 1, 15, 1, 0, 4, 1, 16, 28, 25, 0, 2, 38, 20, 27, 19, 15, 8, 1, 1, 1, 29, 2, 1, 13, 2, 1, 0, 6, 23, 44, 18, 2, 2, 37, 29, 1, 6, 7, 0, 0, 21, 1, 1, 1, 10, 2, 1, 1, 17, 1, 1, 6]
eightt=(0.8, 5.65, 94)

nine=[0, 0, 0, 0, 0, 1, 1, 2, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 3, 1, 1, 1, 0, 2, 0, 1, 0, 1, 0, 1, 2, 1, 1, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 2, 1, 0, 1, 1, 3, 2, 0, 2, 1, 0, 1, 2, 0, 0, 1, 1, 3, 3, 1, 0, 1, 15, 3, 0, 1, 3, 9, 7, 1, 22, 38, 22, 37, 0, 1, 2, 0, 1, 7, 0, 53, 1, 13, 1, 1, 18, 1, 2, 1, 1, 4, 0, 0, 24, 9, 36, 2, 18, 8, 38, 1, 1, 4, 0, 1, 1, 2, 22, 0, 54, 1, 21, 1, 1, 1, 2, 25, 7, 0, 9, 66, 1, 13, 67, 1, 3, 1, 2, 1, 1, 1, 49, 0, 0, 0, 3, 2, 49, 2, 1, 12, 18, 23, 1, 1, 1, 1, 1, 25, 15, 0, 3, 8, 1, 1, 15, 7, 43, 4, 0, 32, 1, 34, 2, 30, 2, 0, 26, 0, 1, 118, 3, 1, 2, 1, 110, 6, 2, 3, 1, 25, 0, 16, 0, 1, 2, 4, 2, 2, 13, 15, 1, 0, 1, 37, 64, 15, 2, 13, 77, 1, 1, 1, 1, 0, 52, 1, 98, 0, 1, 2, 23, 2, 19, 36, 5, 0, 45, 52, 40, 1, 0, 1, 57, 2, 2, 1, 4, 1, 1, 4, 1, 1, 1, 1, 10, 1, 0, 1, 15, 4, 2, 1, 1, 0, 66, 0, 5, 38, 12, 0, 1, 1, 55, 1, 27, 0, 36, 1, 22, 20, 1, 3, 3, 1, 3, 50, 5, 1, 0, 1, 70, 1, 2, 13, 1, 1, 38, 1, 78, 2, 6, 31, 1, 26, 17, 1, 3, 1, 1, 2, 48, 87, 17, 1, 1]
ninee=(0.9, 10.063, 118)


ten =[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 4, 0, 0, 0, 2, 1, 2, 1, 0, 2, 1, 5, 0, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 3, 1, 2, 1, 0, 0, 4, 1, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 2, 0, 4, 3, 1, 1, 4, 3, 1, 2, 1, 1, 21, 8, 1, 0, 11, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 19, 1, 2, 6, 0, 3, 22, 6, 8, 20, 0, 1, 54, 98, 13, 0, 4, 1, 108, 1, 5, 1, 6, 0, 0, 0, 0, 1, 1, 5, 107, 6, 1, 128, 10, 18, 2, 23, 6, 1, 15, 2, 49, 62, 36, 82, 2, 1, 0, 0, 1, 0, 2, 0, 1, 14, 45, 2, 0, 1, 1, 34, 6, 60, 100, 95, 28, 0, 2, 1, 16, 2, 84, 37, 1, 1, 95, 11, 0, 32, 0, 0, 26, 0, 2, 36, 19, 1, 3, 2, 1, 54, 31, 1, 0, 0, 1, 150, 8, 1, 16, 39, 2, 1, 1, 1, 12, 0, 0, 88, 5, 54, 1, 54, 5, 1, 14, 26, 68, 51, 1, 17, 10, 12, 37, 1, 2, 7, 0, 64, 14, 11, 3, 5, 1, 2, 1, 10, 1, 8, 2, 105, 0, 140, 0, 178, 1, 1, 26, 1, 0, 17, 1, 49, 1, 4, 8, 0, 24, 4, 0, 1, 3, 34, 2, 162, 1, 1, 27, 49, 275, 11, 1, 0, 6, 44, 2, 1, 0, 1, 1, 20, 1, 1, 83, 18, 1, 1, 1, 3, 1, 1, 17, 1, 62, 0, 1, 10, 150, 19, 1, 1, 2, 16, 1, 25, 3, 1, 1, 41, 10, 73, 11, 0, 0, 1, 105, 1, 108, 1]
tenn = (1, 15.673, 275)

import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.plot(np.arange(len(six)), six, '.')
plt.plot(np.arange(len(six)), seven, 'r.')
plt.plot(np.arange(len(six)), eight, 'g.')
plt.plot(np.arange(len(six)), nine, 'm.')
plt.plot(np.arange(len(six)), ten, 'k.')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Varying Gamma over 300 Epochs. Eta = 1, Epsilon = 0.001.')
plt.legend(["Gamma = {}, Avg = {}, Max = {}.".format(sixx[0],sixx[1], sixx[2]),"Gamma = {}, Avg = {}, Max = {}.".format(sevenn[0],sevenn[1], sevenn[2]), "Gamma = {}, Avg = {}, Max = {}.".format(eightt[0],eightt[1], eightt[2]), "Gamma = {}, Avg = {}, Max = {}.".format(ninee[0],ninee[1], ninee[2]), "Gamma = {}, Avg = {}, Max = {}.".format(tenn[0],tenn[1], tenn[2])])
plt.show()

'''
eta: 1, gamma: 1, epsilon: 0.0025.
[0, 0, 0, 2, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 2, 1, 1, 3, 1, 2, 1, 2, 2, 3, 1, 0, 1, 12, 2, 0, 3, 2, 0, 2, 2, 2, 1, 1, 12, 21, 2, 2, 0, 0, 2, 0, 1, 0, 3, 1, 0, 2, 1, 24, 12, 0, 0, 0, 5, 10, 0, 0, 28, 20, 6, 0, 0, 3, 65, 0, 3, 13, 1, 14, 1, 1, 1, 1, 1, 1, 5, 3, 0, 1, 1, 0, 11, 1, 1, 2, 8, 5, 1, 12, 2, 13, 15, 11, 108, 1, 2, 3, 1, 8, 1, 0, 8, 3, 4, 10, 18, 11, 1, 1, 6, 1, 0, 6, 5, 0, 1, 1, 1, 16, 5, 1, 1, 10, 0, 1, 6, 3, 1, 1, 1, 10, 3, 0, 0, 1, 2, 0, 1, 0, 10, 11, 1, 20, 3, 6, 0, 1, 1, 2, 2, 0, 0, 1, 3, 8, 0, 1, 5, 3, 8, 6, 17, 0, 0, 11, 3, 0, 0, 1, 1, 2, 14, 5, 1, 5, 1, 33, 1, 11, 1, 1, 1, 1, 33, 21, 0, 0, 0, 1, 4, 0, 1, 1, 1, 0, 0, 1, 2, 6, 1, 18, 12, 21, 0, 9, 2, 28, 1, 10, 27, 29, 1, 19, 14, 3, 30, 26, 2, 1, 1, 6, 16, 13, 0, 1, 13, 8, 4, 36, 1, 8, 5, 2, 1, 1, 30, 19, 3, 58, 19, 20, 1, 3, 0, 5, 2, 1, 0, 6, 2, 7, 1, 30, 1, 17, 1, 1, 2, 18, 36, 7, 14, 21, 28, 2, 4, 1, 1]
Average score over all epochs: 5.64
Max score over all epochs: 108


eta: 1, gamma: 1, epsilon: 0.005.
[0, 0, 0, 1, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 2, 1, 2, 2, 0, 2, 1, 1, 0, 2, 2, 1, 4, 2, 1, 0, 3, 6, 1, 6, 1, 6, 1, 2, 1, 1, 1, 2, 0, 1, 0, 12, 1, 7, 3, 0, 0, 0, 1, 13, 0, 1, 1, 0, 6, 0, 1, 0, 23, 1, 0, 1, 15, 20, 2, 0, 1, 8, 19, 1, 3, 41, 0, 2, 43, 2, 1, 4, 97, 0, 9, 11, 0, 3, 17, 65, 0, 18, 64, 1, 1, 19, 2, 9, 10, 1, 16, 0, 17, 0, 0, 2, 1, 13, 40, 32, 44, 1, 1, 1, 9, 2, 8, 1, 26, 1, 1, 5, 2, 35, 1, 3, 10, 0, 3, 0, 1, 1, 0, 16, 1, 1, 1, 2, 46, 2, 1, 2, 16, 55, 8, 0, 1, 75, 14, 21, 15, 1, 14, 27, 45, 7, 7, 27, 12, 38, 1, 0, 2, 2, 24, 29, 1, 22, 1, 1, 15, 5, 1, 6, 24, 6, 0, 0, 1, 40, 10, 2, 10, 1, 1, 2, 11, 0, 0, 48, 111, 3, 1, 1, 20, 34, 86, 1, 3, 1, 6, 25, 1, 2, 65, 1, 0, 174, 36, 1, 15, 29, 3, 11, 24, 0, 20, 1, 1, 13, 1, 7, 31, 1, 2, 1, 1, 1, 30, 46, 34, 32, 1, 46, 12, 103, 0, 1, 1, 6, 32, 1, 1, 2, 26, 7, 66, 1, 0, 82, 20, 1, 1, 1, 0, 0, 5, 1, 41, 6, 1, 11, 5, 0, 5, 8, 30, 166, 2, 0, 126, 2, 31, 0, 8, 1, 0, 1, 1, 1, 22, 19, 1, 19, 30, 1, 36, 1, 0, 26, 1]
Average score over all epochs: 11.74
Max score over all epochs: 174


eta: 1, gamma: 1, epsilon: 0.0075.
[0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 1, 1, 1, 0, 1, 2, 0, 3, 1, 1, 1, 1, 2, 1, 1, 2, 1, 0, 1, 2, 0, 0, 2, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 1, 1, 1, 2, 4, 0, 2, 3, 1, 7, 2, 2, 2, 0, 0, 1, 4, 1, 0, 5, 4, 1, 9, 4, 8, 11, 1, 2, 21, 9, 1, 0, 0, 4, 0, 0, 8, 2, 12, 15, 6, 0, 1, 0, 1, 14, 0, 2, 1, 0, 3, 1, 1, 2, 30, 0, 11, 0, 0, 0, 4, 9, 0, 1, 9, 1, 2, 0, 1, 2, 3, 1, 0, 1, 1, 4, 2, 0, 3, 11, 3, 0, 0, 1, 0, 0, 4, 1, 2, 14, 3, 6, 2, 1, 3, 3, 1, 2, 8, 14, 6, 5, 4, 23, 4, 8, 3, 1, 4, 6, 1, 0, 0, 0, 3, 0, 1, 1, 2, 17, 59, 1, 0, 1, 0, 1, 0, 2, 1, 0, 1, 31, 19, 1, 1, 0, 1, 1, 1, 20, 2, 2, 2, 4, 1, 2, 6, 2, 1, 0, 24, 7, 0, 20, 2, 3, 0, 1, 3, 0, 1, 16, 0, 1, 18, 10, 3, 1, 6, 1, 0, 17, 22, 28, 3, 5, 1, 5, 2, 1, 0, 7, 10, 0, 8, 2, 7, 13, 3, 12, 1, 23, 4, 1, 2, 1, 2, 6, 35, 0, 1, 1, 1, 2, 5, 1, 1, 1, 3, 1, 11, 8, 17, 1, 1, 18, 48, 6, 2, 11, 5, 1, 18, 0, 3, 0, 0, 10, 15, 6, 6, 1, 21, 5, 1, 0, 0, 2, 2, 11, 3, 12, 5, 1, 13, 0, 1, 1, 3, 25, 0, 0, 0, 3, 2, 6]
Average score over all epochs: 4.346666666666667
Max score over all epochs: 59


eta: 1, gamma: 1, epsilon: 0.01.
[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 0, 1, 0, 3, 2, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 3, 0, 2, 1, 2, 0, 0, 1, 4, 5, 0, 1, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0, 6, 1, 4, 21, 0, 11, 3, 0, 1, 0, 1, 0, 0, 0, 0, 1, 11, 2, 0, 0, 9, 0, 24, 1, 0, 1, 8, 1, 4, 11, 0, 0, 2, 0, 1, 1, 0, 2, 5, 4, 1, 4, 0, 11, 1, 0, 1, 1, 1, 6, 7, 1, 4, 0, 3, 1, 1, 1, 10, 30, 1, 5, 1, 14, 8, 0, 1, 1, 27, 1, 10, 0, 4, 1, 0, 0, 1, 8, 1, 8, 9, 7, 6, 15, 13, 14, 1, 0, 1, 17, 0, 6, 31, 1, 4, 25, 80, 2, 18, 0, 1, 25, 1, 57, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 5, 0, 15, 1, 14, 1, 3, 0, 1, 1, 2, 0, 1, 2, 1, 8, 20, 15, 1, 6, 2, 1, 7, 3, 3, 1, 7, 3, 2, 8, 1, 17, 0, 1, 3, 4, 2, 0, 2, 15, 1, 1, 10, 1, 1, 1, 4, 1, 2, 1, 2, 22, 0, 10, 0, 1, 1, 2, 0, 2, 1, 12, 11, 13, 1, 1, 2, 11, 1, 3, 1, 0, 12, 42, 1, 10, 0, 0, 26, 2, 20, 2, 3, 12, 2, 5, 8, 1, 0, 1, 5, 1, 4, 2, 1, 0, 3, 2, 1, 0, 1, 17, 1, 7, 2, 1, 4, 1, 12]
Average score over all epochs: 4.113333333333333
Max score over all epochs: 80
'''

one = [0, 1, 0, 1, 0, 0, 0, 1, 0, 2, 1, 1, 0, 1, 1, 1, 2, 1, 1, 0, 1, 1, 7, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 2, 1, 3, 6, 9, 0, 1, 3, 4, 0, 0, 5, 3, 7, 1, 0, 1, 1, 1, 0, 0, 1, 2, 7, 1, 14, 1, 8, 1, 10, 1, 1, 1, 1, 3, 0, 10, 8, 47, 1, 5, 1, 23, 2, 15, 1, 5, 18, 3, 1, 10, 0, 35, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 7, 0, 3, 10, 13, 25, 1, 1, 1, 1, 1, 1, 1, 0, 1, 11, 1, 1, 19, 7, 13, 2, 32, 0, 0, 0, 2, 0, 1, 2, 15, 11, 23, 1, 0, 1, 1, 0, 4, 45, 6, 2, 3, 0, 15, 12, 1, 0, 10, 42, 23, 0, 40, 1, 1, 0, 1, 31, 32, 8, 4, 75, 1, 0, 48, 2, 0, 26, 6, 2, 1, 1, 4, 1, 0, 0, 9, 11, 0, 3, 18, 0, 7, 5, 24, 1, 1, 1, 5, 132, 4, 2, 4, 19, 0, 1, 1, 5, 3, 29, 33, 0, 0, 22, 4, 1, 1, 1, 0, 18, 0, 21, 5, 8, 23, 18, 8, 1, 5, 0, 1, 1, 1, 0, 0, 27, 7, 2, 3, 3, 8, 1, 0, 10, 1, 1, 0, 2, 0, 1, 22, 11, 0, 0, 29, 0, 6, 44, 2, 0, 0, 2, 25, 18, 1, 1, 6, 31, 3, 13, 77, 3, 1, 1, 11, 30, 1, 69, 0, 1, 3, 27, 28, 2, 29, 2, 63, 1, 0, 1, 1, 24, 25, 0, 1, 1, 13, 2, 12, 2, 1, 103, 25, 88, 0, 33, 0, 52, 1, 1, 9, 3, 0]
onee = (0.0005, 8.21, 132)

ten =[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 4, 0, 0, 0, 2, 1, 2, 1, 0, 2, 1, 5, 0, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 3, 1, 2, 1, 0, 0, 4, 1, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 2, 0, 4, 3, 1, 1, 4, 3, 1, 2, 1, 1, 21, 8, 1, 0, 11, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 19, 1, 2, 6, 0, 3, 22, 6, 8, 20, 0, 1, 54, 98, 13, 0, 4, 1, 108, 1, 5, 1, 6, 0, 0, 0, 0, 1, 1, 5, 107, 6, 1, 128, 10, 18, 2, 23, 6, 1, 15, 2, 49, 62, 36, 82, 2, 1, 0, 0, 1, 0, 2, 0, 1, 14, 45, 2, 0, 1, 1, 34, 6, 60, 100, 95, 28, 0, 2, 1, 16, 2, 84, 37, 1, 1, 95, 11, 0, 32, 0, 0, 26, 0, 2, 36, 19, 1, 3, 2, 1, 54, 31, 1, 0, 0, 1, 150, 8, 1, 16, 39, 2, 1, 1, 1, 12, 0, 0, 88, 5, 54, 1, 54, 5, 1, 14, 26, 68, 51, 1, 17, 10, 12, 37, 1, 2, 7, 0, 64, 14, 11, 3, 5, 1, 2, 1, 10, 1, 8, 2, 105, 0, 140, 0, 178, 1, 1, 26, 1, 0, 17, 1, 49, 1, 4, 8, 0, 24, 4, 0, 1, 3, 34, 2, 162, 1, 1, 27, 49, 275, 11, 1, 0, 6, 44, 2, 1, 0, 1, 1, 20, 1, 1, 83, 18, 1, 1, 1, 3, 1, 1, 17, 1, 62, 0, 1, 10, 150, 19, 1, 1, 2, 16, 1, 25, 3, 1, 1, 41, 10, 73, 11, 0, 0, 1, 105, 1, 108, 1]
tenn = (0.001, 15.673, 275)

six = [0, 0, 0, 2, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 3, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2, 1, 1, 2, 1, 1, 3, 1, 2, 1, 2, 2, 3, 1, 0, 1, 12, 2, 0, 3, 2, 0, 2, 2, 2, 1, 1, 12, 21, 2, 2, 0, 0, 2, 0, 1, 0, 3, 1, 0, 2, 1, 24, 12, 0, 0, 0, 5, 10, 0, 0, 28, 20, 6, 0, 0, 3, 65, 0, 3, 13, 1, 14, 1, 1, 1, 1, 1, 1, 5, 3, 0, 1, 1, 0, 11, 1, 1, 2, 8, 5, 1, 12, 2, 13, 15, 11, 108, 1, 2, 3, 1, 8, 1, 0, 8, 3, 4, 10, 18, 11, 1, 1, 6, 1, 0, 6, 5, 0, 1, 1, 1, 16, 5, 1, 1, 10, 0, 1, 6, 3, 1, 1, 1, 10, 3, 0, 0, 1, 2, 0, 1, 0, 10, 11, 1, 20, 3, 6, 0, 1, 1, 2, 2, 0, 0, 1, 3, 8, 0, 1, 5, 3, 8, 6, 17, 0, 0, 11, 3, 0, 0, 1, 1, 2, 14, 5, 1, 5, 1, 33, 1, 11, 1, 1, 1, 1, 33, 21, 0, 0, 0, 1, 4, 0, 1, 1, 1, 0, 0, 1, 2, 6, 1, 18, 12, 21, 0, 9, 2, 28, 1, 10, 27, 29, 1, 19, 14, 3, 30, 26, 2, 1, 1, 6, 16, 13, 0, 1, 13, 8, 4, 36, 1, 8, 5, 2, 1, 1, 30, 19, 3, 58, 19, 20, 1, 3, 0, 5, 2, 1, 0, 6, 2, 7, 1, 30, 1, 17, 1, 1, 2, 18, 36, 7, 14, 21, 28, 2, 4, 1, 1]
sixx = ( 0.0025, 5.64, 108)


seven = [0, 0, 0, 1, 0, 1, 1, 2, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 2, 1, 1, 2, 1, 2, 2, 0, 2, 1, 1, 0, 2, 2, 1, 4, 2, 1, 0, 3, 6, 1, 6, 1, 6, 1, 2, 1, 1, 1, 2, 0, 1, 0, 12, 1, 7, 3, 0, 0, 0, 1, 13, 0, 1, 1, 0, 6, 0, 1, 0, 23, 1, 0, 1, 15, 20, 2, 0, 1, 8, 19, 1, 3, 41, 0, 2, 43, 2, 1, 4, 97, 0, 9, 11, 0, 3, 17, 65, 0, 18, 64, 1, 1, 19, 2, 9, 10, 1, 16, 0, 17, 0, 0, 2, 1, 13, 40, 32, 44, 1, 1, 1, 9, 2, 8, 1, 26, 1, 1, 5, 2, 35, 1, 3, 10, 0, 3, 0, 1, 1, 0, 16, 1, 1, 1, 2, 46, 2, 1, 2, 16, 55, 8, 0, 1, 75, 14, 21, 15, 1, 14, 27, 45, 7, 7, 27, 12, 38, 1, 0, 2, 2, 24, 29, 1, 22, 1, 1, 15, 5, 1, 6, 24, 6, 0, 0, 1, 40, 10, 2, 10, 1, 1, 2, 11, 0, 0, 48, 111, 3, 1, 1, 20, 34, 86, 1, 3, 1, 6, 25, 1, 2, 65, 1, 0, 174, 36, 1, 15, 29, 3, 11, 24, 0, 20, 1, 1, 13, 1, 7, 31, 1, 2, 1, 1, 1, 30, 46, 34, 32, 1, 46, 12, 103, 0, 1, 1, 6, 32, 1, 1, 2, 26, 7, 66, 1, 0, 82, 20, 1, 1, 1, 0, 0, 5, 1, 41, 6, 1, 11, 5, 0, 5, 8, 30, 166, 2, 0, 126, 2, 31, 0, 8, 1, 0, 1, 1, 1, 22, 19, 1, 19, 30, 1, 36, 1, 0, 26, 1]
sevenn = (0.005,11.74,174)


eight =[0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 1, 1, 1, 0, 1, 2, 0, 3, 1, 1, 1, 1, 2, 1, 1, 2, 1, 0, 1, 2, 0, 0, 2, 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 1, 1, 1, 2, 4, 0, 2, 3, 1, 7, 2, 2, 2, 0, 0, 1, 4, 1, 0, 5, 4, 1, 9, 4, 8, 11, 1, 2, 21, 9, 1, 0, 0, 4, 0, 0, 8, 2, 12, 15, 6, 0, 1, 0, 1, 14, 0, 2, 1, 0, 3, 1, 1, 2, 30, 0, 11, 0, 0, 0, 4, 9, 0, 1, 9, 1, 2, 0, 1, 2, 3, 1, 0, 1, 1, 4, 2, 0, 3, 11, 3, 0, 0, 1, 0, 0, 4, 1, 2, 14, 3, 6, 2, 1, 3, 3, 1, 2, 8, 14, 6, 5, 4, 23, 4, 8, 3, 1, 4, 6, 1, 0, 0, 0, 3, 0, 1, 1, 2, 17, 59, 1, 0, 1, 0, 1, 0, 2, 1, 0, 1, 31, 19, 1, 1, 0, 1, 1, 1, 20, 2, 2, 2, 4, 1, 2, 6, 2, 1, 0, 24, 7, 0, 20, 2, 3, 0, 1, 3, 0, 1, 16, 0, 1, 18, 10, 3, 1, 6, 1, 0, 17, 22, 28, 3, 5, 1, 5, 2, 1, 0, 7, 10, 0, 8, 2, 7, 13, 3, 12, 1, 23, 4, 1, 2, 1, 2, 6, 35, 0, 1, 1, 1, 2, 5, 1, 1, 1, 3, 1, 11, 8, 17, 1, 1, 18, 48, 6, 2, 11, 5, 1, 18, 0, 3, 0, 0, 10, 15, 6, 6, 1, 21, 5, 1, 0, 0, 2, 2, 11, 3, 12, 5, 1, 13, 0, 1, 1, 3, 25, 0, 0, 0, 3, 2, 6]
eightt= (0.0075, 4.347,59)


nine = [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 0, 0, 1, 0, 3, 2, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 2, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 2, 3, 0, 2, 1, 2, 0, 0, 1, 4, 5, 0, 1, 1, 0, 1, 0, 0, 0, 3, 2, 1, 0, 0, 0, 6, 1, 4, 21, 0, 11, 3, 0, 1, 0, 1, 0, 0, 0, 0, 1, 11, 2, 0, 0, 9, 0, 24, 1, 0, 1, 8, 1, 4, 11, 0, 0, 2, 0, 1, 1, 0, 2, 5, 4, 1, 4, 0, 11, 1, 0, 1, 1, 1, 6, 7, 1, 4, 0, 3, 1, 1, 1, 10, 30, 1, 5, 1, 14, 8, 0, 1, 1, 27, 1, 10, 0, 4, 1, 0, 0, 1, 8, 1, 8, 9, 7, 6, 15, 13, 14, 1, 0, 1, 17, 0, 6, 31, 1, 4, 25, 80, 2, 18, 0, 1, 25, 1, 57, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 5, 0, 15, 1, 14, 1, 3, 0, 1, 1, 2, 0, 1, 2, 1, 8, 20, 15, 1, 6, 2, 1, 7, 3, 3, 1, 7, 3, 2, 8, 1, 17, 0, 1, 3, 4, 2, 0, 2, 15, 1, 1, 10, 1, 1, 1, 4, 1, 2, 1, 2, 22, 0, 10, 0, 1, 1, 2, 0, 2, 1, 12, 11, 13, 1, 1, 2, 11, 1, 3, 1, 0, 12, 42, 1, 10, 0, 0, 26, 2, 20, 2, 3, 12, 2, 5, 8, 1, 0, 1, 5, 1, 4, 2, 1, 0, 3, 2, 1, 0, 1, 17, 1, 7, 2, 1, 4, 1, 12]
ninee = (0.01, 4.113, 80)

plt.figure()
plt.plot(np.arange(len(six)), one, 'c.')
plt.plot(np.arange(len(six)), ten, 'k.')
plt.plot(np.arange(len(six)), six, '.')
plt.plot(np.arange(len(six)), seven, 'r.')
plt.plot(np.arange(len(six)), eight, 'g.')
plt.plot(np.arange(len(six)), nine, 'm.')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Varying Epsilon over 300 Epochs. Eta = 1, Gamma = 1.')
plt.legend(["Epsilon = {}, Avg = {}, Max = {}.".format(onee[0],onee[1], onee[2]),"Epsilon = {}, Avg = {}, Max = {}.".format(tenn[0],tenn[1], tenn[2]),"Epsilon = {}, Avg = {}, Max = {}.".format(sixx[0],sixx[1], sixx[2]),"Epsilon = {}, Avg = {}, Max = {}.".format(sevenn[0],sevenn[1], sevenn[2]), "Epsilon = {}, Avg = {}, Max = {}.".format(eightt[0],eightt[1], eightt[2]), "Epsilon = {}, Avg = {}, Max = {}.".format(ninee[0],ninee[1], ninee[2])])
plt.show()



"""
eta: 0.9, gamma: 1, epsilon: 0.001.
[0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 1, 2, 1, 1, 1, 0, 2, 1, 3, 1, 1, 1, 1, 2, 1, 0, 2, 0, 1, 1, 1, 2, 0, 0, 1, 2, 1, 1, 1, 4, 3, 1, 3, 0, 1, 1, 4, 0, 1, 1, 1, 18, 8, 1, 2, 3, 1, 8, 0, 6, 12, 0, 0, 0, 89, 2, 22, 0, 1, 0, 1, 53, 1, 3, 14, 16, 9, 1, 1, 8, 1, 1, 12, 1, 7, 11, 1, 0, 3, 9, 2, 1, 1, 6, 15, 20, 1, 0, 1, 0, 34, 2, 0, 4, 2, 2, 39, 38, 1, 25, 2, 22, 0, 0, 0, 13, 1, 38, 0, 5, 25, 0, 1, 5, 1, 0, 1, 1, 2, 1, 3, 11, 1, 1, 3, 6, 11, 8, 1, 1, 70, 3, 0, 6, 1, 7, 5, 51, 0, 10, 1, 0, 0, 0, 2, 0, 1, 52, 8, 1, 1, 3, 35, 1, 8, 21, 5, 9, 0, 1, 50, 17, 0, 3, 2, 4, 2, 1, 10, 0, 2, 7, 0, 119, 8, 0, 41, 6, 0, 2, 23, 1, 2, 49, 4, 7, 1, 48, 2, 1, 1, 16, 14, 9, 33, 17, 11, 53, 1, 1, 0, 0, 5, 3, 22, 0, 62, 1, 0, 2, 14, 6, 1, 1, 1, 26, 2, 13, 13, 22, 2, 0, 0, 0, 12, 4, 3, 13, 2, 12, 1, 1, 10, 1, 1, 25, 147, 2, 2, 1, 2, 0, 4, 2, 12, 0, 1, 12, 36, 59, 39, 7, 34, 1, 0, 1, 42, 2, 2, 34, 1, 2, 1, 5, 97, 0, 0, 1, 0, 0, 12, 0, 18, 4, 15, 85, 5]
Average score over all epochs: 8.626666666666667
Max score over all epochs: 147


eta: 0.8, gamma: 1, epsilon: 0.001.
[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 3, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 3, 0, 1, 2, 2, 2, 1, 8, 1, 3, 2, 1, 2, 2, 1, 6, 10, 21, 0, 11, 12, 9, 1, 0, 1, 1, 0, 1, 13, 6, 1, 9, 2, 1, 7, 3, 2, 1, 2, 2, 38, 1, 12, 2, 0, 6, 1, 5, 14, 8, 17, 11, 1, 0, 17, 9, 15, 7, 1, 9, 9, 6, 1, 1, 1, 2, 1, 11, 35, 1, 1, 12, 4, 1, 5, 0, 21, 6, 2, 3, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 4, 1, 1, 4, 1, 8, 12, 1, 11, 11, 1, 1, 1, 0, 1, 1, 3, 26, 0, 30, 6, 0, 0, 18, 2, 16, 1, 1, 1, 1, 5, 1, 27, 1, 34, 1, 18, 2, 7, 51, 0, 19, 3, 7, 15, 1, 2, 15, 1, 19, 1, 1, 9, 1, 3, 3, 2, 12, 13, 5, 1, 1, 11, 1, 26, 2, 7, 0, 3, 1, 1, 0, 0, 1, 1, 15, 6, 1, 1, 97, 0, 0, 0, 1, 31, 0, 0, 0, 4, 1, 1, 4, 1, 1, 34, 1, 1, 0, 10, 2, 21, 1, 25, 16, 1, 4, 14, 2, 1, 4, 2, 0, 0, 20, 13, 0, 0, 22, 2, 6, 3, 17, 3, 6, 19, 56, 2, 7, 24, 1, 2, 1, 1, 3, 27, 9, 2, 10, 1, 2, 1, 6, 1, 3, 11, 1, 3, 3, 26, 2, 2, 1, 1, 0, 13, 20, 2, 30, 3, 1]
Average score over all epochs: 5.6066666666666665
Max score over all epochs: 97


eta: 0.7, gamma: 1, epsilon: 0.001.
[1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 0, 0, 1, 0, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 2, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2, 2, 2, 2, 1, 4, 0, 2, 0, 1, 2, 6, 1, 11, 0, 23, 7, 12, 0, 1, 1, 14, 0, 15, 10, 14, 1, 92, 0, 1, 0, 22, 2, 4, 3, 16, 0, 3, 2, 4, 94, 0, 0, 10, 8, 3, 3, 0, 0, 3, 2, 8, 0, 0, 10, 0, 5, 0, 1, 5, 0, 5, 0, 0, 0, 23, 1, 1, 1, 35, 2, 1, 1, 5, 22, 2, 0, 0, 2, 1, 5, 18, 5, 10, 39, 3, 2, 1, 55, 14, 0, 48, 28, 1, 0, 34, 6, 11, 0, 6, 44, 21, 0, 17, 1, 1, 1, 1, 0, 14, 23, 0, 19, 0, 0, 1, 18, 8, 1, 55, 1, 0, 2, 0, 1, 8, 11, 0, 0, 5, 29, 2, 28, 4, 45, 1, 55, 8, 4, 1, 2, 60, 1, 0, 17, 21, 20, 2, 0, 0, 3, 0, 20, 15, 15, 2, 10, 1, 2, 1, 10, 20, 21, 7, 2, 6, 4, 0, 0, 32, 3, 1, 1, 8, 9, 1, 14, 1, 75, 1, 0, 1, 0, 57, 1, 68, 15, 86, 0, 36, 2, 10, 42, 1, 1, 1, 3, 0, 1, 0, 43, 42, 2, 24, 1, 17, 1, 0, 0, 5, 26, 14, 12, 6, 133, 5, 10, 1, 26, 55, 0, 1, 102, 1, 21, 0, 1, 8, 1, 1, 4, 1, 1, 1, 30, 0, 2, 0, 0, 1, 0, 28, 7, 0, 1, 0, 16, 35, 1]
Average score over all epochs: 9.046666666666667
Max score over all epochs: 133


eta: 0.6, gamma: 1, epsilon: 0.001.
[1, 0, 1, 2, 1, 0, 1, 0, 0, 1, 0, 1, 2, 1, 0, 0, 0, 2, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 0, 1, 2, 1, 1, 1, 0, 0, 2, 0, 0, 2, 0, 0, 1, 1, 1, 5, 2, 1, 1, 0, 7, 3, 1, 3, 2, 1, 6, 2, 3, 2, 19, 3, 15, 13, 3, 0, 2, 4, 8, 0, 6, 0, 22, 2, 9, 1, 16, 0, 1, 1, 4, 18, 1, 9, 1, 4, 0, 2, 6, 7, 5, 4, 21, 3, 3, 2, 2, 1, 2, 2, 1, 0, 1, 0, 9, 1, 0, 0, 5, 8, 2, 0, 6, 8, 3, 0, 1, 4, 1, 0, 3, 1, 42, 3, 1, 1, 7, 1, 1, 7, 2, 1, 1, 1, 2, 1, 1, 1, 4, 0, 1, 0, 1, 35, 7, 4, 0, 12, 57, 1, 1, 1, 2, 5, 35, 1, 1, 10, 20, 3, 1, 10, 1, 0, 28, 0, 10, 0, 4, 0, 0, 1, 1, 2, 6, 0, 13, 0, 19, 8, 0, 0, 0, 2, 0, 1, 14, 20, 10, 9, 34, 2, 0, 0, 0, 20, 0, 11, 1, 0, 0, 24, 0, 16, 4, 2, 1, 13, 38, 16, 0, 2, 2, 8, 3, 1, 0, 24, 22, 3, 2, 0, 17, 3, 0, 2, 0, 0, 2, 4, 2, 11, 5, 3, 6, 2, 5, 12, 2, 21, 44, 7, 2, 17, 10, 1, 24, 13, 1, 0, 1, 6, 52, 0, 0, 4, 9, 3, 0, 11, 1, 1, 1, 0, 1, 0, 1, 11, 2, 4, 8, 0, 0, 2, 25, 9, 57, 1, 5, 12, 1, 34, 3, 0, 5]
Average score over all epochs: 5.113333333333333
Max score over all epochs: 57
"""

six = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 1, 2, 1, 1, 1, 0, 2, 1, 3, 1, 1, 1, 1, 2, 1, 0, 2, 0, 1, 1, 1, 2, 0, 0, 1, 2, 1, 1, 1, 4, 3, 1, 3, 0, 1, 1, 4, 0, 1, 1, 1, 18, 8, 1, 2, 3, 1, 8, 0, 6, 12, 0, 0, 0, 89, 2, 22, 0, 1, 0, 1, 53, 1, 3, 14, 16, 9, 1, 1, 8, 1, 1, 12, 1, 7, 11, 1, 0, 3, 9, 2, 1, 1, 6, 15, 20, 1, 0, 1, 0, 34, 2, 0, 4, 2, 2, 39, 38, 1, 25, 2, 22, 0, 0, 0, 13, 1, 38, 0, 5, 25, 0, 1, 5, 1, 0, 1, 1, 2, 1, 3, 11, 1, 1, 3, 6, 11, 8, 1, 1, 70, 3, 0, 6, 1, 7, 5, 51, 0, 10, 1, 0, 0, 0, 2, 0, 1, 52, 8, 1, 1, 3, 35, 1, 8, 21, 5, 9, 0, 1, 50, 17, 0, 3, 2, 4, 2, 1, 10, 0, 2, 7, 0, 119, 8, 0, 41, 6, 0, 2, 23, 1, 2, 49, 4, 7, 1, 48, 2, 1, 1, 16, 14, 9, 33, 17, 11, 53, 1, 1, 0, 0, 5, 3, 22, 0, 62, 1, 0, 2, 14, 6, 1, 1, 1, 26, 2, 13, 13, 22, 2, 0, 0, 0, 12, 4, 3, 13, 2, 12, 1, 1, 10, 1, 1, 25, 147, 2, 2, 1, 2, 0, 4, 2, 12, 0, 1, 12, 36, 59, 39, 7, 34, 1, 0, 1, 42, 2, 2, 34, 1, 2, 1, 5, 97, 0, 0, 1, 0, 0, 12, 0, 18, 4, 15, 85, 5]
sixx =(0.9,8.627, 147)


seven =[0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 0, 0, 3, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 3, 0, 1, 2, 2, 2, 1, 8, 1, 3, 2, 1, 2, 2, 1, 6, 10, 21, 0, 11, 12, 9, 1, 0, 1, 1, 0, 1, 13, 6, 1, 9, 2, 1, 7, 3, 2, 1, 2, 2, 38, 1, 12, 2, 0, 6, 1, 5, 14, 8, 17, 11, 1, 0, 17, 9, 15, 7, 1, 9, 9, 6, 1, 1, 1, 2, 1, 11, 35, 1, 1, 12, 4, 1, 5, 0, 21, 6, 2, 3, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 4, 1, 1, 4, 1, 8, 12, 1, 11, 11, 1, 1, 1, 0, 1, 1, 3, 26, 0, 30, 6, 0, 0, 18, 2, 16, 1, 1, 1, 1, 5, 1, 27, 1, 34, 1, 18, 2, 7, 51, 0, 19, 3, 7, 15, 1, 2, 15, 1, 19, 1, 1, 9, 1, 3, 3, 2, 12, 13, 5, 1, 1, 11, 1, 26, 2, 7, 0, 3, 1, 1, 0, 0, 1, 1, 15, 6, 1, 1, 97, 0, 0, 0, 1, 31, 0, 0, 0, 4, 1, 1, 4, 1, 1, 34, 1, 1, 0, 10, 2, 21, 1, 25, 16, 1, 4, 14, 2, 1, 4, 2, 0, 0, 20, 13, 0, 0, 22, 2, 6, 3, 17, 3, 6, 19, 56, 2, 7, 24, 1, 2, 1, 1, 3, 27, 9, 2, 10, 1, 2, 1, 6, 1, 3, 11, 1, 3, 3, 26, 2, 2, 1, 1, 0, 13, 20, 2, 30, 3, 1]
sevenn =(0.8,5.607, 97)


eight =[1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 0, 0, 1, 0, 1, 1, 0, 1, 2, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 2, 1, 0, 2, 1, 2, 0, 1, 0, 2, 1, 2, 2, 2, 2, 1, 4, 0, 2, 0, 1, 2, 6, 1, 11, 0, 23, 7, 12, 0, 1, 1, 14, 0, 15, 10, 14, 1, 92, 0, 1, 0, 22, 2, 4, 3, 16, 0, 3, 2, 4, 94, 0, 0, 10, 8, 3, 3, 0, 0, 3, 2, 8, 0, 0, 10, 0, 5, 0, 1, 5, 0, 5, 0, 0, 0, 23, 1, 1, 1, 35, 2, 1, 1, 5, 22, 2, 0, 0, 2, 1, 5, 18, 5, 10, 39, 3, 2, 1, 55, 14, 0, 48, 28, 1, 0, 34, 6, 11, 0, 6, 44, 21, 0, 17, 1, 1, 1, 1, 0, 14, 23, 0, 19, 0, 0, 1, 18, 8, 1, 55, 1, 0, 2, 0, 1, 8, 11, 0, 0, 5, 29, 2, 28, 4, 45, 1, 55, 8, 4, 1, 2, 60, 1, 0, 17, 21, 20, 2, 0, 0, 3, 0, 20, 15, 15, 2, 10, 1, 2, 1, 10, 20, 21, 7, 2, 6, 4, 0, 0, 32, 3, 1, 1, 8, 9, 1, 14, 1, 75, 1, 0, 1, 0, 57, 1, 68, 15, 86, 0, 36, 2, 10, 42, 1, 1, 1, 3, 0, 1, 0, 43, 42, 2, 24, 1, 17, 1, 0, 0, 5, 26, 14, 12, 6, 133, 5, 10, 1, 26, 55, 0, 1, 102, 1, 21, 0, 1, 8, 1, 1, 4, 1, 1, 1, 30, 0, 2, 0, 0, 1, 0, 28, 7, 0, 1, 0, 16, 35, 1]
eightt = (0.7,9.047, 133)

nine =[1, 0, 1, 2, 1, 0, 1, 0, 0, 1, 0, 1, 2, 1, 0, 0, 0, 2, 1, 2, 2, 0, 2, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 0, 1, 2, 1, 1, 1, 0, 0, 2, 0, 0, 2, 0, 0, 1, 1, 1, 5, 2, 1, 1, 0, 7, 3, 1, 3, 2, 1, 6, 2, 3, 2, 19, 3, 15, 13, 3, 0, 2, 4, 8, 0, 6, 0, 22, 2, 9, 1, 16, 0, 1, 1, 4, 18, 1, 9, 1, 4, 0, 2, 6, 7, 5, 4, 21, 3, 3, 2, 2, 1, 2, 2, 1, 0, 1, 0, 9, 1, 0, 0, 5, 8, 2, 0, 6, 8, 3, 0, 1, 4, 1, 0, 3, 1, 42, 3, 1, 1, 7, 1, 1, 7, 2, 1, 1, 1, 2, 1, 1, 1, 4, 0, 1, 0, 1, 35, 7, 4, 0, 12, 57, 1, 1, 1, 2, 5, 35, 1, 1, 10, 20, 3, 1, 10, 1, 0, 28, 0, 10, 0, 4, 0, 0, 1, 1, 2, 6, 0, 13, 0, 19, 8, 0, 0, 0, 2, 0, 1, 14, 20, 10, 9, 34, 2, 0, 0, 0, 20, 0, 11, 1, 0, 0, 24, 0, 16, 4, 2, 1, 13, 38, 16, 0, 2, 2, 8, 3, 1, 0, 24, 22, 3, 2, 0, 17, 3, 0, 2, 0, 0, 2, 4, 2, 11, 5, 3, 6, 2, 5, 12, 2, 21, 44, 7, 2, 17, 10, 1, 24, 13, 1, 0, 1, 6, 52, 0, 0, 4, 9, 3, 0, 11, 1, 1, 1, 0, 1, 0, 1, 11, 2, 4, 8, 0, 0, 2, 25, 9, 57, 1, 5, 12, 1, 34, 3, 0, 5]
ninee = (0.6,5.133, 57)

ten =[1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 2, 2, 4, 0, 0, 0, 2, 1, 2, 1, 0, 2, 1, 5, 0, 1, 1, 1, 1, 1, 2, 2, 1, 1, 0, 3, 1, 2, 1, 0, 0, 4, 1, 2, 1, 1, 2, 0, 1, 1, 2, 2, 0, 2, 0, 4, 3, 1, 1, 4, 3, 1, 2, 1, 1, 21, 8, 1, 0, 11, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 19, 1, 2, 6, 0, 3, 22, 6, 8, 20, 0, 1, 54, 98, 13, 0, 4, 1, 108, 1, 5, 1, 6, 0, 0, 0, 0, 1, 1, 5, 107, 6, 1, 128, 10, 18, 2, 23, 6, 1, 15, 2, 49, 62, 36, 82, 2, 1, 0, 0, 1, 0, 2, 0, 1, 14, 45, 2, 0, 1, 1, 34, 6, 60, 100, 95, 28, 0, 2, 1, 16, 2, 84, 37, 1, 1, 95, 11, 0, 32, 0, 0, 26, 0, 2, 36, 19, 1, 3, 2, 1, 54, 31, 1, 0, 0, 1, 150, 8, 1, 16, 39, 2, 1, 1, 1, 12, 0, 0, 88, 5, 54, 1, 54, 5, 1, 14, 26, 68, 51, 1, 17, 10, 12, 37, 1, 2, 7, 0, 64, 14, 11, 3, 5, 1, 2, 1, 10, 1, 8, 2, 105, 0, 140, 0, 178, 1, 1, 26, 1, 0, 17, 1, 49, 1, 4, 8, 0, 24, 4, 0, 1, 3, 34, 2, 162, 1, 1, 27, 49, 275, 11, 1, 0, 6, 44, 2, 1, 0, 1, 1, 20, 1, 1, 83, 18, 1, 1, 1, 3, 1, 1, 17, 1, 62, 0, 1, 10, 150, 19, 1, 1, 2, 16, 1, 25, 3, 1, 1, 41, 10, 73, 11, 0, 0, 1, 105, 1, 108, 1]
tenn = (1, 15.673, 275)

plt.figure()
plt.plot(np.arange(len(six)), ten, 'k.')
plt.plot(np.arange(len(six)), six, '.')
plt.plot(np.arange(len(six)), seven, 'r.')
plt.plot(np.arange(len(six)), eight, 'g.')
plt.plot(np.arange(len(six)), nine, 'm.')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Varying Eta over 300 Epochs. Epsilon = 0.001, Gamma = 1.')
plt.legend(["Eta = {}, Avg = {}, Max = {}.".format(tenn[0],tenn[1], tenn[2]), "Eta = {}, Avg = {}, Max = {}.".format(sixx[0],sixx[1], sixx[2]),"Eta = {}, Avg = {}, Max = {}.".format(sevenn[0],sevenn[1], sevenn[2]), "Eta = {}, Avg = {}, Max = {}.".format(eightt[0],eightt[1], eightt[2]), "Eta = {}, Avg = {}, Max = {}.".format(ninee[0],ninee[1], ninee[2])])
plt.show()

