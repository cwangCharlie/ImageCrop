

# (x, y, width, height)
fieldBounds = ((8, 0, 1204, 1594), (621, 13, 545, 13), (621, 39, 545, 13), (31, 55, 1135, 1486))

# [left_x, top_y , right_x, bot_y]
coordsToCheck = [0, 0, 1250, 1600]


'''
Given bounds of all input fields, this function returns the number of fields enclosed by 
a set of coordinates
Inputs:
    fieldBounds: Tuples of all input field coordinates
    coordsToCheck: Array of coordinates to test
Outputs:
    count: Number of input fields enclosed by test coordinates
'''
def numFields(fieldBounds, coordsToCheck):
    count = 0
    left_x = coordsToCheck[0]
    right_x = coordsToCheck[2]
    top_y = coordsToCheck[1]
    bot_y = coordsToCheck[3]
    for bounds in fieldBounds:
        if(left_x <= bounds[0] and 
            right_x >= (bounds[0] + bounds[2]) and
            top_y <= bounds[1] and
            bot_y >= (bounds[1] + bounds[3])):
            count += 1
    print(count)
    return count

numFields(fieldBounds, coordsToCheck)