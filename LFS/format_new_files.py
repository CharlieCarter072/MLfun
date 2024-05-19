# The only use of this file is to format the large data set files after download, as they are read only in pycharm

with open('train.csv', 'r') as file:
    # Read all lines from the file
    lines = file.readlines()


# with open('train.csv', 'w') as file:
#     # Write all lines except the first one
#     file.writelines(lines[1:])

print(len(lines[0].split(",")))

