import sys
import re
import pandas as pd
import numpy  as np

# print instructions
print('This is a parser to go from the CSV format used in the analysis\n'
      'to the same format used to generate the starting dataset.\n\n'
      'You need to pass a filename to the script at the command line\n'
      'pointing to a CSV file. The parser will produce a new file with the\n'
      'same name and the suffix "_parsed" and extension ".dat".\n'
      'A second command line argument must be included with the number of\n'
      'the starting column from which to extract files. If omitted the\n'
      'default behaviour is to start from the first column (i.e. column 0).\n'
      'Examples:\n\n'
      '         python parser.py data_file.csv 1   # start from the 2nd col.\n'
      '         python parser.py data_file.csv 0   # start from the 1st col.\n'
      '         python parser.py data_file.csv     # start from the 1st col.\n\n'
      'You can then import the file into Mathematica using:\n\n'
      '         Import[FILENAME, "Text"];\n\n'
      'Then just copy and paste the output and remove the initial and final\n'
      'quotes. Scientific notation has already been accounted for, so\n'
      'data is ready for use.\n\n'
      'You need to install the Python module `pandas` for this script\n'
      'to work.\n'
     )

# there must be a command line argument
if len(sys.argv) == 0:
    print("\n\nPlease, provide a filename as command line argument.\n\n")
    sys.exit(1)

# read data
filename = sys.argv[1]
data = pd.read_csv(filename)

# the second command line argument is the starting column
start_col = 0
if len(sys.argv) > 2:
    start_col = int(sys.argv[2])

# divide into solutions
solutions = pd.unique(data['solutions'])
columns   = list(data.columns)
columns   = [re.sub(r"^level_(.*)", r"\1", col) for col in columns]

##############################################################################
#                                                                            #
# CREATE A LIST OF LISTS                                                     #
#                                                                            #
##############################################################################

# iterate over each solution and extract a dictionary
solution_list = []
for sol in solutions:
    # select a single solution
    selection = data.loc[data['solutions'] == sol]
    row, col  = selection.shape

    # create a list containing name of the column + corresponding values
    column_list = []
    for i in range(start_col,col):
        number_list = list(selection.iloc[:,i].values)
        number_list.insert(0, columns[i])
        column_list.append(number_list)

    # append "single solution" list to larger list
    solution_list.append(column_list)

# change the list to string (substitute [] with {} for Mathematica compatibility)
solution_string = re.sub(r"\[", "{", str(solution_list))
solution_string = re.sub(r"\]", "}", str(solution_string))
solution_string = re.sub(r"\'", "", str(solution_string))
solution_string = re.sub(r"([0-9])e", r"\1 * 10^", str(solution_string))

# print list to file (substitute old extension)
new_file = re.sub(r"[.]csv$|[.]dat$|[.]tsv$|[.]data$", r"_parsed.dat",
                  filename
                 )
with open(new_file, 'w') as f:
    f.write(solution_string)
