# Required imports
from deephaven.TableTools import readCsv
from deephaven import Plot

# Read the CSV file into a Deephaven table
creditcard = readCsv("/data/examples/CreditCardFraud/csv/creditcard.csv")

def plot_valid_vs_fraud(col_name):
    # Set the creditcard table as global to make sure we can access it for the plot
    global creditcard

    # Make sure the input corresponds to a column
    allowed_col_names = [item for item in range(1, 29)] + ["V" + str(item) for item in range(1, 29)]
    if col_name not in allowed_col_names:
        raise ValueError("The column name you specified is not valid.")
    if isinstance(col_name, int):
        col_name = "V" + str(col_name)

    # Some convenience variables for plotting
    num_valid_bins = 50
    num_fraud_bins = 50
    valid_label = col_name + "_Valid"
    fraud_label = col_name + "_Fraud"
    valid_string = "Class = 0"
    fraud_string = "Class = 1"

    # Create a fancy histogram plot
    valid_vs_fraud = \
        Plot\
        .histPlot(valid_label, creditcard.where(valid_string), col_name, num_valid_bins)\
        .twinX()\
        .histPlot(fraud_label, creditcard.where(fraud_string), col_name, num_valid_bins)\
        .show()
    return valid_vs_fraud

valid_vs_fraud_V4 = plot_valid_vs_fraud("V4")
valid_vs_fraud_V12 = plot_valid_vs_fraud("V12")
valid_vs_fraud_V14 = plot_valid_vs_fraud("V14")
