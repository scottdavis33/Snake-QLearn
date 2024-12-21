import matplotlib.pyplot as plt
from IPython.display import clear_output

def plot(scores, mean_scores, records):
    clear_output(wait=True)  # Clear the output to refresh the plot if needed
    plt.figure()  # Start a new figure
    plt.title('Record, Avg. Score, and Scores vs. # of Games')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')

    # Plotting with labels and colors
    plt.plot(scores, color='blue', label='Scores')            # Scores in blue
    plt.plot(mean_scores, color='green', label='Mean Scores') # Mean scores in green
    plt.plot(records, color='red', label='Records')           # Records in red

    plt.ylim(ymin=0)
    
    # Adding text annotations for the last values on the plot
    plt.text(len(scores)-1, scores[-1], str(scores[-1]), color='blue')
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]), color='green')
    plt.text(len(records)-1, records[-1], str(records[-1]), color='red')

    plt.legend()  # Display the legend to show labels
    plt.show(block=True)  # Make sure the plot blocks further execution until closed

# Ensure that your script calls plot() function after the game ends
