This is an R data file.  The matrix of data is just called "data".  Each row represents a single fixation, so there are multiple rows per trial.  Here are the descriptions of the variables:

- trial = trial number from 1 to 100
- rating1 = rating for the left item from 1 to 10
- rating2 = rating for the middle item from 1 to 10
- rating3 = rating for the right item from 1 to 10
- roirating = rating for the currently fixated item
- rt = reaction time for the trial (in milliseconds)
- chosenrating = rating for the chosen item
- subject = participant ID #
- eventduration = duration of the current fixation (in milliseconds)
fix_num = the fixation number within the current trial (e.g. 1 = first - fixation, 2 = second fixation, etc.)
choice1,2,3 = dummy variables for whether the left, middle, or right item was - chosen in this trial
- leftroi = dummy variable for whether the current fixation is to the left item
middleroi = dummy variable for whether the current fixation is to the middle - item
- rightroi = dummy variable for whether the current fixation is to the right item
- num_fixations = total number of fixations in the current trial
rev_fix_num = fixation number prior to the last fixation in the trial (e.g. 1 - = last fixation, 2 = second-to-last fixation, etc.)