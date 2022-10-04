# Intro
The code here is pretty much everything I use to create my visuals. Currently only including code used to create my tables using Soccerment & StatsBomb data but will be adding code that I use to create other visuals using Fbref data.

Feel free to download any of this code and use it yourself, it's on the MIT license. No need to give attribution to me, but any credit or even a tag (@egudi_analysis on twitter) is much appreciated! And I'd love to see the work you do with this code, of course!

# Installing Jupyter Notebook
All of this code is written in Python but using Jupyter Notebook. Jupyter Notebook is free and safe, and easy, to install. If you don't have it on your computer yet, there are several ways to get it; I'll give two examples below.

## Project Jupyter
One method is by downloading direct from Project Jupyter: https://jupyter.org/install

Follow the short instructions there to download. You can download and start coding in seconds.

## Anaconda
Another way to get Jupyter Notebook is to install it via Anaconda, which acts as an environment to host much of your coding needs: https://www.anaconda.com/products/distribution

This method is what I use, however there are a couple extra steps to begin working in Jupyter Notebook. But, it is still easy to follow and Anaconda gives good instructions.

# Running These Code Files: No Major Coding Experience Needed!
Once you have Jupyter Notebook, you can begin running these code files.

Download the files from this repository and open them with Jupyter Notebook. The instructions should hopefully be straight forward for all files.

## Soccerment Folder
The 1st code file to run is the "StrikerRatings", where I import, clean and plot the data. I include the data I exported from Soccerment which you can use to load into the StrikerRatings file and run the code.

To run that code, all you need to do is click "Run" in the top bar in Jupyter Notebook. There are some minor manual adjustmentes which I detail in the file and then you can start making your own tables from the data!

## StatsBomb Folder
The 1st code file to run is "StatsBombDataLoad", where I load the StatsBomb data for all La Liga Seasons Messi played in. Feel free to skip this file and begin working with the Messi.csv file I have already exported and uploaded within the Data folder. HOWEVER, IF YOU WANT TO ANALYZE OTHER PLAYERS/EVENTS OTHER THAN MESSI (OR PLAYERS MESSI PASSED TO), YOU MUST RUN THIS FILE TO GET ALL OF THE EVENTS.

The 2nd code file to run is "StatsBombExpectedThreateHeatmaps", where I then import the csv file exported from the first file and then clean and add relevent columns used for the xT viz. Within this file is also the code used to plot the heatmaps using the mplsoccer package. Everything should be automated to just spit out the viz, but make sure to update the titles if you are looking at another player and add your own twitter handle for the credits.

Please contact me if you want to change anything but are lost or get stuck!

# Notes
Overall, I hope that you won't need too much coding experience to run these programs. You'll of course need coding experience if you want to customize them (colors, titles, etc.), but if you're wanting to fully customize them I'll assume you know how code some decent amount of Python anyway! If not, feel free to reach out to me and I can help you get your 'brand' or 'style' instead of mine.

Finally, never hesitate to contact me, ever. I hope people can use this not only as a guide for their code, or as a way to quickly download some ratings from favorite leagues, begin working with event data, but also to help anyone who wants to learn Python. If you have any questions at all, or especially if there are problems with the code (or just want to chat about why I did X or Y), shoot me a DM on Twitter! @egudi_analysis.

Also, I have not been coding for too long so code may be somewhat inefficient so for any experienced coders any feedback is appreciated!

Good luck everyone and thanks for using this!
