# Practical 4 README

To run the code, you will need the following packages: 
1) numpy.random
2) pygame
3) random
4) matplotlib
5) os

and the code for the SwingyMonkey class. Note that this code has been altered slightly from the distributed code, so that the coordinates of the monkey at death are recorded. This class can be found in this git directory and these packages can be installed easily with pip install or other similar methods.

After downloading this directory, to run the code and display the video of the monkey learning, cd into the directory of the code (SwingingMonkey.py and final.py should be in the same directory) and run "python final.py" in terminal.

If you'd like sound, in the following function at the bottom of the code under "run_games":
swing = SwingyMonkey(sound=False,                 
                     text="Epoch %d" % (ii),      
                     tick_length = t_len,          
                     action_callback=learner.action_callback,
                     reward_callback=learner.reward_callback)
change sound to "True".

If you'd like to see the monkey (who wouldn't?), comment out the lines

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ['SDL_VIDEODRIVER'] = 'dummy'

Also, you'll notice that their is a gridsearch.py file and a graphs.py file in the directory. The former is a tool we used to help us find optimal parameters, and the latter is an example of how we made some of the graphs in our write-up.

Enjoy!

Your friend,

Monkey