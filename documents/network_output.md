1. Why does network prefer to have [-1,1] output for continous results rather than any value? because RL actions could be between +5 to -5?

because math wise it make sense, if output range is very huge say 5000 to -5000, network guessing will be very wild, making training slow
it might guess 400, then 5000 next time

2. does `action_scaling=True` make network truly change to action space distribution?

no network internally is still using -1 to 1 and will remain like that for its output, it is just True allows library to use translator that
translates say 1 to +5 for action for the env