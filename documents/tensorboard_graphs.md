### Graphs to pay attention to

### Train

1. train/length
    - it records time steps on X-AXIS and on Y-AXIS total length when episode ended AT SPECIFIC TIME STEP
    - For example at 4600 Time steps episode total length was 400,

2. train/reward
    - it shows team reward acquired at specific time step



### Test

1. test/length
    - similar to train/length but it is for testing only

2. test/length_std
    - it shows standard deviation aka how stable is agent behavior for example if std is high it means data points are spread apart it says our agent behaves differently in each episode, for example finishing game in one ep and dying at the very start in the next one, low std means our agent is more stable and gets same reward/length on avg

3. test/reward

4. test/reward_std


#### Note you can see weird behavior like high value for length of ep but very low reward that says that agent learn how to survive but it doesn't progress or walk forward
