## Task 1
Description: Observing the sky for a duration of 3 minutes yields a 60% probability of spotting a plane. Your assignment is to calculate and explain the probability of spotting a plane within 1 minute based on this observation. Provide a detailed solution outlining the reasoning behind your calculations.

### Assumptions:
- planes appear randomly at a constant rate
- planes are independent events

### Mathematical Techniques Used:
- Poisson Process

### What We Are Given:
- P(At least 1 plane) in 3 minutes = 60%

### What We Need To Do:
- P(At least 1 plane) in 1 minute.

### Mathematical Steps
```plaintext
P(T<=t) = 1 - e^(-lt)

Where: 
    P(t<=t) the probability of spotting at lest 1 plane in time t
    l is the amount of planes per minute
    t is the time duration
    e is euler's number

after 3 minutes t = 3, therefore after 3 minutes:
    P(T<=t) = 1 - e^(-3l) = 0.6
    e^(-3l) = 0.4
    -3l = ln(0.4)
    therefore...
    l = (-ln(0.4)) / 3

    => l = 0.3054 (approximately)

Now simply use the formula but for 1 minute:
    P(T<=t) = 1 - e^(-l)
            = 1 - e^(0.3054)
            = 1 - 0.7368

    => P(T<=t) = 0.2631 (approx)
```
### The Final Result

The probability of spotting at least one plane within 1 minute is approximately 26.34%
