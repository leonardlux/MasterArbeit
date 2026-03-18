
# current questions

choice of logical repr in the end
    - should not matter
    - but at least if I choose diff, for X and Z: d>3 fails! (Z multi, X single and X obs)
        - difference only appears for ML decoder 

# Thematischer Bezug

Read more:
    Nielsen and Chuang 
    Gottesmann 

# Anders
Formalismus vereinheitlichen!
Write a lot of stuff into my latex overview file 

# ToDos 

0. more results:
basic: 
look at |+> (Z-errors/phase-flips) and |0> (X-errors/bit-flips) states
save a circuit diagram for distance 3
and MWPM Decoder and ML Decoder
    - 'full' range to show asymptotic behavior
        - use data collapse method here as well (should also work)
    - indepth around p_th 
        - determine p_th
            + discard d=3
        - compare to literature

This week
1. save data
    1. save results to file
2. threshold for circuit lvl noise
    - threshold dependent on rounds for circ lvl noise

## Orderd ToDo List

8. reread fundamental threshold (understand how to figure out the fundamental threshold for models with Y-errors)

10. compare to d rounds stabilizer measurement as one QEC cycle
    + for each of the d put error on data qubit
11. check order of stabilizer cnots (to get the free distance improvement)

12. enable different order

13. enable bell state as inital state + different logic measurement 

14. write down error propagation in latex 

## Other ToDos
0. 2 obs will be a problem for my current count logical implementation!
0. reimplement noise model func and propagated noise model, such that both are id by one parameter!
0. increase throughput:
    use numba more!! (for faster results) 
    and parralism with numba
1. compare to MWPM with allknowing detector Error model!
2. enable complex error models
3. tex error propagation part

## Question from midterm

Why does CSS always has transversal CNOT
Quantum memory under circuit noise use density matrix operations

sub set sampling might be valid as a sampling method for low error prob
show more asymptotic behavior (and explain the exponents fucntion)

# 18.03 Wednesday

00. implement |+> state
    + works for everything 
01. jitted everything and parrallised something (little return)
02. 
    
# 17.03 Tuesday 

00. get rid of error of the Hadmard
01. Meeting 
02. working out data collapse method
03. surface_code.py and mwpm decoder |+> state