
# current questions

# Thematischer Bezug

Read more:
    Nielsen and Chuang 
    Gottesmann 

# Anders
Formalismus vereinheitlichen!
Write a lot of stuff into my latex overview file 

# ToDos 

This week
0. get |+> states to work (Z-errors)
1. jit & parralise everything
    1. save results to file
    2. reduce redundancy in parameters passed on!
        + observable and z_stab
        + noise_model, error_func, noise_model_string
1. code capacity check threshold for MWPM and ML Decoder  
    0. discard distance 3!
    1. find literature
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


# Done today 17.03

00. get rid of error of the Hadmard