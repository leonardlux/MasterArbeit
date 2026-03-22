
# current questions

choice of logical repr in the end
    - should not matter
    - but at least if I choose diff, for X and Z: d>3 fails! (Z multi, X single and X obs)
        - difference only appears for ML decoder 

data collapse method only work close to threshold 
    + should i show asymtotic behavior? (is there anything interesting)

ML might have numerical stability problems for asymptotic behavior
    + is this a problem?! (is it due do my coding?)
    + or is it a mathematical artefact (does not seem to be one)

# Thematischer Bezug

Read more:
    Nielsen and Chuang 
    Gottesmann 

# Anders
Formalismus vereinheitlichen!
Write a lot of stuff into my latex overview file 

# ToDos 

0. run scripts on cluster!

1. Plot log_error_rate over full noise_length 
    + optional selective rounds
    1. show data for basic noise level for broad range of ML/MWPM decoder
        + talk about interesting interaction

2. Determine Threshold
    + TODO: should return errors

20. enable compare thresholds method!
    
3. compare threshold for basic case in one plot 
    + with error bars!
    + compare to literature

4. save a circuit diagram for distance 3 as a result file

5. determine threshold (and nu for circ level function)

5. add titles to plots!

## Orderd ToDo List

10. compare to d rounds stabilizer measurement as one QEC cycle
    + for each of the d put error on data qubit
    + check order of stabilizer cnots in surface code (to get the free distance improvement)

8. reread fundamental threshold (understand how to figure out the fundamental threshold for models with Y-errors)

12. enable different order

13. enable bell state as inital state + different logic measurement 

14. write down error propagation in latex 

## Other ToDos
0. 2 obs will be a problem for my current count logical implementation!
2. enable complex error models (yes!)
3. tex error propagation part

## Question from midterm

Why does CSS always has transversal CNOT
Quantum memory under circuit noise use density matrix operations

sub set sampling might be valid as a sampling method for low error prob
show more asymptotic behavior (and explain the exponents fucntion)

# 22.03 Sunday

00. determine threshold
    + selective distance
    + selective noise range

# 21.03 Saturday 

00. solved bugs in new implementation
01. enabled analysis from saved data
02. recovered results for basic test
03. plot data collpase method results


# 20.03 Friday

01. write data to file
02. write config to file -> read config -> gen data from config -> write data
03. enable write data to folder
04. start replacing old results functions with new ones 

# 19.03 Thursday

00. config file
01. reimplement syndrome decoding (more physics applied) 
02. deleted all knowing MWPM (not of interest correct ? otherwise reimplement correctly!)
03. generate data from config file

# 18.03 Wednesday

00. implement |+> state
    + works for everything 
01. jitted everything and parrallised something (little return)
    
# 17.03 Tuesday 

00. get rid of error of the Hadmard
01. Meeting 
02. working out data collapse method
03. surface_code.py and mwpm decoder |+> state