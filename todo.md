
# current questions

choice of logical repr in the end
    - should not matter
    - but at least if I choose diff, for X and Z: d>3 fails! (Z multi, X single and X obs)
        - difference only appears for ML decoder 

data collapse method works close to threshold 
    + should i show asymtotic behavior? (is there anything interesting)
    + is there any way to have a systemtatic first guess for nu?

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

        
6. Solve precision problem: 
    + Luis: I agree that looks like precision problems, can you check the values of matrix elements of the A matrix ?  There one can see roughly how they decrease with p and anticipate when one runs into these problems

7. distance 27 for luis


## This week!

1. write down error propagation derivation for luis

## Simple stuff for breaks

4. save a circuit diagram for distance 3 as a result file 
    + automate the saving process more (as a result file!)

12. enable different order

12. p_window in analysis -> from single value to list each entry is for one round



## Orderd ToDo List

10. compare to d rounds stabilizer measurement as one QEC cycle
    + for each of the d put error on data qubit
    + check order of stabilizer cnots in surface code (to get the free distance improvement)

8. reread fundamental threshold (understand how to figure out the fundamental threshold for models with Y-errors)

13. enable bell state as inital state + different logic measurement 

14. write data to file with similar but not identical configs


## Other ToDos
0. 2 obs will be a problem for my current count logical implementation!
2. enable complex error models (yes!)

## Question from midterm

Why does CSS always has transversal CNOT
Quantum memory under circuit noise use density matrix operations

sub set sampling might be valid as a sampling method for low error prob
show more asymptotic behavior (and explain the exponents fucntion)

# 26.03 Thursday

01. write data to same folder if they have the same config
    + make them destinguishable by date and unique ID
02. enable read in and combination of multiple data files into one data dict
03. restructured both data and config data structure and accordingly result scripts
    + result scripts analysis needs still work
04. got slurm to work and submitted some tasks 
    + array works
05. checked all parameter
06. basic, circ working again (and both datasets with 100k shots)

# 25.03 Wednesday

00. got access to cluster again
    01. copied github there
    02. installed all dependencies
    03. got scripts to run
01. multi round fix p works and returns expected results
02. multi round determine threshold and plot for each round


# 24.03 Tuesday

00. clean up basic and circ lvl script 
    + reduce the amount of parameters and shorten as much as possible!
01. determined threshold for circ lvl noise 1 round
02. remove hadarmard error from propageted noise model
03. shifting distance window!
04. multiple rounds fixed p analysis script written

# 23.03 Monday

00. look at min_distance dependency
01. compare basic threshold to literature
02. enable error on threshold
03. error propagated the error of inv_nu to nu 
04. optional selective rounds

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