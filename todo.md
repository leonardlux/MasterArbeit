Question?
+ numerical precision 
    + should I implement mwpm as input to ML decoding (as pauli frame)

# ToDos 

## Open Todos in case of extra time

0. FT decoding using ML <- implement!

1. qubit init: basic approach show that this breaks
    + just as intuition
    + otherwise put in writing 

2. win or lose beeing local in time (MWPM for multiple rounds of Steane Type QEC)
        
## Question from midterm

Why does CSS always has transversal CNOT
Quantum memory under circuit noise use density matrix operations

sub set sampling might be valid as a sampling method for low error prob
show more asymptotic behavior (and explain the exponents fucntion)





# 30.04 Thursday

00. wrote a bit of analysis to check how the decoder fails
talk:
todo
- plot linearistion and compare to results without linearisation (is linearisation a good approx?)
- discuss a bit the syndrome channel symmetry points in realtion to the correlated channel
    + optimizing the decoder ? is there any point of interest there?

# 29.04 Wednesday 

00. implemented test decoding case through config
01. generated a bunch of data


# 28.04 Tuesday 

00. ML Decoder works on open boundary conditions
01. finally jitted my ML implementation


# 23.04 Thursday

01. meeting 
02. error model with correlation with data qubit
03. linear order terms 


# 17.04 Friday

01. Worked on todo regarding error model:
    + product channel and symmetry depolirzing channel
    + all physical p are equal

# 13.04 Monday

01. threshold log error rate per round implemented -> nothing changed
    + make from a theory point of view sense 

# 30.03 Monday - 02.04. Thursday 

01. Start writing down error propagation in latex for luis
02. Finishing that one as well .... took more time than expected

# 27.03 Friday

01. Multi round analysis combined into one file
02. analysis done for multi round and got some interesitng results

# 26.03 Thursday

01. write data to same folder if they have the same config
    + make them destinguishable by date and unique ID
02. enable read in and combination of multiple data files into one data dict
03. restructured both data and config data structure and accordingly result scripts
    + result scripts analysis needs still work
04. got slurm to work and submitted some tasks 
    + array works
05. checked all parameter for basic idea of working! (it does)
06. basic, circ working again (and both datasets with 100k shots)
07. multi round also working again


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