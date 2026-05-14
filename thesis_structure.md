# Master Thesis Structure

# Title: Circuit Level Characterization of Steane Type Error Correction


# Abstract


# Introduction


# Theoretical Basics/Fundamentals

## Quantum Error Correction (QEC)

### Threshold Theorem

Meaning of threshold vs distance for the capacity of a code


## Stabilizer Codes and their Structure

0. What is a stabilizer code
    + Concept of code stabilizers 
    + Resulting Group structure:
        + Stabilizer Group
        + Centralizer and Cosets
        + Pauli Group and how it is detectable
1. Code distance and correctable errors
    + uncorrectable vs undetectable

### Unrotated Surface Code

0. Defintion & Structure
    + Scaline with distance 
    + Properties (Transversal CNOTs)
1. Benefits  
2. Existence of Rotated Surface code and reasons why we did not choose this one


## Decoding 

0. From Syndrome to Correction
    + Explain in Group picture image
1. How to find the correct Recovery

### MWPM Decoding

0. Definiton: Lowest Weight Error
1. Shortly Describe the Method, Principal of matching Graphs (how much detail?)
2. Describe benefits and shortcommings

### ML Decoding

0. Definiton using group structure
1. Why is it hard?
2. How much detail for the Method used?


## Error/Noise Model

### Code Capacity Model
Just errors on data qubits

### Circuit Level Noise
0. concept of fault tolerance and residual error
1. Principal of fault tolerance check

### Phenomological Noise Model
Correct Defintion
Why useful?



## Syndrome Extraction

### Steane Type Syndrome Extraction = Steane Type Error Correction

Basic Principle

0. Log trival operations
1. Error Propagations through CNOTs
Resulting Conditions:
0. Works for all codes with transversal CNOTs (-> CSS Codes)(understsand <- direction )

Properties:

0. Discuss Ordering 
    + X/Z vs. Order symmetry
1. Discuss how to treat residual error and why it is still correctable
2. Discuss encoding problem?

## Pauli Frame Tracking

## Logical Encoding/Preparation of Logical States (ToDo)

## Finit Size Scaling analysis aka. how to determine threshold

0. explain how this methods works 
0. show assumptions 


# Methodology

## Data Generation: Steane Type Error Correction (Syndrome/Observable)

0. Stim
Describe Circuit:
1. Encode surface code qubits:
    1. starts in arbitrary Coset -> Pauli Frame Tracking
    2. ToDo: different ways of simulating FT log prepared states

2. Steane Type Error Correction:
    1. Order vs X-/Z-Symmetry
    2. Detectors need to be deterministic 

3. Repeat for Multi Rounds

4. FT-Check
    1. Final Error Free Syndrome Extraction -> Check if observable is recoverable up to corretable error
    2. Motivate why we can use MWPM regardeless of other Decoding choice

5. Observable

Stim generates Syndrome and observable result. 

Modifiable parameters of Circuit

0. Distance
1. Rounds
2. Observable
    + X
    + Z
    + ToDo: X & Z
3. Noise Model
    + Code capcity
    + Circuit Level Noise
4. Encoding Circuit (ToDo)
5. Order 
    + not implemented for reasons -> symmtry (ToDo, just to show)

### Data generation Repeated Measurement readout.

0. shortly describe the difference
1. find good argumenation why we need d rounds of measurement readout for ft


## Decoding (Predictions)

0. given Syndrome -> determines correct Prediction
1. all assume independet X and Z noise (could implement for correlated noise (TODO?))

### ML* Implementaion:

0. Propagate Error Model -> split up 
    + discuss if we find anything interesting in this new model
1. use newly calculated phenomological noise model for ML Decoding
2. Discuss if this is truly ML implementation
3. Discuss numerical imprecission Problems
    + quantify those (TODO)

### MWPM Implementation:

1. All knowing:
    + just pymatching over the full circuit, more info
2. comparable amount of information     
    + assuming perfect syndrome extraction (compare ML to MWPM)
    + DEM from basic surface code (noise rate not relevant, short discussion)
3. Mutliple rounds correlated MWPM, but with perfect syndrome extraction
    + Shortly discuss problem in using pymatching
    + TODO acutally implement this


## Different Runs

### Single Round

1. MWPM: all info
2. MWPM: assume perfect syndrome extraction for surface code 
3. ML: assume perfect syndrome extraction for surface code 

### Multi Round

1. MWPM: all info
2. MWPM: repeated single round decoding (TODO)
    + point out difference: aka change of incomming state
3. ML: repated single round decoding

### How to determine Threshold aka. FSSA/Data collapse

0. describe implementation
1. argue why per round is not needed 


# Results

## Code capacity case

0. show basic curves, show threshold and determined threshold and fssa results
1. compare to literatur values to determined threshold
2. discuss difference to literatur (smal scale effects)


## Single Round 

1. Compare different decoding strategies 
2. compare different observables -> show difference in order

## Mutli Round

1. Show threshold develops over multiple rounds
    + compare decoder
    + compare observable

# Conclusion
