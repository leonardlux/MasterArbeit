# Master Thesis Structure

# Title: Circuit Level Characterization of Steane Type Error Correction


# Abstract


# Introduction

Eplain the title of parts


# Theoretical Basics/Fundamentals

## Quantum Memory

### Quantum Circuits

0. concept of quantum memory experiments

### Pauli Error 

0. What are real world physical noise process (some examples)
1. How to find a good theoretical approach to model those
    + assumption model them as pauli errors
1. introduce 
    + bit-/phase- flip noise
    + introduce depolirizing noise

#### Code Capacity Model

0. Just errors on data qubits
1. Simplest model
    + Surface code is designed for this model 
2. describe the resulting error model

#### Circuit Level Noise

0. Ancilla qubits and gates are faulty 
    + more realworld like
    + motivate the need to think about syndrome extraction
0. concept of fault tolerance and residual error
    + error on qubits only with prob p
1. Principal of fault tolerance check
2. Add two qubit depolirizing noise 
3. describe the resulting full error model

##### Phenomological Noise Model

0. Defintion
1. Why useful?



## Quantum Error Correction (QEC)

Motivation: Why is it needed, and what are the basic prinicples
    + Example

0. What is the basic theoretical idea behind Quantum error correction 
    + Encode log data in higher dimensional hilbertspaces 
    + Use redundancy for error correction
1. what are the main problems/Why doe we need Quantum error correction?
    + (no clone theorem) find better arguments xD
    + measurements collapse quantum states 
2. What is the definition of a successful recovery


### Threshold Theorem

0. why is the concept of threshold relevant
1. explain how to determine threshold (may be later?)

Meaning of threshold vs distance for the capacity of a code (maybe a bit to early)

## Stabilizer Codes and their Structure

0. What is a stabilizer code
    + Concept of code stabilizers 
    + Resulting Group structure:
        + Stabilizer Group
        + Centralizer and Cosets
        + Pauli Group and how it is detectable
    + Principle of Syndrome and why it is not enough -> leads to Decoding Problem
1. Code distance and correctable errors
    + uncorrectable vs undetectable
    + here talk about the difference between threshold and distance?
2. General Properties of stabilizer codes
    + Are all stabilizer codes CSS codes? No!
    + Setup introduction to Surface/Toric codes

### Unrotated Surface Code

0. Defintion & Structure
    + logical and stabilizers
    + Scaling with distance 
    + Properties (Transversal CNOTs)
1. Benefits  
    + in comparison to toric code/planar
    + and some scientific consensus
2. Existence of Rotated Surface code and reasons why we did not choose this one

## Syndrome Extraction

0. talk about influence of circuit level noise 
    + -> need to think about a smart way of syndrome extraction
1. ToDo: How does the group picture change?
    + can we find a nice way to express this?

### 'Basic' Surface code ancilla syndrome extraction ciruit

0. Talk about basic layout
    + for code capcity case
    + explain concept of hook error 
2. error on ancilla qubit and how they propagate through correction
    + -> not fault tolerant
    + -> d repeated measurements needed (ToDo find good paper)

### Steane Type Syndrome Extraction = Steane Type Error Correction

0. Log trival operations
1. Error Propagations through CNOTs
Resulting Conditions:
0. Works for all codes with transversal CNOTs (-> CSS Codes)(understsand <- direction )

Properties:

0. Discuss Ordering 
    + X/Z vs. Order symmetry
1. Discuss how to treat residual error and why it is still correctable
2. Discuss trade offs 
    + change more space requirement for less time requirement
3. Discuss encoding problem?


## Decoding 

0. From Syndrome to Correction
    + Explain in Group picture image
1. How to find the correct Recovery
    + different approaches 
2. definiton of logical error rate 
    + how to count logical errors

### ML Decoding

0. Definiton using group structure
    + show to be optimal decoding strategy
1. Why is it hard?
2. How much detail for the Method used (?)
    + Mapping to RBIM  
    + Mapping to those circuits (TODO: properly understand all of this)

### MWPM Decoding

0. Definiton: Perfect Matching on the Matching graphs 
    + uniform p -> shortest error chain -> cyclic in p
1. Shortly Describe the Method, Principal of matching Graphs (how much detail?)
2. Describe benefits and disadvantages in relation to ML 
3. Talk about time correlated MWPM?
    + this is needed for Multi Round experiments 


## Pauli Frame Tracking

0. Motivate uses case 
    + Express as virutal correction (by change of next input, in case of clifford)
1. Introduce basic concept 
    + track pauli (all clifford)
    + save on needed physical gates
2. Relevance in this work 
    + given that all operations are non clifford


## Logical Encoding/Preparation of Logical States (ToDo)

0. Important for steane type error correction
0. How are we preparing a logical |0> / |+> state,
    + show that they are logical 0 or +
    + talk about pauli frame tracking -> random coset of centralizer 
    + still observable fixed to be correct 
        + tricial but not that trivial

## Finit Size Scaling analysis aka. how to determine threshold (Move to a different point in the thesis)

0. Motivate on why we need this:
    + determine thershold
    + threshold beeing a oder/unorder transition as described in ML chapter
0. explain how this methods works 
0. show assumptions 


# Methodology


## The Circuits 

### Steane Type Error Correction 

+ generate Syndrome and Observable

0. Stim
Describe Circuit:
1. Encode surface code qubits:
    1. starts in arbitrary Coset -> Pauli Frame Tracking
    2. ToDo: different ways of simulating FT log prepared states

2. Steane Type Error Correction:
    1. Order vs X-/Z-Symmetry
    2. Detectors need to be deterministic 
        + not truly pauli frame tracking but similar (tracking base syndrome)
        + only trigger on change
    3. Repeat for Multi Rounds

4. FT-Check
    1. Final Error Free Syndrome Extraction -> Check if observable is recoverable up to corretable error
    2. Motivate why we can use MWPM regardeless of other Decoding choice
        + fall back to p beeing cyclical in MWPM

5. Observable
    1. Shortly discuss the influence of different choices of logical observables

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

### Surface Code repated measurements (standard way)

0. shortly describe the difference
1. find good argumenation why we need d rounds of measurement readout for ft

### ToDo: Encoding Circuits

+ Describe the functioning of different encoding circuits


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

#### Short Tangent: Effective Error model(?)

+ (?) not well structured

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

## Different Encodings

# Conclusion
