#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 06:33:49 2024

@author: joncparamore
"""

from math import exp, log
import scipy.optimize
import numpy as np
#setting directory
#cd /Users/joncparamore/Desktop/Full_test/



#Fill output (input-output) with forms from data
#inp_out is a dictionary that associates ordered pairs of morpheme tags with their surface forms. 
# (Stem-tag, Suffix-tag) : Surface-Form
#output is the tool used to import the data to different functions.
def data_import_prep(filename):
    with open (filename, "r", encoding = "utf-8") as myfile:
        data = [line.replace('\ufeff', '') for line in myfile.read().splitlines()]
    inp_out = {}
    for x in data:
        parts = x.split(',"')
        surface_form = parts[0]
        tags = parts[1].replace('"', '').replace('(', '').replace(')', '').split(',')
        pair = (int(tags[0]), int(tags[1]))
        inp_out[(pair[0], pair[1])] = surface_form
    return inp_out
inp_out = data_import_prep('pan_data.csv')

####feature assignment
#set of nasal vowels
NASALV = ['A', '4', 'E']
#set of oral vowels
ORALV = ['ɑ','ɒ', 'ə']
#set of glides
ORALG = ['ʋ']
NASALG = ['V']
#obstruents
ORALOBS = ['t', 's', 'g', 'ʃ', 'd', 'r']
NASALOBS= ['T', 'S', 'G', 'Z', 'D', 'R']
OBS = ORALOBS + NASALOBS
#Defining nasal consonants
NASALC = ['n']
#list of all nasal segments
NASAL = NASALV + NASALG + NASALOBS + NASALC
#set of all oral segments
ORAL = ORALV + ORALG + ORALOBS
#round vowels
RDV = ['ɒ', '4']
#low round vowels
LowRd = ['ɒ', '4']
# Mapping from oral vowels to nasal vowels
oral_to_nasal = {'ɑ': 'A', 'ɒ': '4', 'ə': 'E',
                 'ʋ': 'V',
                 't': 'T', 's': 'S', 'g': 'G', 'ʃ': 'Z', 'd': 'D', 'r': 'R'}
# Mapping from nasal vowels to oral vowels
nasal_to_oral = {v: k for k, v in oral_to_nasal.items()}

#Mapping from round to unround vowels
round_to_unround = {'ɒ': 'ɑ', '4': 'A'}

####Constraints and weights from phonotactic stage
#List of the names of the constraints used in this simulation
CONSTRAINTS = "IDnas IDFin(nas) Sprd-L *NasObs *NasG ID(nas)/_V(-nas) *VN ID(rd) *LowRd".split()
#Whether the constraints are faithfulness (f) or markedness (m), ties to whether they are biased high or low.  
CONTYPE="f f m m m cf m f m".split()
#setting all constraint weights to 50
initial_WEIGHTS = [50] * len(CONTYPE)

####Candidate Generation
#helper function to check if a candidate has any nasal segments
def has_nasal(candidate):
    for segment in candidate:
        if segment in NASAL:
            return True
    return False
		
#main candidate generation function
#input is all SRs from output (stem, suf)
#output is the candset list with all potential candidates for the entire dataset.
#Take in an input of [stem, affix]
def candidates(inpair):
    #combine stem and suffix to create UR
    UR = f"{inpair[0]}{inpair[1]}"
    
    #defining fully faithful candidate (straightforward for me since no deletion, etc.) This is where
    #the correspondence relations between UR and SR are generated.
    FFC = tuple([(i, i) for i in range(len(UR))])

    #initialize list to store potential candidates
    candset = []
    
    #Step 1: create the fully faithful candidate and add it to candset
    candidate = UR
    candset.append((candidate, FFC))
    
    #Step 2: Generate all possible unrounded candidates for URs with underlying LowRd vowels
    unrounded_candidates = [UR]
    for i in range(len(UR)):
        if UR[i] in LowRd:
            # Generate a candidate where the low round vowel becomes unrounded
            #make the UR a mutable list
            new_cand = list(UR)
            new_cand[i] = round_to_unround[new_cand[i]]
            unrounded_candidates.append(new_cand)
            
    #### Step 3: Apply nasalization and denasalization to each candidate ####
    for candidate in unrounded_candidates:   
        #only generate nasal candidates if the original candidate has a nasal segment
        if has_nasal(candidate):
            #for each segment in the candidate index
            for seg_index in range(len(candidate)):
                #if the segment is nasal in the UR
                if UR[seg_index] in NASAL:
                    #iterate over each segment index preceding the nasal segment
                    for i in range(0,seg_index+1):
                        #if i is the nasal segment in question, generate a candidate with it 
                        #and all nasal segments directly to its left unnasalized
                        if i == (seg_index):
                            #make candidate a mutable list
                            new_cand = list(candidate)
                            #change target segment to oral
                            if new_cand[i] in nasal_to_oral:
                                new_cand[i] = nasal_to_oral[new_cand[i]]
                                #change immediate preceding target segment to oral
                                if new_cand[i-1] in nasal_to_oral:
                                    new_cand[i-1] = nasal_to_oral[new_cand[i-1]]
                                    #change 2 left preceding target segment to oral
                                    if new_cand[i-2] in nasal_to_oral:
                                        new_cand[i-2] = nasal_to_oral[new_cand[i-2]]
                                        #change 3 left preceding target segment to oral
                                        if new_cand[i-3] in nasal_to_oral:
                                            new_cand[i-3] = nasal_to_oral[new_cand[i-3]]
                                # Add the new candidate to candset
                                candset.append(("".join(new_cand), FFC))
                        #if i is the directly preceding segment, generate a candidate with it nasalized
                        if i == (seg_index-1):
                            #make candidate a mutable list
                            new_cand = list(candidate)
                            if new_cand[i] in oral_to_nasal:
                                new_cand[i] = oral_to_nasal[new_cand[i]]
                                # Add the new candidate to candset
                                candset.append(("".join(new_cand), FFC))
                        #if i is two segments before, generate a candidate with it and the the next segment nasalized
                        elif i == (seg_index-2):
                            new_cand = list(candidate)
                            if new_cand[i+1] in OBS:
                                continue
                            elif new_cand[i] in oral_to_nasal:
                                new_cand[i] = oral_to_nasal[new_cand[i]]
                                if new_cand[i+1] in oral_to_nasal:
                                    new_cand[i+1] = oral_to_nasal[new_cand[i+1]]
                                    # Add the new candidate to candset
                                    candset.append(("".join(new_cand), FFC))
                        elif i == (seg_index-3):
                            new_cand = list(candidate)
                            if (new_cand[i+1] in OBS) or (new_cand[i+2] in OBS):
                                continue
                            elif new_cand[i] in oral_to_nasal:
                                new_cand[i] = oral_to_nasal[new_cand[i]]
                                if new_cand[i+1] in oral_to_nasal:
                                    new_cand[i+1] = oral_to_nasal[new_cand[i+1]]
                                    if new_cand[i+2] in oral_to_nasal:
                                        new_cand[i+2] = oral_to_nasal[new_cand[i+2]]
                                        # Add the new candidate to candset
                                        candset.append(("".join(new_cand), FFC))
                        elif i == (seg_index-4):
                            new_cand = list(candidate)
                            if (new_cand[i+1] in OBS) or (new_cand[i+2] in OBS) or (new_cand[i+3] in OBS):
                                continue
                            elif new_cand[i] in oral_to_nasal:
                                new_cand[i] = oral_to_nasal[new_cand[i]]
                                if new_cand[i+1] in oral_to_nasal:
                                    new_cand[i+1] = oral_to_nasal[new_cand[i+1]]
                                    if new_cand[i+2] in oral_to_nasal:
                                        new_cand[i+2] = oral_to_nasal[new_cand[i+2]]
                                        if new_cand[i+3] in oral_to_nasal:
                                            new_cand[i+3] = oral_to_nasal[new_cand[i+3]]
                                            # Add the new candidate to candset
                                            candset.append(("".join(new_cand), FFC))
                if UR[seg_index] in NASALOBS:
                    new_cand = list(candidate)
                    new_cand[seg_index] = nasal_to_oral[new_cand[seg_index]]
                    # Add the new candidate to candset
                    candset.append(("".join(new_cand), FFC))
                
        if not has_nasal(candidate):
            #for each segment in the candidate index
            for seg_index in range(len(candidate)):
                #if the segment is nasal in the UR
                if (UR[seg_index] in ORALV) and (UR[seg_index] == UR[-1]):
                    new_cand = list(candidate)
                    new_cand[seg_index] = oral_to_nasal[new_cand[seg_index]]
                    candset.append(("".join(new_cand), FFC))
    
    #### Step 3: Ensure all candidates are unique ####
    # Ensure all candidates are unique by converting to set and back to list
    candset = list(set(candset))
    
    return candset

#test: iterate over each word form in output list
#candset = []
#for stem, suf in inp_out:
    #surfaceform = inp_out[(stem, suf)]
    #add the surface form and its variants to the candidate set
    #candset += candidates([surfaceform, ""])

####Violations definitions
#Function that determines violations for each input-output candidate for all constraints in model
#inputs are (1) SR tags in inp_out (stem,suf) and (2) candidates generated by candidates()
#output is a list of violations for the nine constraints..e.g., [1, 2, 0, 0, etc.]
def vios(stem, suf, outpair):
    UR = f"{stem}{suf}"
    SR = outpair[0]
    corresp = outpair[1]
    # Create a dictionary for correspondence relations
    crelation = {x[0]: x[1] for x in corresp}
    #create list to track violations
    violations = []
    
    ####ID(nas)
    #For every segment, A, assign a violation if the output value for the (nasal) feature dominated by A 
    #does not match the input value for the (nasal) feature dominated by A.
    #define variable to hold number of violations
    IDnas = 0
    #iterate over every segment in the UR
    for i in range(len(UR)):
            UR_segment = UR[i] #defining what the current UR segment is
            if i in crelation:
                output_segment = SR[crelation[i]]
                #check for violations of ID(nas)
                if (UR_segment not in NASAL and output_segment in NASAL) or (UR_segment in NASAL and output_segment not in NASAL):
                    IDnas += 1         
    #append number of violations of ID(nas) to the violation vector, violations
    violations.append(IDnas)
    
    ####IDFin(nas)
    #For every segment, A, assign a violation if the output value for the (nasal) feature dominated by A 
    #does not match the input value for the (nasal) feature dominated by A in the final position of a prosodic word.
    #define variable to hold number of violations
    IDfinnas = 0
    wordfinal_segment = len(UR) - 1
    #iterate over every segment in the UR
    if wordfinal_segment in crelation:
        UR_segment = UR[wordfinal_segment] #defining what the word-final UR segment is
        output_segment = SR[crelation[wordfinal_segment]]
        #check for violations of IDFin(nas)
        if (UR_segment not in NASAL and output_segment in NASAL) or (UR_segment in NASAL and output_segment not in NASAL):
            IDfinnas += 1
    #append number of violations of IDFin(nas) to the violation vector, violations
    violations.append(IDfinnas)
    
    ####Sprd-L
    #For every occurrence of a (+nas) feature in a prosodic word, if that (+nas) feature is dominated by some segment, 
    #assign a violation for every segment to the left of that segment in the prosodic word that does not dominate the (+nas) feature.
    #define variable to hold number of violations
    SprdL = 0
    counted_non_nasal_segments = set() #makes sure each segment to the left of one or more nasals is only counted once as violation
    #iterate over every segment in the UR
    for i in range(len(SR)):
        output_segment = SR[i] #defining what the current output segment is
        #Check if the current segment is NASAL
        if output_segment in NASAL:
            #For each segment to the left of the current output nasal segment
            for j in range(i-1, -1, -1):
                left_output_segment = SR[j]
                # If an obstruent is encountered, 
                if (left_output_segment in OBS):
                    #if it is oral and hasn't been counted, count it and stop
                    if (left_output_segment not in NASAL) and (left_output_segment not in counted_non_nasal_segments):
                        counted_non_nasal_segments.add(left_output_segment)
                        SprdL += 1  # Count the obstruent as a violation
                        break
                    #if it oral and HAS been counted, just stop
                    elif (left_output_segment not in NASAL) and (left_output_segment in counted_non_nasal_segments):
                        break
                    #if the obstruent is nasal, just stop
                    elif (left_output_segment in NASAL):
                        break
                elif (left_output_segment not in NASAL) and (left_output_segment not in counted_non_nasal_segments):
                    # Add the left segment to the set if it hasn't been counted
                    counted_non_nasal_segments.add(left_output_segment)
                    SprdL += 1
                
                    
    #append number of violations of Sprd-L to the violation vector, violations
    violations.append(SprdL)
    
    ####*NasObs
    #Assign a violation for every obstruent or liquid that is nasal on the surface.     
    #define variable to hold number of violations
    StarNasObs = 0
    #iterate over every segment in the SR
    for i in range(len(SR)):
        output_segment = SR[i]
        if output_segment in NASALOBS:
            StarNasObs += 1
    #append number of violations of StarNasObs to the violation vector, violations
    violations.append(StarNasObs)
            
    ####*NasG
    #Assign a violation for every glide that is nasal on the surface.      
    #define variable to hold number of violations
    StarNasG = 0
    #iterate over every segment in the SR
    for i in range(len(SR)):
        output_segment = SR[i]
        if output_segment in NASALG:
            StarNasG += 1
    #append number of violations of StarNasG to the violation vector, violations
    violations.append(StarNasG)
    
    ####ID(nas)/_V(-nas)
    #Let A be a segment that occurs before an oral vowel, __V(-nasal), in the input. 
    #Assign one violation if the output correspondent of A does not have the same specifications for (nasal) as A.
    #define variable to hold number of violations
    IDnas_Voral = 0
    #iterate over every segment in the UR
    for i in range(len(UR)):
            UR_segment = UR[i] #defining what the current UR segment is
            if i != wordfinal_segment:
                following_UR_segment = UR[i+1]
                if i in crelation:
                    output_segment = SR[crelation[i]] #defining corresponding output segment
                    #check for violations of ID(nas)/_V(-nas)
                    if (following_UR_segment in ORALV) and (output_segment in NASAL):
                        IDnas_Voral += 1
    #append number of violations of IDnas_Voral to the violation vector, violations
    violations.append(IDnas_Voral)  
        
    ####*VN
    #Assign a violation for every oral vowel that occurs before a nasal consonant
    #define variable to hold number of violations
    StarVN = 0
    #iterate over every segment in the UR
    for i in range(len(UR)):
            UR_segment = UR[i] #defining what the current UR segment is
            if (UR_segment in NASALC) and (i in crelation) and (i > 0):
                preN_seg = SR[crelation[i-1]] #defining pre-N segment
                #check for violations of *VN
                if preN_seg in ORALV:
                    StarVN += 1
    #append number of violations of *VN to the violation vector, violations
    violations.append(StarVN)
    
    ####ID(rd)
    #For every segment, A, assign a violation if the output value for the (round) feature of A 
    #does not match the input value for the (round) feature of A.
    #define variable to hold number of violations
    IDrd = 0
    #iterate over every segment in the UR
    for i in range(len(UR)):
            UR_segment = UR[i] #defining what the current UR segment is
            if i in crelation:
                output_segment = SR[crelation[i]] #defining corresponding output segment
                #check for violations of ID(rd)
                if (UR_segment not in RDV and output_segment in RDV) or (UR_segment in RDV and output_segment not in RDV):
                    IDrd += 1         
    #append number of violations of ID(rd) to the violation vector, violations
    violations.append(IDrd)
    
    ####*LowRd
    ##Assign a violation for every low round vowel that appears on the surface.
    #define variable to hold number of violations
    StarLowRd = 0
    #iterate over every segment in the SR
    for i in range(len(SR)):
        output_segment = SR[i]
        if output_segment in LowRd:
            StarLowRd += 1
    #append number of violations of StarLowRd to the violation vector, violations
    violations.append(StarLowRd)
        
    return violations

def violations_test(data):
    violations = []
    for stem, suf in data:
        surface_form = inp_out[(stem, suf)]
        candset = candidates([surface_form, ""])
        for candidate in candset:
            for i in range(len(initial_WEIGHTS)):
                violations.append(((f'surface form: {surface_form}'), (f' candidate: {candidate[0]}'), (f'{CONSTRAINTS[i]}: {vios(surface_form, '', candidate)[i]}')))
    return violations
#test_vios = violations_test(inp_out)

####Harmony Score Generator
#inputs are (1) SR from output, (2) candidates generated by candidates(), and (3) initial WEIGHTS 
#output is a harmony score value for the UR-SR candidate
def harmony(inputpair, outputpair, WEIGHTS):
    #set the base harmony score at zero
    harmonyscore = 0
    #find the violation vector of the current input-output candidate
    violations = vios(inputpair[0], inputpair[1], outputpair)
    #Perform dot product of the constraint weight and violation vectors
    for x in range(len(violations)):
        harmonyscore += WEIGHTS[x] * (-violations[x])
    #return harmony score
    return harmonyscore

####probability generator
#inputs are (1) SR in output and (2) initital WEIGHTS. 
#candidates() function called to generate candidate set.
#harmony() function called to provide harmony scores for each candidate.
#output is dictionary with each candidate as a key and the associated P(x) as the value.
def probs(inputpair,WEIGHTS):
    #generate candidates using candidates function
    candset=candidates(inputpair)
    
    #initialize total harmony score
    htotal=0
    
    #initialize dictionary to store harmony score for each candidate
    harmdict={}
    
	#For each candidate find the harmony score and add it to the total harmony score of all candidates.
    for candidate in list(set(candset)):
        harmdict[candidate] = exp(harmony(inputpair, candidate, WEIGHTS))
        htotal += harmdict[candidate]
    
    #Calculate the probability for each candidate
    for candidate in harmdict: 
        if htotal < 1e-150: #If all forms have very low probability give all 0 probability
            harmdict[candidate]=0
        else:
            harmdict[candidate] = harmdict[candidate] / htotal #calculate MaxEnt P(x)
    
    #debugging print statements
    #print(f"probs() called with input: {inputpair}")#, weights: {WEIGHTS}")
    #print(f"Returned probabilities: {harmdict}")
    return harmdict

####L2 Gaussian Prior
#Defining relevant constants for the prior
#numcon represents the number of constraints in the grammar.
numcon=len(CONTYPE)
#SIGMAM is a constant that represents the plasticity of markedness constraints, the higher SIGMAM, the more markedness constraints can move.
SIGMAM=10
#SIGMAF does the same thing for faithfulness constraints.
SIGMAF=25
#SIGMACF does the same thing for contextual faithfulness constraints
SIGMACF = 10
#mbias is the constraint weight that markedness constraints are biased towards---meant to be high, in order to cause restrictiveness
mbias=100
#cfbias is the constraint weight that contextual faithfulness constraints are biased towards --also meant to be high to cause restrictiveness
cfbias = 100
#Bound used to prevent constraint weights from getting below too negative.
negative=-0.00000001

#Prior function
#input is the inital WEIGHTS
#output is prior value
#Calculate the L2 gaussian prior, the sum of the difference of each weight from it's bias point (mbias or 0) squared divided by the associated SIGMA value
def prior(WEIGHTS):
    total=0
    for i in range(len(WEIGHTS)):
        w = WEIGHTS[i]
        if CONTYPE[i] == 'f':
            total += w**2 / (2 * SIGMAF**2)
        elif CONTYPE[i] == 'cf':
            total += (w - cfbias)**2 / (2 * SIGMACF**2)
        else:
            total += (w - mbias)**2 / (2 * SIGMAM**2)
    return total	

####Objective function
#input is a vector of initialWEIGHTS
#probs() function called in Objective
#prior is called in Objective
#inp_out SRs used in Objective
#output is the value of the objective: the negative log likelihood of the data (objective_value) + prior value
def objective(parameters):
    #Separate the constraint weights (w) from the UR frequencies (lvect)
    w=parameters[:numcon]
    
    #initialize variable to store sum of all negative log likelihood values for all data points
    objective_value = 0
    
    #for each datapoint, find the stem and suffix morpheme tags
    #iterate over each word form in inp_out dict
    for stem, suf in inp_out:
        surface_form = inp_out[(stem, suf)]
        #initialize empty dictionary to store calculated probabilities for each candidate surface form
        outputprobs = {}
        #variable that stores the output of the probs() function to calculate probabilities of candidates, given current surface form and weights (w)
        probcand = probs((surface_form, ""), w)
        #iterate over each candidate/probability pair in probcand
        for x in probcand:
            #check if candidate already in outputprobs and add the probability if so
            if x[0] in outputprobs:
                outputprobs[x[0]] += probcand[x]
            #if not there, add it and its probability to the dictionary
            else:
                outputprobs[x[0]] = probcand[x]
        #extract the likelihood of the observed surface form from outputprobs
        likelihood = outputprobs[surface_form]
        
        #if likelihood of the winner is super low, print error to mess up the objective function
        if likelihood <= 1e-150:
            objective_value += float('inf')
        #otherwise, add the negative log likelihood of the surface form arising for the given output to objective_value
        else:
            objective_value += -log(likelihood)
            
    #As a failsafe, if some parameter is outside of its bounds, make the objective really big to let the algorithm know it messed up
    for x in parameters:
        if x < negative:
            return float('inf')
    
    #return the sum of the negative log likelihood with the prior on constraints
    return objective_value + prior(w) 

####Gradient Descent function
#Function to calculate the gradient of the objective function
#input is initialWEIGHTS
#inp_out SRs used to defined surface forms
#vios() called
#probs() called
#output is the gradient (slope of error at objective function value)
def gradient(parameters):
    #Initialize an empty list to store the gradients for each weight
    grad = []
    #assign parameters input to WEIGHTS
    WEIGHTS = parameters
    
    #initialize lists to store the observed 'Observed' and expected 'Expected' violation counts for each constraint: zeros as initial violations
    Observed = [0] *len(WEIGHTS)
    Expected = [0] *len(WEIGHTS)

    #iterate over each surface_form in inp_out,
    for stem, suf in inp_out:
        surface_form = inp_out[(stem, suf)]
        #initializing an empty tuple (FFC)
        FFC = ()
        #and for each segment in surface_form,
        for x in range(len(surface_form)):
            #populating it with correspondence points
            FFC = FFC + ((x,x),)
        #Call vios() function to calculate violations for the surface_form and its correspondence relations
        violations = vios(surface_form, '', (surface_form, FFC))
        
        #add the violations for the observed surface form to the 'Observed' list
        for x in range(len(violations)):
            Observed[x] += violations[x]
    
    #now iterate over each surface_form in inp_out again, calculating 'Expected' violations
    for stem, suf in inp_out:
        surface_form = inp_out[(stem, suf)]
        #get probability of each candidate for surface_form, given current weights
        probdist = probs((surface_form, ''), WEIGHTS)
        #iterate over each candidate in probdist
        for cand in probdist:
            #calculate expected violations for each candidate
            violations = vios(surface_form, '', cand)
            for x in range(len(violations)):
                Expected[x] += probdist[cand] * violations[x]
    
    #iterate over observed and expected violations and calculate the gradient for each weight
    for x in range(len(Observed)):
        if CONTYPE[x] == 'f':
            grad.append((Observed[x] - Expected[x] + WEIGHTS[x] / (SIGMAF**2)))
        elif CONTYPE[x] == 'cf':
            grad.append((Observed[x] - Expected[x] + (WEIGHTS[x] - cfbias) / (SIGMACF**2)))
        else:
            grad.append((Observed[x] - Expected[x] + (WEIGHTS[x] - mbias) / (SIGMAM**2)))
    
    #print out current constraint weights for inspection
    print("---CONSTRAINTS------")
    printconstraints(parameters[:numcon])
    
    #Calculate magnitude of the gradient vector, showing how large the gradient is overall
    print(f" gradient {np.linalg.norm(grad)}")
    
    #Print current value of objective function
    print(f" objective  {objective(parameters)}")
    
    #print full gradient vector
    print(grad)
    
    #convert the gradient list to a NumPy array for use in optimization algorithm
    return np.asarray(grad)


#function to print constraint weights
#Print the weight of each constraint, with its name.
def printconstraints(dictionary):
    string=""
    for x in range(numcon):
        string += f"\t{CONSTRAINTS[x]}\t\t{dictionary[x]:.2f},"
        #print out four constraints per line
        if x%4==3:
            print(f"{string}")
            string=""
        print(f"{string}") 

#Create bounds for the optimization algorithm (don't let constraints or frequencies get negative or weights get above 200)
#Create bounds for the optimize algorithm (don't let constraints or frequencies get negative or weights get above 200)
wbound=(0,100)
bounds=[wbound,]*(numcon)

####SLSQP optimization algorithm
#inputs are (1) the objective function, (2) array of initialWEIGHTS, (3) bounds of initialWEIGHTS, (4) gradient() function, eps (stepsize), etc.
#output is a summary of the optimization.
#Run the SLSQP optimization algorithm on the objective function using the gradient function as a jacobian.
res=scipy.optimize.minimize(objective,#efines my objective function as the function I want to minimize
                            np.asarray(initial_WEIGHTS), #converts initial WEIGHTS to a NumPy array for optimization starting point
                            method='SLSQP', #Specifies that SLSQP algorithm should be used
                            bounds=bounds, #applies previously-defined bounds to set limits on constraint weights
                            jac=gradient, #Specifies above gradient function as the Jacobian, allowing optimizer to use the gradient to find the direction of steepest descent
                            options={'maxiter':1000, #limit the maximum number of iterations to 1000
                                     'ftol':1e-10, #Sets the function tolerance to 1e-50, meaning the optimizer will stop when changes in the function value are smaller than this threshold.
                                     'eps':1e-5, #sets step size of the Jacobian (step_size = gradient * Learning_Rate)
                                     'disp': True #display convergence message once optimization is complete
                                     })

####post-optimization printing

#stores optimized weights and prints them
w1=res.x 
print("------Optimized Constraint Weights------")
print(w1)


#Print the results of the simulation.
print("CONSTRAINT WEIGHTINGS:")
for x in range(len(w1)):
	print (f"{CONSTRAINTS[x]}\t\t{w1[x]:.2f}")
    
phonotactic_constraints = np.round(w1, 2)
print(phonotactic_constraints)