"""
Created on Mon Sep 16 07:39:41 2024

@author: joncparamore
"""

from math import exp, log
import scipy.optimize
import numpy as np
import copy

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

#LEX
#manually generate potential URs and set their probabilities of being the correct UR equal at initialization
LEX = {
  0: {'': 1},
  1: {'ʋA': 1}, 
  2: {'sɑ': 0.5, 'sA': 0.5},  
  3: {'gA': 1},  
  4: {'tɑʋAn': 0.3333333333333333, 'tɑʋɑn': 0.3333333333333333, 'tɑʋɒn': 0.3333333333333333},
  5: {'ʊʃɑ': 0.5, 'ʊʃA': 0.5},
  6: {'ʊdA': 1},
  7: {'prəʋAn': 0.3333333333333333, 'prəʋɑn': 0.3333333333333333, 'prəʋɒn': 0.3333333333333333}
}

# Probabilities_list is a list of probabilities
# tag_UR_pairs_list is a list of tag-UR pairs corresponding to Probabilities_list
Probabilities_list=[]
tag_UR_pairs_list=[]
for morpheme_tag in LEX:
	for Potential_UR in LEX[morpheme_tag]:
		Probabilities_list.append(LEX[morpheme_tag][Potential_UR])
		tag_UR_pairs_list.append([morpheme_tag,Potential_UR])
        
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
#phonotactic_weights from phonotactic learning stage
phonotactic_weights = [51.37,  44.83,  92.83, 100.00, 99.48, 100.00, 100.00, 0.00, 100.00]

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
candset = []
for stem, suf in inp_out:
    surfaceform = inp_out[(stem, suf)]
    #add the surface form and its variants to the candidate set
    candset += candidates([surfaceform, ""])
#candset = candidates(['kəlɑʋɒn', ''])

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
            for i in range(len(phonotactic_weights)):
                violations.append(((f'surface form: {surface_form}'), (f' candidate: {candidate[0]}'), (f'{CONSTRAINTS[i]}: {vios(surface_form, '', candidate)[i]}')))
    return violations
test_vios = violations_test(inp_out)

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
        harmonyscore += WEIGHTS[x]*(-violations[x])
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

# pLEX function
# Given a Lexicon with frequency counts for each underlying form, returns the normalized probability of each underlying form.        
def pLEX(Lexicon):
    # Initialize dictionary to hold normalized probabilities
    tLEX = copy.deepcopy(Lexicon)

    # Iterate over each morpheme tag in the dictionary
    for morpheme_tag in tLEX:
        # Initialize the total variable to hold the total sum of probabilities for each morpheme_tag
        total_probability = 0
        # Iterate over each potential UR for that morpheme tag
        for Potential_UR in tLEX[morpheme_tag]:
            # Add the absolute value of its probability to total
            total_probability += abs(float(tLEX[morpheme_tag][Potential_UR]))
            #print(total_probability)

        # Once the total of all potential URs for the morpheme tag is summed
        # Check to see if the total probability of URs is approaching zero
        # If it is, print warnings about which tag is the issue
        if total_probability < 1e-50:
            print(f"Sum of UR probabilities for {morpheme_tag} is zero")

        # Iterate over each potential UR again
        #The total probability for all potential URs of a given tag CANNOT be 0.
        #If it is, that means the tag has no Potential UR that is a good fit for it.
        for Potential_UR in tLEX[morpheme_tag]:
            if total_probability < 1e-150:
                tLEX[morpheme_tag][Potential_UR] = 0.0
            else:
                # And divide the probability by the sum of probabilities, normalizing it.
                tLEX[morpheme_tag][Potential_UR] = float(tLEX[morpheme_tag][Potential_UR]) / total_probability

    return tLEX

# Creates a list of parameters for the optimization function, first the weights of the constraints, then the frequencies of the URs. 
def wToParam(weights,Probabilities_list,tag_UR_pairs_list):
	CombinedParams=[]
	for w in weights:
		CombinedParams.append(w)
	for i in Probabilities_list:
		CombinedParams.append(i)
	return [CombinedParams,tag_UR_pairs_list]

  
#function to concatenate a list of probabilities and a list of tag_UR pairs (e.g., (1, 'VA')) into a single parameter
#provides a good way to refer to tag UR pairs and the corresponding probability to that tag-UR pair: just use the index, 
#which is the same for both
def vectToParam(Lexicon):
    probs_list = []
    tag_UR_pairs = []
    URprobs = copy.deepcopy(Lexicon)

    for tag in URprobs:
        if len(URprobs[tag]) > 1:#warning: turned this off
            for UR in URprobs[tag]:
                probs_list.append(URprobs[tag][UR])
                tag_UR_pairs.append((tag, UR))
    return [probs_list, tag_UR_pairs] 

Probabilities_and_tag_UR_pairs = vectToParam(LEX)

####L2 Gaussian Prior
#Defining relevant constants for the prior
#numcon represents the number of constraints in the grammar.
numcon=len(CONTYPE)
#SIGMAM is a constant that represents the plasticity of markedness constraints, the higher SIGMAM, the more markedness constraints can move.
SIGMAM=20
#SIGMAF does the same thing for faithfulness constraints.
SIGMAF=25
#SIGMACF does the same thing for contextual faithfulness constraints
SIGMACF = 10
#mbias is the constraint weight that markedness constraints are biased towards---meant to be high, in order to cause restrictiveness
mbias=100
#cfbias is the constraint weight that contextual faithfulness constraints are biased towards --also meant to be high to cause restrictiveness
cfbias = 100
#sigma is used as a scaling factor for the prior which prevents frequencies of URs from getting too high.
sigma=.001
#Bound used to prevent constraint weights from getting below too negative.
negative=-0.00000001

#Prior function
#input is the phonotactic_weights
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

#Calculate the sum of the absolute value of all probabilities of underlying representations.	
#This prior sums the absolute value of all probabilities of the complete list of UR probabilities.
#it is multiplied by the scaling factor (sigma) in the objective. 
#The result is a larger addition to the objective function as this sum is higher,
#which seems to create a preference for UR probabilities to be minimal 
def priorl(UR_probabilities):
	return sum((probability**2) for probability in UR_probabilities)

####Objective Function
#input is a single vector with phonotactic_weights as the first 9 indices [0:numcon] and LEXprobs as the remaining inices [numcon:]
#LEXURkey called on to get associated morpheme tag and UR
#pLEX() function used to get updated probabilities
#output called on to get surfaceforms
#probs() function called
#output is the value of the objective: the negative log likelihood of the data (probsum) + prior value of weights and prior value of URs.
def objective(parameters):
    # Separate the constraint weights (w) from the UR probabilities (LEXprobs)
    WEIGHTS = parameters[:numcon]
    UR_probabilities = parameters[numcon:]
    #assign the list of (tag, PotentUR) tuples to a variable
    tag_UR_pairs_list = Probabilities_and_tag_UR_pairs[1]
    
    #build a dictionary to hold probability of each UR (before normalization)
    URprobs = {}
    for i in range(len(tag_UR_pairs_list)):
        tag, UR = tag_UR_pairs_list[i]
        probability = UR_probabilities[i]
        if tag in URprobs:
            URprobs[tag][UR] = probability
        else:
            URprobs[tag] = {UR: probability}
    
    #build a dictionary of normalized orobabilities using pLEX
    URprobs_Normalized = pLEX(URprobs)
    
    objective_value = 0
    #For each surface form (e.g., sIn) find the corresponding stem, suffix pair (e.g., (3, 1))
    for stemtag, suftag in inp_out:
        likelihood = 0
        surface_form = inp_out[(stemtag, suftag)]
        #i.e., if there is only one potential UR for the stemtag
        if stemtag not in URprobs_Normalized:
            URprobs_Normalized[stemtag] = {list(LEX[stemtag].keys())[0]: 1}
        for stem in URprobs_Normalized[stemtag]:
            #i.e., if there is only one potential UR for the suftag
            if suftag not in URprobs_Normalized:
                URprobs_Normalized[suftag] = {list(LEX[suftag].keys())[0]: 1}
            for suf in URprobs_Normalized[suftag]:
                #initialize empty dict for each stem suffix pair
                prob_stem_suf_input_pair = {}
                #candidate_probs is a dict of probabilities of each candidate given the input stem and suf
                #for each stem suffix input pair, the MaxEnt probability for each candidate is stored in candidate_probs
                candidate_probs = probs((stem, suf), WEIGHTS)
                #iterate over every candidate in the generated dictionary of candidates with probabilities
                for candidate in candidate_probs:
                    #if the string of that candidate is in the dict storing probabilities for the current stem,suf input
                    if candidate[0] in prob_stem_suf_input_pair:
                         #assign its probability to that candidate key
                         prob_stem_suf_input_pair[candidate[0]] += candidate_probs[candidate]
                    else:
                        #otherwise add that candidate key and assign its probability
                        prob_stem_suf_input_pair[candidate[0]] = candidate_probs[candidate]
                #if the surfaceform is in prob_stem_suf_input_pair
                #add p(stem)p(suf)p(surfaceform|stem,suf) to likelihood
                if surface_form in prob_stem_suf_input_pair:
                    #prob_surfaceform_wins is probability of surfaceform being the best candidate 
                    #given the current stem, suf input pair/weights
                    prob_surfaceform_wins = prob_stem_suf_input_pair[surface_form]
                    #likelihood of the surfaceform given the current UR probabilities and the surfaform winning
                    #multiply the probability of the stem input times the probability of the suffix input times the probability the surface_form wins
                    #URs with higher probability will increase the likelihood if they result in the surfaceform having a high probability as well. 
                    likelihood += URprobs_Normalized[stemtag][stem] * URprobs_Normalized[suftag][suf] * prob_surfaceform_wins
        #IF likelihood of the winner for using potential UR is super low, print error and mess up the objective
        if likelihood <= 1e-150:
            print(URprobs_Normalized[stemtag])
            print((f"{surface_form} has gotten tooo small"))
            objective_value += 1e500  # Large penalty, but not infinite
        else:
            #add negative log likelihood to objective_value.
            objective_value += -log(likelihood)
            
    #As a failsafe, if some parameter is outside of its bounds, make the objective really big to let the algorithm know it messed up
    for param in parameters:
        if param < negative:
            return float("inf")

    #Return the sum of negative log likelihood with the prior on constraints and morpheme probabilities.
    return objective_value + prior(WEIGHTS) + (sigma * priorl(UR_probabilities))

####Gradient Descent function
#Function to calculate the gradient of the objective function
#input is a single vector with phonotactic_weights as the first 9 indices [0:numcon] and LEXprobs as the remaining inices [numcon:]
#tag_UR_pairs_list called on
#pLEX() function used to get updated probabilities
#inp_out called on to get surface_forms
#candidates() function called to generate candidates for specific surfaceforms
#violations() called to get violation profiles
#harmony() called to get harmony scores
#probs() called
#output is the gradient for each parameter   
#initialize iteration counter to track iteration number during optimization 
gradient_iteration = 0   
def gradient(parameters):
    global gradient_iteration  # Access the global variable
    
    # Increment the iteration count at the beginning of the gradient function
    gradient_iteration += 1
    print(f"Iteration {gradient_iteration}")  # Print the iteration count
    ####PART 1: preamble with parameter variables assigned, gradient list, probability dictionaries and candidate set dictionary initiated.
    grad = []
    WEIGHTS = parameters[:numcon]
    probs_list = parameters[numcon:]
    tag_UR_pairs = Probabilities_and_tag_UR_pairs[1]
    
    URprobs = {}
    for i in range(len(tag_UR_pairs)):
        tag, UR = tag_UR_pairs[i]
        probability = probs_list[i]
        if tag in URprobs:
            URprobs[tag][UR] = probability
        else:
            URprobs[tag] = {UR: probability}
    
    #normalize the dictionary of URprobs
    URprobs_Normalized = pLEX(URprobs)

    candidate_dict = {}
    ####PART 2: Find the gradients for each constraint weight, factoring in the Prior
    #iterate over each constraint and its weight
    for i in range(len(WEIGHTS)):
        #print(f"----Constraint {CONSTRAINTS[i]}----")
        #initiate gradient with value of 0
        gradient_value = 0
        #iterate over each surface form
        for stemtag, suftag in inp_out:
            #assign the surface form
            surface_form = inp_out[(stemtag, suftag)]
            #print(f"----surface form {surface_form}----")
            #variable to store likelihood of observed form given UR
            likelihood = 0
            #ts reflects how the violations of the current constraint by different candidates influence the gradient.
            ts = 0
            #i.e., if there is only one potential UR for the current surface stemtag
            if stemtag not in URprobs_Normalized:
                URprobs_Normalized[stemtag] = {list(LEX[stemtag].keys())[0]: 1.0}
            #i.e., if there is only one potential UR for the current surface suftag, add it to URprobs_Normalized[suftag]
            if suftag not in URprobs_Normalized:
                URprobs_Normalized[suftag] = {list(LEX[suftag].keys())[0]: 1.0}
            #iterate over each possible stem-suffix UR pair, given the current surface form, and compute the likelihood of the surface_form
            #given that stem-suffix UR pair's probability and the harmony score of the surface_form as an output candidate of that pair.
            for stem in URprobs_Normalized[stemtag]:
                for suf in URprobs_Normalized[suftag]:
                    #make each possible stem-suffix UR pair the input
                    input = (f"{stem}{suf}")
                    #print(f"----input {input}----")
                    #print(f"This is the potential input: {input}")
                    #print(f"This is the surface form: {surface_form}")
                    #retrieve candidates for this stem-suf input pair being worked on from candidate_dict if possible. Otherwise, create them with candidates() function.
                    if stem in candidate_dict:
                        if suf in candidate_dict[stem]:
                            candset = candidate_dict[stem][suf]

                        else:
                            temp = candidates((stem, suf))
                            candidate_dict[stem][suf] = temp
                            candset = temp
                    else:
                        temp = candidates((stem, suf))
                        candidate_dict[stem] = {suf: temp}
                        candset = temp
                       # print(f"this is a candset for {input}: {candset}")
                    #candy is a list of candidates without correspondence relations
                    candy = []
                    for candidate in candset:
                        if candidate[0] not in candy:
                            candy.append(candidate[0])
                    cand_MaxEnt_value = 0
                    #accumulates the exponentiated harmony scores (cand_MaxEnt_score) for all candidates in the candidate set
                    
                    
                    #if the surface_form (i.e., the desired winner) is in the list of candidates, calculate its likelihood given the current potential UR
                    if surface_form in candy:
                        candset_MaxEnt_Sum = 0
                        cand_violations = 0
                        #weighted_violations = 0
                        #dividing harmony score by harmony score scaling constant prevents an entire candidate set from having an almost zero Z value
                        #by making harmony scores less large and negative.
                        Harmony_score_scaling_constant = 50
                        for candidate in candset:
                            #harmony score (divided by a scaling constant to ensure MaxEnt scores aren't tiny)
                            cand_harm_score = (harmony((stem, suf), candidate, WEIGHTS)) / Harmony_score_scaling_constant

                            #cand_MaxEnt_score is P*(x) in Hayes & Wilson (2008, p.384)
                            cand_MaxEnt_score = exp(cand_harm_score)

                            #reflects the contribution of all candidates to the expected violation of the i-th constraint.
                            #After iterating over all candidates, provides the total weighted violations for the current constraint
                            cand_MaxEnt_value += vios(stem, suf, candidate)[i] * cand_MaxEnt_score

                            #used to normalize the MaxEnt scores across candidates and ensure that 
                            #the probabilities of all candidates sum to 1.
                            #Essentially, this is the denominator in the MaxEnt model. (Z in Hayes & Wilson (2008, p.384))
                            candset_MaxEnt_Sum += cand_MaxEnt_score
                            
                            #if the string of the current candidate is the surface form
                            if candidate[0] == surface_form:
                                desired_winner_MaxEnt_score = cand_MaxEnt_score
                                #weighted_violations += cand_harm_score * vios(stem, suf, candidate)[i]
                                cand_violations += vios(stem, suf, candidate)[i] #this returns an integer value of the number of violations for constraint i.
                        
                        #define the probability of the desired winner for Constraint i
                        desired_winner_probability = desired_winner_MaxEnt_score / candset_MaxEnt_Sum
                        current_UR_probability = URprobs_Normalized[stemtag][stem] * URprobs_Normalized[suftag][suf]
                        likelihood_observed_data = current_UR_probability * desired_winner_probability

                        #likelihood_observed_data2 is the combined probability of the current stem,suf URs divided by the MaxEnt sum of the candidate set.
                        likelihood_observed_data2 = current_UR_probability  / candset_MaxEnt_Sum
                        
                        likelihood += likelihood_observed_data

                        ts += (cand_MaxEnt_value *  desired_winner_probability - cand_violations) * likelihood_observed_data2

            #if the overall likelihood for a certain surface form is really low, blow up the gradient to make an error
            if likelihood < 1e-150:
                print(f"hmmm {surface_form} {likelihood}")
                gradient_value -= float('inf') 
            else: 
                gradient_value -= ts / likelihood
                
        #find the gradient of the prior and add it to the sum of the gradient of the objective
        wi = WEIGHTS[i]
        if CONTYPE[i] == 'f':
            gradient_value += wi / (SIGMAF**2)
        elif CONTYPE[i] == 'cf':
            gradient_value += (wi - cfbias) / (SIGMACF**2)
        else:
            gradient_value += (wi - mbias) / (SIGMAM**2)
        #penalize negative weights
        if wi < 0:
            gradient_value = -20
        grad.append(gradient_value)
             
    ####PART 3: Find the gradients for each UR probability
    inputprobs = {}
    #iterate over each of the tag, UR pair in tag_UR_pairs: (see probabilities_and_tag_UR_pairs[1])
    for i in range(len(tag_UR_pairs)):
        tag, UR = tag_UR_pairs[i]
        total_tag_prob = 0
        #print(f"tag {tag}")
        for Potential_UR in URprobs_Normalized[tag]:#warning: switched to normalized URprobs_Normalized
            if Potential_UR == UR:
                current_UR_prob = URprobs_Normalized[tag][Potential_UR] #warning: switched to normalized URprobs_Normalized
                #print(f"Calculated UR probability for UR {Potential_UR}: {UR_val}")
            total_tag_prob += URprobs_Normalized[tag][Potential_UR] #warning: switched to normalized URprobs_Normalized
        #print(f"Here is the total probability of tag {tag}: {total_tag_prob}") #run this to ensure they sum to 1
        #calculate the derivative for the current morpheme tag
        #the numerator calculates how much of the total tag probability the current UR's probability is.
            #the smaller the numerator, the smaller the derivative will be of the current potential UR.
        #the denominator should always be 1, so it shouldn't affect things.
        derivative_UR = float(total_tag_prob - current_UR_prob) / (total_tag_prob**2)
        liky = 0
        
        #for each stem suf tag tuple in inp_out (e.g, (3,1))
        for tags in inp_out:
            #if the current tag being iterated over from 20 lines up is in the tuple
            if tag in tags:
                stemtag, suftag = tags
                surface_form = inp_out[tags]
                cand_MaxEnt_value = 0
                cand_MaxEnt_value_2 = 0
                candset_MaxEnt_Sum = 0
                
                #if the current tag is a suffix
                if tag in "0 1".split():
                    ## For each stem paired with this suffix, find the likelihood of the output form given this suffix.
                    for stem in URprobs_Normalized[stemtag]:
                        pinput = (stem, UR)
                        if pinput in inputprobs:
                            outprob = inputprobs[pinput]
                        else:
                            outprob = {}
                            outtemp = probs((stem, UR), WEIGHTS) #store a dictionary of candidates as keys with their probabilities as values
                            for candidate in outtemp:
                                if candidate[0] in outprob:
                                    outprob[candidate[0]] += outtemp[candidate]
                                else:
                                    outprob[candidate[0]] = outtemp[candidate]
                            inputprobs[pinput] = outprob
                        #OUTPROB = sum(eH(surfaceform))/sum(eH(z)) for all surfaceform
                        if surface_form in outprob:
                            pout = outprob[surface_form] 
                        else:
                            pout = 0
                        cand_MaxEnt_value += (URprobs_Normalized[stemtag][stem] * pout)
                        for suf in URprobs_Normalized[suftag]:
                            input = (stem, suf)
                            if input in inputprobs:
                                outsuf = inputprobs[input]
                            else:
                                outsuf = {}
                                outtemp = probs((stem, suf), WEIGHTS)
                                for candidate in outtemp:
                                    if candidate[0] in outsuf:
                                        outsuf[candidate[0]] += outtemp[candidate]
                                    else:
                                        outsuf[candidate[0]] = outtemp[candidate]
                                inputprobs[input] = outsuf
                            if surface_form in outsuf:
                                candset_MaxEnt_Sum += URprobs_Normalized[stemtag][stem] * URprobs_Normalized[suftag][suf] * outsuf[surface_form]
                                if suf != UR:
                                    cand_MaxEnt_value_2 += URprobs_Normalized[stemtag][stem] * outsuf[surface_form] * URprobs[suftag][suf] / total_tag_prob#warning:nonnormalized URprobs used
                
                #if the current tag is a stem: do the same thing you did with suffixes above
                else:
                    #For each suffix paired with this stem, find the likelihood of the output form given this stem.
                    for suf in URprobs_Normalized[suftag]:
                        sinput = (UR, suf)
                        if sinput in inputprobs:
                            outprob = inputprobs[sinput]
                        else:
                            outprob = {}
                            outtemp = probs((UR, suf), WEIGHTS) #store a dictionary of candidates as keys with their probabilities as values
                            for candidate in outtemp:
                                if candidate[0] in outprob:
                                    outprob[candidate[0]] += outtemp[candidate]
                                else:
                                    outprob[candidate[0]] = outtemp[candidate]
                            inputprobs[sinput] = outprob
                        #OUTPROB = sum(eH(surfaceform))/sum(eH(z)) for all surfaceform
                        if surface_form in outprob:
                            pout = outprob[surface_form] 
                        else:
                            pout = 0
                        cand_MaxEnt_value += (URprobs_Normalized[suftag][suf] * pout)
                        for stem in URprobs_Normalized[stemtag]:
                            input = (stem, suf)
                            if input in inputprobs:
                                outsuf = inputprobs[input]  
                            else:
                                outsuf = {}
                                outtemp = probs((stem, suf), WEIGHTS)
                                for candidate in outtemp:
                                    if candidate[0] in outsuf:
                                        outsuf[candidate[0]] += outtemp[candidate]
                                    else:
                                        outsuf[candidate[0]] = outtemp[candidate]
                                inputprobs[input] = outsuf
                            if surface_form in outsuf:
                                candset_MaxEnt_Sum += URprobs_Normalized[stemtag][stem] * URprobs_Normalized[suftag][suf] * outsuf[surface_form]
                                if stem != UR:
                                    cand_MaxEnt_value_2 += URprobs_Normalized[suftag][suf] * outsuf[surface_form] * URprobs[stemtag][stem] / total_tag_prob#warning:nonnormalized URprobs used
                
                if candset_MaxEnt_Sum > 1e-50:
                    liky += (derivative_UR * cand_MaxEnt_value - cand_MaxEnt_value_2) / candset_MaxEnt_Sum
        
        gradi = (-liky) + ((2 * current_UR_prob) * sigma)
        grad.append(gradi)
        
    printprob(parameters)
    
    print("------CONSTRAINTS------")
    printconstraints(parameters[:numcon])
    
    print("------LEXICON------")
    printdict(URprobs_Normalized)
    
    print(f"gradient	{np.linalg.norm(grad)}")
    print(f" objective {objective(parameters)}")
    
    #print("\n---GRADIENT VALUES----")
    #print(grad)
    return np.asarray(grad)


#printliky function
# Given a dictionary `d`, print those forms that have received less than 0.9 probability.
def printliky(dictionary):
    y = 0
    string = ""
    # Check if any value in the dictionary is less than 0.9
    if any(v < 0.9 for v in dictionary.values()):
        print("---SURFACE FORMS WITH LESS THAN 0.9 PROBABILITY----")
    
    for tag in dictionary:
        if dictionary[tag] < .9:
            if len(tag) < 8:
                string += f"\t{tag}\t\t{dictionary[tag]:.2f},"
            else:
                string += f"\t{tag}\t{dictionary[tag]:.2f},"
            if y%3 == 2:
                print(string)
                string = ""
            y += 1
    print(string)

#function to return a vector of probabilities from a dictionary containing them 
def paramToVect(UR_probabilities_list, all_tag_UR_pairs, param_tag_UR_pairs):
    probabilities_vector = []
    
    #iterate over each morpheme tag-Potential UR pair in the Lexicon
    for i in range(len(all_tag_UR_pairs)): #e.g. (2, 'si')
        #make a tuple with the structure (tag, Potential_UR)
        tag_UR_tuple = (all_tag_UR_pairs[i][0], all_tag_UR_pairs[i][1])
        #if that tuple is in the list of tuples that are used as parameters in the optimization (i.e., if they have multiple Potential URs)
        if tag_UR_tuple in param_tag_UR_pairs:
            #assign its probability from the result of the optimization to a variable
            probability = UR_probabilities_list[param_tag_UR_pairs.index(tag_UR_tuple)]
            #and append it to the final probabilities vector
            probabilities_vector.append(probability)
        else:
            #if there is only one potential UR, give it a probability of 1
            probabilities_vector.append(1.0)
    
    return probabilities_vector
    
#function to print the probability of each output form given the grammar
def printprob(parameter):
    ew1 = parameter[:numcon]
    lprevect = paramToVect(parameter[numcon:], tag_UR_pairs_list, Probabilities_and_tag_UR_pairs[1])#paramtovect and paramlex
    URfreqs = {}
    for i in range(len(tag_UR_pairs_list)):
        tag, UR = tag_UR_pairs_list[i]
        probability = lprevect[i]
        if tag in URfreqs:
            URfreqs[tag][UR] = probability
        else:
            URfreqs[tag] = {UR: probability}
    Normalized_probs = pLEX(URfreqs)
    likylist = {}
    for stem_suf_pair in inp_out:
        prob = {}
		#print y
        lstem = stem_suf_pair[0]
        lsuf = stem_suf_pair[1] #warning: removed int() from wrapping this
		#print LEX
        for stem in LEX[lstem]:
            for suf in LEX[lsuf]:
                pbr = Normalized_probs[lstem][stem] * Normalized_probs[lsuf][suf]
                #input = (f"{stem}{suf}")
                proby = probs((stem, suf), ew1)
                for yum in proby:
                    x = yum[0]
                    if x in prob:
                        prob[x] += pbr * proby[yum]
                    else:
                        prob[x] = pbr * proby[yum]
        likylist[inp_out[stem_suf_pair]] = prob[inp_out[stem_suf_pair]]
    printliky(likylist)
    
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

#function printdict: Given a dictionary of probabilities of URs for each morpheme, print them (similar dict structure to LEX)
def printdict(dictionary):
    #iterate over each morpheme tag in the dictionary
    for tag in dictionary:
        #if the number of potential URs for the morpheme tag is greater than 1
        if len(dictionary[tag]) > 1: #warning: turned this off
            #create an empty string variable
            string = ""
            #for each potential UR, print the UR and its current probability value.
            for potential_UR in dictionary[tag]:
                if len(potential_UR) < 8:
                    string += (f"\t{potential_UR}\t\t{dictionary[tag][potential_UR]:.2f},")
                else:
                    string += (f"\t{potential_UR}\t{dictionary[tag][potential_UR]:.2f},")
            print(string)   

gradient_params = wToParam(phonotactic_weights, Probabilities_and_tag_UR_pairs[0], Probabilities_and_tag_UR_pairs[1])

####bounds
#Create bounds for the optimize algorithm (don't let constraints or frequencies get negative or weights get above 200)
wbound = (0, 100)
weight_bounds = [wbound] * (numcon)
#get length of URs in parameters by subtracting number of constraints from total parameter count
lexlen = len(gradient_params[0]) - len(phonotactic_weights)
probability_bounds = [(0, 1)] * lexlen #warning change 1 to None per Ohara
bigbound = weight_bounds + probability_bounds


#Run the SLSQP optimize algorithm on the objective function using the gradient function as a jacobian.
res=scipy.optimize.minimize(objective,np.asarray(gradient_params[0]), method='SLSQP',bounds=bigbound,jac=gradient,
                            options={'maxiter':20000,'ftol':1e-50, 'eps':1e-5, 'disp': True})

w1 = res.x


final_weights = w1[:numcon]

Final_UR_probabilities = paramToVect(w1[numcon:], tag_UR_pairs_list, Probabilities_and_tag_UR_pairs[1])

#lexvectout = []
#for UR_prob in Final_UR_probabilities:
    #lexvectout.append(UR_prob)

URprobs = {}
for i in range(len(tag_UR_pairs_list)):
    tag, UR = tag_UR_pairs_list[i]
    probability = Final_UR_probabilities[i]
    if tag in URprobs:
        URprobs[tag][UR] = probability
    else:
        URprobs[tag] = {UR: probability}

URprobs_Normalized = pLEX(URprobs)

lprobout = []
for i in range(len(tag_UR_pairs_list)):
    tag, UR = tag_UR_pairs_list[i]
    lprobout.append(URprobs_Normalized[tag][UR])

final_lexicon = lprobout

#Print the results of the simulation.
print("\nOPTIMIZED CONSTRAINT WEIGHTINGS:")
for weight in range(len(final_weights)):
	print(f"{CONSTRAINTS[weight]}: {final_weights[weight]}")
    
print("\nOPTIMIZED LEXICON PROBABILITIES:")
for UR_prob in range(len(final_lexicon)):
	print(f"{tag_UR_pairs_list[UR_prob]}: {final_lexicon[UR_prob]}")

#oragnize and print the URs for each morpheme, sorting by probability
lexyprobs = {}
for i in range(len(tag_UR_pairs_list)):
    tag = tag_UR_pairs_list[i][0]
    UR = tag_UR_pairs_list[i][1]
    if tag in lexyprobs:
        lexyprobs[tag][UR] = final_lexicon[i]
    else:
        lexyprobs[tag] = {UR: final_lexicon[i]}
        


#Calculate the Probability of each output form given its morpheme tags.
prod = 1
print("----SURFACE FORM PROBABILITIES")
for stem_suf_tag in inp_out:
    prob = {}
    stem_tag = stem_suf_tag[0]
    suf_tag = stem_suf_tag[1]
    
    for stem in LEX[stem_tag]:
        for suf in LEX[suf_tag]:
            pbr = lexyprobs[stem_tag][stem] * lexyprobs[suf_tag][suf]
            input = (f"{stem}{suf}")
            prob_stem_suf_tag = probs((stem, suf), final_weights)
            for yum in prob_stem_suf_tag:
                x = yum[0]
                if x in prob:
                    prob[x] += pbr * prob_stem_suf_tag[yum]
                else:
                    prob[x] = pbr * prob_stem_suf_tag[yum]
    sorted_prob = [(k,prob[k]) for v,k in sorted([(v,k) for k,v in prob.items()],reverse=True)]
    
    prod *= prob[inp_out[stem_suf_tag]]
    #Print a warning if the output form is not produced with .9 probability, or is not the preferred output.
    if prob[inp_out[stem_suf_tag]] < .9:
        for stem in LEX[stem_tag]:
            for suf in LEX[suf_tag]:
                input = (f"{stem}{suf}")
                prob_inp = probs((stem, suf), final_weights)
                sorts = [(k,prob_inp[k]) for v,k in sorted([(v,k) for k,v in prob_inp.items()],reverse=True)]
        if sorted_prob[0][1] != prob[inp_out[stem_suf_tag]]:
            print(f"\n{inp_out[stem_suf_tag]} does not win:\t{sorted_prob[:]}")
    
    print(f"\n{inp_out[stem_suf_tag]} final surfacing probability:\t{sorted_prob[:]}")
    #print(f"\n{inp_out[stem_suf_tag]} final surfacing probabilities:\t{', '.join([f'{prob:.2f}' for prob in sorted_prob])}")
            
#print overall probability of data
print(f"Probability of the Data is {prod}")

abstract_weights = [0.07, 100.00, 5.42, 100.00, 0.02, 100.00, 100.00, 0.00, 100.00]