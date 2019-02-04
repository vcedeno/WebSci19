#!/usr/bin/env python
### V. Cedeno
### 09 October 2018.
### ABM
###     
### output files:
###      simulation output
###     


### Modification History
### --------------------

import sys
import json
import os
import csv
import operator
from datetime import time, datetime, timedelta

sys.path.insert(0, os.getcwd()+'/src/h1')

import h1

## ------------------------------
def returnJson(numlines,jreader,jname,rowname,rowid,returnrowid):
    """
    	ireader: csv file
        session_code: the session_code we want the data from.
    """
    data=[]
    for i in range(numlines):
    	#print(jreader[jname][i][rowname])
    	#print(jreader["CompletedSessionSummary"][i]["experiment_type"])
    	if jreader[jname][i][rowname]==rowid:
    		if jreader[jname][i][returnrowid] not in data:
    			data.append(jreader[jname][i][returnrowid])
    return data

### -----------------------------
### init_players.
def init_players(nodes):
	players = [[] for i in range(nodes)]
	return players
	
### -----------------------------
### Start.
def main():

    if (len(sys.argv) != 2):
    	print ("  Error.  Incorrect usage.")
    	print ("  usage: exec infile outfile.")
    	print ("  Halt.")
    	quit()
    
    configfile=sys.argv[1]
    json_file = open(configfile, 'r')
    json_data = json.load(json_file)
    numlines= (len(json_data))
    #for i in range(numlines):
    
    #experimentfile=os.getcwd()+json_data["experiment"]
    functions=json_data["functions"]
    numlines= (len(functions))
    
    for i in range(numlines):
    	function=functions[i]["function"]
    	if function=='h1':
    		graphfile=functions[i]["graphfile"]
    		beta=functions[i]["beta"]
    		iteration=functions[i]["iteration"]
    		nodes=functions[i]["nodes"]
    		edges=functions[i]["edges"]
    		
    		word_dist=functions[i]["word_dist"]
    		letter_req_dist=functions[i]["letter_req_dist"]
    		frac_rpl_dist=functions[i]["frac_rpl_dist"]
    		time_rpl_dist=functions[i]["time_rpl_dist"]
    		
    		timeSeconds=functions[i]["timeSeconds"]
    		seed_value=functions[i]["seed_value"]
    		experiment_id=functions[i]["experiment_id"]
    		players=functions[i]["players"]
    		players_info=init_players(nodes)
       			
    		numlinesp=(len(players))
    		for j in range(nodes):
    			agent_id=players[j]["agent_id"]
    			corpus=players[j]["corpus"]
    			corpus_name=players[j]["corpus_name"]
    			vocabulary=players[j]["vocabulary"]
    			aptitudeWord=players[j]["aptitudeWord"]
    			aptitudeLetterRequest=players[j]["aptitudeLetterRequest"]
    			letterReplyType=players[j]["letterReplyType"]
    			initialLetters=players[j]["initialLetters"]
    			players_info[agent_id].append(corpus)
    			players_info[agent_id].append(vocabulary)
    			players_info[agent_id].append(aptitudeWord)
    			players_info[agent_id].append(aptitudeLetterRequest)
    			players_info[agent_id].append(letterReplyType)
    			players_info[agent_id].append(initialLetters)
    			players_info[agent_id].append(corpus_name)
    		#print(players_info)
    		h1.main(graphfile,beta,nodes,edges,timeSeconds,seed_value,iteration,players_info,word_dist,letter_req_dist,frac_rpl_dist,time_rpl_dist,experiment_id)

    endTime=datetime.now()

    #print (" elapsed time (seconds): ",endTime-startTime)
    #print (" elapsed time (hours): ",(endTime-startTime)/3600.0)

    print (" -- good termination --")

    return	

## --------------------------
## Execution starts.
if __name__ == '__main__':
    main()
    #print (" -- good termination from main --")
