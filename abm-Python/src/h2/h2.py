import sys
import os
import csv
import json
import numpy as np
import re
from datetime import time, datetime, timedelta
from numpy import * 
import subprocess
import pandas as pd
import random 
  
### -----------------------------
### output.
def output(matrix_states,graphfile,iteration,payload,iterationA):
	s1,s2=graphfile.split('.txt')
	graphname=s1.split('/')[-1]
	if not os.path.exists(os.getcwd()+'/test/results/h2'):
		os.makedirs(os.getcwd()+'/test/results/h2')
	if not os.path.exists(os.getcwd()+'/test/results/h2/output'):
		os.makedirs(os.getcwd()+'/test/results/h2/output')
	if not os.path.exists(os.getcwd()+'/test/results/h2/output/'+graphname):
		os.makedirs(os.getcwd()+'/test/results/h2/output/'+graphname)
	with open(os.getcwd()+'/test/results/h2/output/'+graphname+'/'+str(iteration)+'.txt','wb') as f:
		matrix_states=matrix_states.astype(object)
		matrix_states=np.column_stack((iterationA,matrix_states,payload))
		np.savetxt(f, matrix_states, fmt='%s', delimiter=" ")
	
### -----------------------------
### init_buffer_request_sent.
def init_buffer_request_sent(nodes):
	buffer_request_sent = [[] for i in range(nodes)]
	return buffer_request_sent

### -----------------------------
### init_buffer_request_received.
def init_buffer_request_received(nodes):
	buffer_request_received = [[] for i in range(nodes)]
	return buffer_request_received

### -----------------------------
### init_request_state.
def init_request_state(nodes):
	request_state = [[] for i in range(nodes)]
	return request_state
			
### -----------------------------
### init_initial_letters.
def init_initial_letters(nodes,matrix_states,numiL):
	for i in range(0,nodes):
		matrix_states[i]=[matrix_states[i][0],matrix_states[i][1],matrix_states[i][2],numiL,matrix_states[i][4],matrix_states[i][5],matrix_states[i][6],matrix_states[i][7],matrix_states[i][8],matrix_states[i][9],matrix_states[i][10]]
	return matrix_states

### -----------------------------
### init_buffer_letters_in_hand.
def init_buffer_letters_in_hand(nodes,players_info):
	buffer_letters_in_hand = [[] for i in range(nodes)]
	for i in range(nodes):
		letters=list(players_info[i][5])
		for row in letters:
			buffer_letters_in_hand[i].append(row.lower())
	return buffer_letters_in_hand
	
### -----------------------------
### init_buffer_available_letters.
def init_buffer_available_letters(nodes,graph,players_info):
	buffer_available_letters = [[] for i in range(nodes)]
	for i in range(nodes):
		for row in graph:
			if int(row[0])==i:
				letters=list(players_info[int(row[1])][5])
				for letter in letters:
					buffer_available_letters[i].append([row[1],letter])
			if int(row[1])==i:
				letters=list(players_info[int(row[0])][5])
				for letter in letters:
					buffer_available_letters[i].append([row[0],letter])
	#print(buffer_available_letters)
	return buffer_available_letters

### -----------------------------
### previous_word.
def init_previous_word(nodes):
	previous_word = [[] for i in range(nodes)]
	return previous_word

### -----------------------------
### buffer_form_words.
def init_buffer_form_words(nodes):
	buffer_form_words = [[] for i in range(nodes)]
	return buffer_form_words
				
### -----------------------------
### init_matrix
def init_matrix(nodes,timeSeconds,num_states):
	timeA=[]
	agentA=[]
	stateA=[]
	xL=[]
	xB=[]
	xW=[]
	xC=[]
	xRequestS=[]
	xReplyR=[]
	xRequestR=[]
	xReplyS=[]
	stateA=np.zeros(nodes*timeSeconds)
	xL=np.zeros(nodes*timeSeconds)
	xB=np.zeros(nodes*timeSeconds)
	xW=np.zeros(nodes*timeSeconds)
	xC=np.zeros(nodes*timeSeconds)
	xRequestS=np.zeros(nodes*timeSeconds)
	xReplyR=np.zeros(nodes*timeSeconds)
	xRequestR=np.zeros(nodes*timeSeconds)
	xReplyS=np.zeros(nodes*timeSeconds)
	
	for i in range(0,timeSeconds):
		for j in range(0,nodes):
			timeA.append(i)
			agentA.append(j)
			#xL.append(numiL)
	matrix_states=np.column_stack((timeA,agentA,stateA,xL,xB,xW,xC,xRequestS,xReplyR,xRequestR,xReplyS))
	matrix_states=matrix_states.astype(int)
	#matrix_states=array(matrix_states)
	#matrix_states=matrix.reshape(-1,num_states)
	return matrix_states

### -----------------------------
### init_parameters.
def init_beta(beta,num_discrete_states):
	par=[]
	parmatrix=[]
	with open(os.getcwd()+beta, 'r') as f:
		graph_file = f.readlines()
	i=0
	for row in graph_file:
		if i==4:
			parmatrix.append(par)
			par=[]
			i=0
		para=row.strip('\n')
		par.append(para.split(' '))
		i=i+1
	parmatrix.append(par)
	return parmatrix
		
### -----------------------------
### init_graph.
def init_graph(graphfile):
	graph=[]
	with open(os.getcwd()+graphfile, 'r') as f:
		graph_file = f.readlines()
	for row in graph_file:
		edge=row.strip('\n')
		edge1,edge2=edge.split(' ')
		graph.append([edge1,edge2])
	return graph
 
### -----------------------------
### get_next_state.
def get_next_state(nodes,states,beta_mtx,num_discrete_states):
	#print(states)
	#print(beta_mtx)
	x=[1,states[4],states[3],states[5],states[6]]
	bx=[None] * num_discrete_states
	#print(bx)
	bx[0]=1
	norm=1
	for i in range(1,num_discrete_states):
		tot=0
		for j in range(0,len(x)):
			tot=tot+float(beta_mtx[i][j])*x[j];
		bx[i]=np.exp(tot)
		norm=norm+bx[i]
	#print(bx)
	for index,p in enumerate(bx):
		bx[index]=p/norm
	statenum=np.random.choice(num_discrete_states, 1, p=bx)[0]
	#print(bx)
	return statenum

### -----------------------------
### buffer_vocabulary.
def init_buffer_vocabulary(nodes,players_info):
	buffer_vocabulary = [[] for i in range(nodes)]
	for i in range(nodes):
		with open(os.getcwd()+players_info[i][0], 'r') as f:
			all_words_file = f.readlines()[0:players_info[i][1]]
			rowword=[]
			for row in all_words_file:
				rowword.append(row.strip('\n').lower())
			buffer_vocabulary[i].append(rowword)
	return buffer_vocabulary
	
### -----------------------------
### is_allowed_specific_char.	
def is_allowed_specific_char(letters,word):
    charRe = re.compile(r'[^%s]' % letters)
    string = charRe.search(word)
    return not bool(string)

### -----------------------------
### wpn_list.
def wpn_list(buffer_letters_in_hand,buffer_form_words,buffer_vocabulary,numiL):	 
	wpn=[]
	allLettersRecLS="".join(str(x) for x in buffer_letters_in_hand)
	for rowword in buffer_vocabulary[0]:
		#print(rowword)
		if rowword.isalpha()==True and allLettersRecLS.isalpha()==True and len(rowword)>=numiL and rowword not in buffer_form_words:
			if is_allowed_specific_char(allLettersRecLS,rowword)==True:				
				wpn.append(rowword)
	return wpn

### -----------------------------
### read_distribution.
def read_distribution(dist_file):
	distribution=[]
	f = open(os.getcwd()+dist_file)
	csv_f = csv.reader(f)
	
	for row in csv_f:
		distribution.append(row)
	distlen=len(distribution[0])
	return distribution,distlen
	
### -----------------------------
### requestLetterFromNbr.
def requestLetterFromNbr(agent_id,aptitudeLetterRequest,corpus,vocabulary,buffer_available_letters,buffer_letters_in_hand,buffer_form_words,buffer_vocabulary,numiL,fracRequest,i_letter_req_dist_len,letter_req_dist_len,letter_req_dist_data,payload_v):	 
	boolletter=0
	letter="0"
	#print(agent_id)
	aptitude=["20p","40p","60p","80p","100p"]
	#print(buffer_letters_in_hand)
	#print(buffer_available_letters)
	#print("payload_v",payload_v)
	#flag=0
	#if len(payload_v)==1:
	#	for row in buffer_available_letters:
	#		if row[1]==payload_v:
	#			flag=1
	#	if flag==0:
	#		buffer_available_letters.append([9999,payload_v])
	#print("buffer_available_letters",buffer_available_letters)	
	rankp=["rank10P","rank20P","rank30P","rank40P","rank50P""rank60P","rank70P","rank80P","rank90P","rank100P"]
	if len(buffer_available_letters)>0:
		wpn=wpn_list(buffer_letters_in_hand,buffer_form_words,buffer_vocabulary,numiL)
		w1=len(wpn)
		letterlist=[]
		for row in buffer_available_letters:
			buffer_letters_in_hand.append(row[1])
			wpnl=wpn_list(buffer_letters_in_hand,buffer_form_words,buffer_vocabulary,numiL)
			w1l=len(wpnl)
			letterlist.append([row[0],row[1],w1l])
			buffer_letters_in_hand.remove(row[1])
		df = pd.DataFrame(letterlist,columns=['player','letter','wihl'])
		df['rank']=df['wihl'].rank(method='average',ascending=0).astype(int)
		ranksdf=pd.unique(df['rank'])
		ranksletters=ranksdf.tolist()	
		
		#print(df)
		#print(payload_v.lower())
		#print(buffer_letters_in_hand)
		
		#containL=df[df['letter'].str.contains('z')]
		exp_rank=df.loc[df['letter'] == payload_v,'rank'].iloc[0]
		
		null_rank=''
		if len(letterlist)>0:
			null_rank=str(random.randint(1,len(letterlist)))
		
		#request actual LR
		if fracRequest>=0 and fracRequest<=0.1:
			rank_percentage="rank10P"
		if fracRequest>0.1 and fracRequest<=0.2:
			rank_percentage="rank20P"
		if fracRequest>0.2 and fracRequest<=0.3:
			rank_percentage="rank30P"
		if fracRequest>0.3 and fracRequest<=0.4:
			rank_percentage="rank40P"
		if fracRequest>0.4 and fracRequest<=0.5:
			rank_percentage="rank50P"
		if fracRequest>0.5 and fracRequest<=0.6:
			rank_percentage="rank60P"
		if fracRequest>0.6 and fracRequest<=0.7:
			rank_percentage="rank70P"
		if fracRequest>0.7 and fracRequest<=0.8:
			rank_percentage="rank80P"
		if fracRequest>0.8 and fracRequest<=0.9:
			rank_percentage="rank90P"
		if fracRequest>0.9 and fracRequest<=1:
			rank_percentage="rank100P"
		lractual=0
		elements = range(letter_req_dist_len-i_letter_req_dist_len)
		initaptitudeLetterRequest=aptitudeLetterRequest
		i=1
		#print(corpus)
		#print(fracRequest)
		#print(rank_percentage)
		#print(aptitude[aptitudeLetterRequest])
		#print("lractualb",lractual)
		while lractual==0:
			for row in letter_req_dist_data:
				#print(row)
				#print(corpus)
				#print(aptitude[aptitudeLetterRequest])
				#print(rank_percentage)
				if row[1]==corpus and row[2]==aptitude[aptitudeLetterRequest] and row[3]==rank_percentage:
					prob=row[i_letter_req_dist_len:letter_req_dist_len]
					#print("number",np.random.choice(elements,1,p=prob))
					lractual=np.random.choice(elements,1,p=prob)[0]+1
					
			if lractual==0 and aptitudeLetterRequest>0:
				aptitudeLetterRequest=aptitudeLetterRequest-i
				for row in letter_req_dist_data:
					if row[1]==corpus and row[2]==aptitude[aptitudeLetterRequest] and row[3]==rank_percentage:
						prob=row[i_letter_req_dist_len:letter_req_dist_len]
						#print("number",np.random.choice(elements,1,p=prob))
						lractual=np.random.choice(elements,1,p=prob)[0]+1
			if lractual==0:
				aptitudeLetterRequest=aptitudeLetterRequest+i
			i=i+1
		i=1
		#print("lractuala",lractual)
		initactual=lractual
		while(lractual not in ranksletters and lractual>=min(ranksletters) and lractual<=max(ranksletters)):
			lractual=initactual-i
			if(lractual not in ranksletters):
				lractual=initactual+i
			i=i+1
			
		#print(df)
		#print("lractual",lractual)
		#print("oi",df[['player','letter','rank']].values.tolist())
		lrlist=df[['player','letter','rank']].values.tolist()
		lrletters=[]
		for row in lrlist:
			if row[2]==lractual:
				lrletters.append([row[0],row[1]])
		#lrletters=df.loc[df['rank'] == lractual,['player','letter']].iloc[0].tolist()	
		#print(agent_id)
		if len(lrletters)>0:
			boolletter=1
			randoml=random.randint(0, (len(lrletters)-1))
			letter=lrletters[randoml]	
	return [boolletter,letter,lractual,exp_rank,null_rank]	
	  
### -----------------------------
### formWord.
def formWord(makepath,agent_id,aptitudeWord,corpus,vocabulary,buffer_letters_in_hand,buffer_form_words,previous_word,buffer_vocabulary,numiL,word_dist_data,i_word_dist_len,word_dist_len,payload_v):	 
	ldname=["LD1","LD2","LD3","LDother",]
	aptitude=["20p","40p","60p","80p","100p"]
	boolword=0
	word="0"
	exepath='cd '+makepath+';'
	#print("buffer_letters_in_hand",buffer_letters_in_hand)
	#print("buffer_form_words")
	wpn=wpn_list(buffer_letters_in_hand,buffer_form_words,buffer_vocabulary,numiL)
	len_wpn=len(wpn)
	rand_wpn=''
	#print("wpn",wpn)
	if len_wpn>0:
		rand_wpn=wpn[random.randint(0, (len_wpn-1))]
	if not previous_word:  
		if len(wpn)>0:
			boolword=1
			word=wpn[0]	
	else:
		pword=previous_word
		localranklist=[]
		for wordrow in wpn:
			ldwords=[]
			ldlist=[]
			if len(wpn)>0:
				minld=9999
				i=0
				while i<len(wpn):
					arguments= ' '+str(pword) + ' '+str(wpn[i])
					ldvalue=subprocess.call(exepath+' ./main'+arguments,shell=True)
					ldwords.append([wpn[i],ldvalue])
					if ldvalue not in ldlist:
						ldlist.append(ldvalue)
					if ldvalue<minld:
						minld=ldvalue
					i=i+1
				#print("agentid",agent_id)
				#print("entre",ldwords)
				##Actual LD distribution
				if minld<4:
					ldname_value=ldname[minld-1]
				else:
					ldname_value=ldname[3]
				ldactual=0
				elements = range(word_dist_len-i_word_dist_len)
				initaptitudeWord=aptitudeWord
				i=1
				while ldactual==0:
					for row in word_dist_data:
						if row[1]==corpus and row[2]==aptitude[aptitudeWord] and row[3]==ldname_value:
							prob=row[i_word_dist_len:word_dist_len]
							ldactual=np.random.choice(elements,1,p=prob)[0]+1
					if ldactual==0 and aptitudeWord>0:
						aptitudeWord=aptitudeWord-i
						for row in word_dist_data:
							if row[1]==corpus and row[2]==aptitude[aptitudeWord] and row[3]==ldname_value:
								prob=row[i_word_dist_len:word_dist_len]
								ldactual=np.random.choice(elements,1,p=prob)[0]+1
					if ldactual==0:
						aptitudeWord=aptitudeWord+i
					i=i+1
							
				#ldactual=1
				i=1
				initactual=ldactual
				while(ldactual not in ldlist and ldactual>=min(ldlist) and ldactual<=max(ldlist)):
					ldactual=initactual-i
					if(ldactual not in ldlist):
						ldactual=initactual+i
					i=i+1
				#print("wpn",wpn)
				#print("minld",minld)
				#print("ldactual",ldactual)
				ldactualwords=[]
				for row in ldwords:
					if row[1]==ldactual:
						ldactualwords.append(row[0])
				if len(ldactualwords)>0:
					boolword=1
					word=ldactualwords[0]	
				#print("ldactualwords",ldactualwords)	
	ld_exp_word=''
	ld_opt_word=''
	ld_null_word=''
	#print("previous_word",previous_word)
	#print("exp_wordpayload_v:",payload_v)
	#print("opt_word:",word)
	#print("rand_wpn:",rand_wpn)
	#print(previous_word)
	if previous_word:
		pword=previous_word
		arguments= ' '+str(pword) + ' '+str(word)
		ld_opt_word=str(subprocess.call(exepath+' ./main'+arguments,shell=True))
		arguments= ' '+str(pword) + ' '+str(payload_v)
		ld_exp_word=str(subprocess.call(exepath+' ./main'+arguments,shell=True))
		arguments= ' '+str(pword) + ' '+str(rand_wpn)
		ld_null_word=str(subprocess.call(exepath+' ./main'+arguments,shell=True))	
	#print(ld_opt_word)
	#print(ld_exp_word)
	#print(ld_null_word)
	return [boolword,word,payload_v,rand_wpn,ld_opt_word,ld_exp_word,ld_null_word]	
 
## ------------------------------
def returnJson(numlines,jreader,rowname,phaseid,windowsize,begin,players_info):
    """
    	jreader: json file
    """
    data=[]
    begin=begin+1
    for i in range(numlines):
    	if jreader[i][rowname]== phaseid:
    		actions=jreader[i]["actionlist"]
    		for index,row in enumerate(actions):
    			data.append([actions[index]["player1"],actions[index]["player2"],actions[index]["actionid"],actions[index]["playerActionSeqid"],actions[index]["timestamp"],actions[index]["payload"]])
    		
    		playersactions=[]
    		players=[]
    		for player in players_info:
    			players.append(player[7])
    		for index,player in enumerate(players):
    			allactions=[]
    			for row in data:
    				for index2,player2 in enumerate(players):
    					if player2==row[0]:
    						player_id_0=index2
    				for index2,player2 in enumerate(players):
    					if player2==row[1]:
    						player_id_1=index2
    				#if row[1]==player and row[2]=="requestsent":
    				#	allactions.append(['requestReceived',int(float(row[4])-float(begin)),row[5],player_id_0])
    				if row[0]==player and row[2]=="replysent":
    					allactions.append(['replySent',int(float(row[4])-float(begin)),row[5].lower(),player_id_1])
    				#if row[1]==player and row[2]=="replysent":
    				#	allactions.append(['replyReceived',int(float(row[4])-float(begin)),row[5],player_id_0])
    				if row[0]==player and row[2]=="requestsent":
    					allactions.append(['requestSent',int(float(row[4])-float(begin)),row[5].lower(),player_id_1])
    				if row[0]==player and row[2]=="word":
    					allactions.append(['word',int(float(row[4])-float(begin)),row[5].lower(),''])

    			allactions=sorted(allactions,key=lambda x: x[1])
    			#print(allactions)
    			playersactions.append([player,allactions])
    return playersactions
    
### -----------------------------
### Start.
def main(graphfile,beta,nodes,edges,timeSeconds,seed_value,iteration,players_info,word_dist,letter_req_dist,frac_rpl_dist,time_rpl_dist,experiment_id,actionfile,phasefile):
			
	s1,s2=graphfile.split('.txt')
	graphname=s1.split('/')[-1]
	if not os.path.exists(os.getcwd()+'/test/results/h2'):
		os.makedirs(os.getcwd()+'/test/results/h2')
	if not os.path.exists(os.getcwd()+'/test/results/h2/output'):
		os.makedirs(os.getcwd()+'/test/results/h2/output')
	if not os.path.exists(os.getcwd()+'/test/results/h2/output/'+graphname):
		os.makedirs(os.getcwd()+'/test/results/h2/output/'+graphname)
	
	output_file=open(os.getcwd()+'/test/results/h2/output/'+graphname+'/'+str(experiment_id)+'.txt','wb')
	csvfile = open(os.getcwd()+'/test/results/h2/output/'+graphname+'/'+str(experiment_id)+'.csv', 'w+')
	csvfile.write('iteration,player,time,idle,letterRequest,letterReply,wordFormed,letterReplyReceived,letterRequestReceived,expAction,expPayload,expPlayerId,null_word,opt_rank,exp_rank,null_rank,opt_replies,exp_replies,null_replies,ld_opt_word,ld_exp_word,ld_null_word\n')

	numite=nodes*timeSeconds
	np.random.seed(seed_value)
	buffer_vocabulary=init_buffer_vocabulary(nodes,players_info)

	word_dist_data,word_dist_len=read_distribution(word_dist)
	i_word_dist_len=4
	letter_req_dist_data,letter_req_dist_len=read_distribution(letter_req_dist)
	i_letter_req_dist_len=4
	frac_rpl_dist_data,frac_rpl_dist_len=read_distribution(frac_rpl_dist)
	i_frac_rpl_dist_len=2
	time_rpl_dist_data,time_rpl_dist_len=read_distribution(time_rpl_dist)
	i_time_rpl_dist_len=3
		
	makepath=os.getcwd()+'/src/h2'
	subprocess.Popen(["make"], cwd=makepath)
	#print(buffer_vocabulary)
	
	for ite in range(0,iteration):
		jsonfilename = phasefile
		json_phase = open(jsonfilename, 'r')
		phase_data = json.load(json_phase)
		numlinesphase= len(phase_data)		

		for i in range(numlinesphase):
			if phase_data[i]["phaseid"] ==experiment_id:
				begin_time=phase_data[i]["begin"]
	
		jsonfilename = actionfile
		json_action = open(jsonfilename, 'r')
		action_data = json.load(json_action)
		numlinesaction= len(action_data)
		#phasesids=["5ydhsfg61"]
		#print(phasesids)
		player_actions=returnJson(numlinesaction,action_data,"phaseid",experiment_id,timeSeconds,begin_time,players_info)
		#print(player_actions_read)
	
		#print(player_actions)
		payload=[]
		payload=['Na']*(nodes*timeSeconds)
		iterationA=[ite]*(nodes*timeSeconds)
		graph=init_graph(graphfile)
		num_discrete_states=4
		beta_matrix=init_beta(beta,num_discrete_states)
		print("ite:",ite)
		
		#print(beta_matrix[0])
		#print(beta_matrix[1])
		#print(beta_matrix[2])
		#print(beta_matrix[1][1][3])
		#print(par[15][0])
		num_states=11
		numiL=3
		num_neighbors=2
		matrix_states=init_matrix(nodes,timeSeconds,num_states)
		matrix_states=init_initial_letters(nodes,matrix_states,numiL)
		buffer_request_sent=init_buffer_request_sent(nodes)
		buffer_request_received=init_buffer_request_received(nodes)
		request_state=init_request_state(nodes)
		buffer_letters_in_hand=init_buffer_letters_in_hand(nodes,players_info)
		buffer_available_letters=init_buffer_available_letters(nodes,graph,players_info)
		previous_word=init_previous_word(nodes)
		buffer_form_words=init_buffer_form_words(nodes)
		#print(buffer_form_words)
		for i in range(0,(numite-nodes+1)):
			i_time=matrix_states[i][0]
			#print("i_time:",i_time)
			agent_id=matrix_states[i][1]
			statenum=matrix_states[i][2]
			
			#print("agent_id:",agent_id)
			#print("statenum:",statenum)
						
			if i%nodes==0:
				players_ts=[['']*22 for _ in range(nodes)]
				for j in range(nodes):
					i_time_in=matrix_states[i+j][0]
					statenumin=matrix_states[i+j][2]
					agent_idin=matrix_states[i+j][1]
					xc_in=matrix_states[i+j][6]
					xb_in=matrix_states[i+j][4]
					reqsent_in=matrix_states[i+j][7]
					payload_value=payload[i+j]
					
					#print("j:",players_ts[agent_idin][1])
					#print("j:",agent_idin)
					#print(players_ts)
					players_ts[agent_idin][1]=str(agent_idin)
					#print(players_ts)
			
					#print(players_ts)
					#print("j:",agent_idin)
					#print(players_ts[agent_idin])
					players_ts[agent_idin][0]=str(ite)
					players_ts[agent_idin][1]=str(agent_idin)
					players_ts[agent_idin][2]=str(i_time_in)
					
					
				
					#print("agent_idin",agent_idin)
					#print(players_ts)
					
					if statenumin==0:
						players_ts[agent_idin][3]=str(statenumin)
					if statenumin!=0 and payload_value=="Na":
						players_ts[agent_idin][3]=str(0)
						#print(players_ts)
					beforestate=matrix_states[i+j-nodes][2]
					#print("agent_id_in:",agent_idin)
					#print("statenumin:",statenumin)
					if beforestate!=statenumin and payload_value!="Na":
						xc_in=0
					elif beforestate!=statenumin and payload_value=="Na" and statenumin!=0:
						xc_in=xc_in+1
					elif beforestate!=statenumin and statenumin==0:
						xc_in=0
					elif beforestate==statenumin and statenumin==0 and i>(nodes-1):
						xc_in=xc_in+1
					matrix_states[i+j][6]=xc_in
					######include experiments real values in output
					time_v=-1
					
					if len(player_actions[agent_idin][1])>0:
						getstate=player_actions[agent_idin][1]
						action_value=getstate[0]
						action_v=action_value[0]
						time_v=action_value[1]
						payload_v=action_value[2]
						player_id_v=action_value[3]
					#if time_v==i_time_in:
					#	print(payload_v)
					#	print(action_v)
					if time_v==i_time_in and (action_v=='replySent' or action_v=='requestSent' or action_v=='word'):
						player_actions[agent_idin][1].remove(player_actions[agent_idin][1][0])
						if action_v=='replySent':
							action_v_bool=1
						if action_v=='requestSent':
							action_v_bool=2
						if action_v=='word':
							action_v_bool=3	
						players_ts[agent_idin][9]=str(action_v_bool)
						players_ts[agent_idin][10]=str(payload_v)
						players_ts[agent_idin][11]=str(player_id_v)
						
					if statenumin==2 and payload_value!='Na'and payload_value!='':
						letterrequest,requestid,opt_rank,exp_rank,null_rank=payload_value.split(",")
						#letterrequest=playerletterl[0] 
						#requestid=int(playerletterl[1])
						requestid=int(requestid)
						matrix_states[i+j][7]=reqsent_in+1
						players_ts[agent_idin][4]=letterrequest
						
						players_ts[agent_idin][13]=opt_rank
						players_ts[agent_idin][14]=exp_rank
						players_ts[agent_idin][15]=null_rank
						
						playerletter=[str(requestid),letterrequest]
						buffer_request_sent[agent_idin].append([str(agent_idin),letterrequest])
				
						buffer_request_received[requestid].append([str(agent_idin),letterrequest])
						#print(i)
						#print("agent_idin",agent_idin)
						#print("requestid",requestid)
						#print("playerletter",playerletter)
						#print(buffer_request_received[requestid])
						#print(buffer_available_letters[agent_idin])
						#print(playerletter)
						buffer_available_letters[agent_idin].remove(playerletter)
						matrix_states[i+requestid][9]=matrix_states[i+requestid][9]+1
						matrix_states[i+requestid][4]=matrix_states[i+requestid][4]+1
						players_ts[requestid][8]=letterrequest
					if statenumin==3:
						if payload_value!='Na':
							opt_word,exp_word,null_word,ld_opt_word,ld_exp_word,ld_null_word=payload_value.split(",")
							buffer_form_words[agent_idin].append(exp_word)
							players_ts[agent_idin][6]=opt_word
							players_ts[agent_idin][12]=null_word
							players_ts[agent_idin][19]=ld_opt_word
							players_ts[agent_idin][20]=ld_exp_word
							players_ts[agent_idin][21]=ld_null_word
							#print("oi buffer_form_words",buffer_form_words)
					if statenumin==1 and payload_value!='Na'and payload_value!='':
						playerletter=payload_value.split(",")
						model_replies=int(playerletter[0])
						exp_replies=int(playerletter[1])
						random_replies=int(playerletter[2])
						playerletter=playerletter[3:]
						
						#print(player_actions[agent_idin])
						#print(payload_value)
						
						plsize=int(len(playerletter)/3)
						#print("model_replies",model_replies)
						#print("plsize",plsize)
						
						players_ts[agent_idin][16]=str(model_replies)
						players_ts[agent_idin][17]=str(plsize)
						players_ts[agent_idin][18]=str(random_replies)
						letterreplyS=''
						for a,b,c in zip(playerletter[0::3], playerletter[1::3], playerletter[2::3]):
							letterreply=a
							replyid=int(b)
							deltatime=int(c)
						#if (i_time_in)==int(playerletter[2]):
							#replyid=int(playerletter[1])
							#letterreply=playerletter[0]
							players_ts[agent_idin][5]=letterreply
							
							returnidletter=[str(replyid),letterreply]
							returnidletter2=[str(replyid),letterreply,int(i_time_in)]
							if returnidletter in buffer_request_received[agent_idin]:
								buffer_request_received[agent_idin].remove(returnidletter)
							#if returnidletter2 in request_state[agent_idin]:
							#	request_state[agent_idin].remove(returnidletter2)
							#print("buffer_request_received",buffer_request_received[agent_idin])
							matrix_states[replyid+i][8]=matrix_states[replyid+i][8]+1
							matrix_states[replyid+i][3]=matrix_states[replyid+i][3]+1
							buffer_letters_in_hand[replyid].append(letterreply)
							matrix_states[i+j][4]=matrix_states[i+j][4]-1
							letterreplyS=letterreplyS+letterreply
						players_ts[replyid][7]=letterreply
							
						#if (plsize>3):
						#	playerletter2=playerletter[3:]
						#	for a,b,c in zip(playerletter2[0::3], playerletter2[1::3], playerletter2[2::3]):
						#		letterreply=a
						#		replyid=int(b)
						#		deltatime=int(c)
						#		request_state[agent_idin].append([replyid,letterreply,deltatime])
				#print("h:",players_ts)
				for row_ts in players_ts:
					row_tsS=",".join(x for x in row_ts)
					csvfile.write(row_tsS+'\n')
					
							#print("request_state[agent_idin]",request_state[agent_idin])

			xl=matrix_states[i][3]
			xb=matrix_states[i][4]
			xw=matrix_states[i][5]
			xc=matrix_states[i][6]
			reqsent=matrix_states[i][7]
			reprec=matrix_states[i][8]
			reqrec=matrix_states[i][9]
			repsent=matrix_states[i][10]
			payload_value=payload[i]
			#print(players_info[agent_id])
			corpus=players_info[agent_id][0]
			vocabulary=players_info[agent_id][1]
			aptitudeWord=players_info[agent_id][2]
			aptitudeLetterRequest=players_info[agent_id][3]
			letterReplyType=players_info[agent_id][4]
			corpus_name=players_info[agent_id][6]
			
			
			if not request_state[agent_id] and i!=numite-nodes:
				time_v=-1
				#if agent_id==1:
					#print(i_time)
					#print(agent_id)
					#print(player_actions[agent_id][1])
				if len(player_actions[agent_id][1])>0:
					getstate=player_actions[agent_id][1]
					action_value=getstate[0]
					action_v=action_value[0]
					time_v=action_value[1]
					payload_v=action_value[2]
					player_id_v=action_value[3]
				#if agent_id==1:
					#print(time_v)
				if time_v==(i_time+1) and (action_v=='replySent' or action_v=='requestSent' or action_v=='word'):
					#player_actions[agent_id][1].remove(player_actions[agent_id][1][0])
					if action_v=='replySent':
						nextstatenum=1
					if action_v=='requestSent':
						nextstatenum=2
					if action_v=='word':
						nextstatenum=3	
				else:
					nextstatenum=0
				#nextstatenum=get_next_state(nodes,matrix_states[i],beta_matrix[statenum],num_discrete_states)
				#print(nextstatenum)
				if nextstatenum==2:
					reqsentB=len(buffer_request_sent[agent_id])
					fracRequest=reqsent/(num_neighbors*numiL)
					#fracRequest=reqsentB/(num_neighbors*numiL)
					boolletter,playerletter,opt_rank,exp_rank,null_rank=requestLetterFromNbr(agent_id,aptitudeLetterRequest,corpus_name,vocabulary,buffer_available_letters[agent_id],buffer_letters_in_hand[agent_id],buffer_form_words[agent_id],buffer_vocabulary[agent_id],numiL,fracRequest,i_letter_req_dist_len,letter_req_dist_len,letter_req_dist_data,payload_v)
					
					#print("#############testing request")
					#print(boolletter)
					#print(playerletter)
					#print(buffer_request_received)
					if boolletter==1:
						requestid=int(playerletter[0])
						letterrequest=(playerletter[1]).lower()
						#print(buffer_available_letters[agent_id])
						if playerletter not in buffer_request_sent[agent_id]:
							#buffer_request_sent[agent_id].append(playerletter)
							#buffer_request_received[requestid].append(playerletter)
							#print("request",requestid)
							#print("requestid+nodes",requestid+nodes)
							#reqsent=reqsent+1
							#matrix_states[requestid+nodes][9]=matrix_states[requestid+nodes][9]+1
							#matrix_states[requestid+nodes][4]=matrix_states[requestid+nodes][4]+1
							#buffer_available_letters[agent_id].remove(playerletter)
							#payload[i+nodes]=letterrequest+','+str(requestid)+','+str(opt_rank)+','+str(exp_rank)+','+str(null_rank)
							
							payload[i+nodes]=payload_v+','+str(player_id_v)+','+str(opt_rank)+','+str(exp_rank)+','+str(null_rank)
							
							#print(matrix_states[requestid+nodes][4]+1)
					#print("payload[i+nodes]",payload[i+nodes])
				if nextstatenum==1:
					action_reply='replySent'
					i_reply=0
					payload_value_reply=''
					count_reply=0
					lenreplyB=len(player_actions[agent_id][1])
					removeBReply=[]
					#print("player_actions[agent_id][1]",player_actions[agent_id][1])
					#print("lenreplyB",lenreplyB)
					
					while action_reply=='replySent' and lenreplyB>0:
						action_reply=player_actions[agent_id][1][i_reply][0]	
						if action_reply=='replySent':
							count_reply=count_reply+1
							if len(removeBReply)>0:
								payload_value_reply=payload_value_reply+','
							payload_value_reply=payload_value_reply+player_actions[agent_id][1][i_reply][2]+','+str(player_actions[agent_id][1][i_reply][3])+','+str(player_actions[agent_id][1][i_reply][1])		
							lenreplyB=lenreplyB-1						 		
							removeBReply.append(player_actions[agent_id][1][i_reply])
						i_reply=i_reply+1
						#print("lenreplyB",lenreplyB)
						
					#print("este",player_actions[agent_id][1])
					#print("removeBReply",removeBReply)
					for rowBR in removeBReply:
						player_actions[agent_id][1].remove(rowBR)
					#print("remove",removeBReply)	
					#if agent_id==1:
					#	print("entro",payload_value_reply)
					###remove
					
					
					sizebuffer=len(buffer_request_received[agent_id])
					#print("sizebuffer",sizebuffer)
					#print(buffer_request_received[agent_id])
					if letterReplyType=='FB':
						 if(sizebuffer==1):
						 	returnidletter=buffer_request_received[agent_id][0]
						 	#buffer_request_received[agent_id].remove(returnidletter)
						 	replyid=int(returnidletter[0])
						 	letterreply=returnidletter[1]
						 	repsent=repsent+1
						 	##xb=xb-1
						 	#matrix_states[replyid+nodes][8]=matrix_states[replyid+nodes][8]+1
						 	#matrix_states[replyid+nodes][3]=matrix_states[replyid+nodes][3]+1
						 	#buffer_letters_in_hand[replyid].append(letterreply)
						 	payload[i+nodes]='1,1,1,'+payload_value_reply
						 elif(sizebuffer>1):
						 	payloaddelta=str(sizebuffer)+','
						 	###get delta timespan from dist
						 	actualreplybuffer=sizebuffer
						 	if actualreplybuffer==2:
						 		stringreplies="SecondsFor2Replies"
						 	if actualreplybuffer==3:
						 		stringreplies="SecondsFor3Replies"
						 	if actualreplybuffer==4:
						 		stringreplies="SecondsFor4Replies"
						 	if actualreplybuffer==5:
						 		stringreplies="SecondsFor5Replies"
						 	if actualreplybuffer>5:
						 		stringreplies="SecondsFor_5Replies"
						 	elements = range(time_rpl_dist_len-i_time_rpl_dist_len)
						 	for row_time in time_rpl_dist_data:
						 		if row_time[1]=="FB" and row_time[2]==stringreplies:
						 			prob=row_time[i_time_rpl_dist_len:time_rpl_dist_len]
						 	deltareturn=np.random.choice(elements,1,p=prob)[0]+1
						 	if deltareturn<(sizebuffer-1):
						 		deltareturn=sizebuffer-1
						 	deltat=int(deltareturn/(sizebuffer-1))
						 	#print("deltareturn",deltareturn)
						 	#print("sizebuffer-1",sizebuffer-1)
						 	#print("deltat",deltat)
						 	#####timedelta=i+nodes
						 	timedelta=i_time+1
						 	#print("buffer_request_received[agent_id]",buffer_request_received[agent_id])
						 	#for index,lb in enumerate(buffer_request_received[agent_id]):
						 		#print("timedelta",timedelta)
						 		#if timedelta<=timeSeconds:
						 			#request_state[agent_id].append([lb[0],lb[1],timedelta])
						 			#print("payloaddelta",payloaddelta)
						 			#payloaddelta=payloaddelta+lb[1]+','+lb[0]+','+str(timedelta)
						 			#timedelta=timedelta+deltat
						 			#if index!=len(buffer_request_received[agent_id])-1:
						 				#payloaddelta=payloaddelta+','
						 	payload[i+nodes]=str(actualreplybuffer)+','+str(count_reply)+','+str(actualreplybuffer)+','+payload_value_reply
						 	#print("payloaddelta",payloaddelta)
					elif letterReplyType=='NFB' or letterReplyType=='Mixed':
						if(sizebuffer==1):
							returnidletter=buffer_request_received[agent_id][0]
							#buffer_request_received[agent_id].remove(returnidletter)
							replyid=int(returnidletter[0])
							letterreply=returnidletter[1]
							repsent=repsent+1
							##xb=xb-1
							#matrix_states[replyid+nodes][8]=matrix_states[replyid+nodes][8]+1
							#matrix_states[replyid+nodes][3]=matrix_states[replyid+nodes][3]+1
							#buffer_letters_in_hand[replyid].append(letterreply)
							payload[i+nodes]='1,1,1,'+payload_value_reply
						elif(sizebuffer>1):
						 	elements = np.arange(0.0, 1.1, 0.25)
						 	for row_dist in frac_rpl_dist_data:
						 		if row_dist[1]=="Mixed and NeverFullBuffer":
						 			prob=row_dist[i_frac_rpl_dist_len:frac_rpl_dist_len]
						 	actual_frac=np.random.choice(elements,1,p=prob)[0]
						 	actualreplybuffer=int(sizebuffer*actual_frac)
						 	
						 	payloaddelta=str(actualreplybuffer)+','+str(random.randint(1,actualreplybuffer))+','
						 	
						 	payload[i+nodes]=str(actualreplybuffer)+','+str(count_reply)+','+str(random.randint(1,sizebuffer))+','+payload_value_reply
						 	
						 	if actualreplybuffer>1:
						 		if actualreplybuffer==2:
						 			stringreplies="SecondsFor2Replies"
						 		if actualreplybuffer==3:
						 			stringreplies="SecondsFor3Replies"
						 		if actualreplybuffer==4:
						 			stringreplies="SecondsFor4Replies"
						 		if actualreplybuffer==5:
						 			stringreplies="SecondsFor5Replies"
						 		if actualreplybuffer>5:
						 			stringreplies="SecondsFor_5Replies"
						 		elements = range(time_rpl_dist_len-i_time_rpl_dist_len)
						 		for row_time in time_rpl_dist_data:
						 			if row_time[1]=="NotFB" and row_time[2]==stringreplies:
						 				prob=row_time[i_time_rpl_dist_len:time_rpl_dist_len]
						 		deltareturn=np.random.choice(elements,1,p=prob)[0]+1
						 		deltat=deltareturn
						 		deltab=[]
						 		deltatinit=i_time+1
						 		deltatend=deltatinit+deltat
						 		deltab.append(deltatinit)
						 		#while len(deltab)!=(sizebuffer):
						 		#	randelta=random.randint((deltatinit+1),deltatend)
						 		#	if randelta not in deltab:
						 		#		deltab.append(randelta)
						 		#for index,lb in enumerate(buffer_request_received[agent_id]):
						 		#	timedelta=deltab[index]
						 		#	if timedelta<=timeSeconds:
						 		#		request_state[agent_id].append([lb[0],lb[1],timedelta])
						 		#		payloaddelta=payloaddelta+lb[1]+','+lb[0]+','+str(timedelta)
						 		#		if index!=len(buffer_request_received[agent_id])-1:
						 		#			payloaddelta=payloaddelta+','
						 		###payload[i+nodes]=payloaddelta
						 		
				if nextstatenum==3:
					boolword,opt_word_value,wordFormed,rand_wpn,ld_opt_word,ld_exp_word,ld_null_word=formWord(makepath,agent_id,aptitudeWord,corpus_name,vocabulary,buffer_letters_in_hand[agent_id],buffer_form_words[agent_id],previous_word[agent_id],buffer_vocabulary[agent_id],numiL,word_dist_data,i_word_dist_len,word_dist_len,payload_v)
					#print(wordFormed)
					#print(ld_opt_word)
					if (boolword==1):
						xw=xw+1
						#buffer_form_words[agent_id].append(wordFormed)
						payload[i+nodes]=opt_word_value+','+wordFormed+','+rand_wpn+','+ld_opt_word+','+ld_exp_word+','+ld_null_word
						previous_word[agent_id]=wordFormed
						
						#print("previous_word",previous_word)
			#if request_state[agent_id] and i!=numite-nodes:
			#	returnidletter=0
				#print("request_state[agent_id]",request_state[agent_id])
				#print("i_time",i_time)
			#	for item in request_state[agent_id]:
			#		if int(item[2])==(i_time+1):
			#			replyid=int(item[0])
			#			letterreply=item[1]
			#			returnidletter=[str(replyid),letterreply,i_time+1]
			#			repsent=repsent+1
						##xb=xb-1
			#			nextstatenum=1
			#			payload[i+nodes]=letterreply+','+str(replyid)+','+str(i_time+1)
						#request_state[agent_id].remove(returnidletter)
			#			xc=0
						#print("returnidletter",returnidletter)
			#	if returnidletter==0:
			#		nextstatenum=0
					##xc=xc+1
			if i!=numite-nodes:
				matrix_states[i+nodes]=[matrix_states[i+nodes][0],matrix_states[i+nodes][1],nextstatenum,xl,xb,xw,xc,reqsent,reprec,reqrec,repsent]
					#print(matrix_states[i+nodes])
				#else:
					#matrix_states[i]=[matrix_states[i][0],matrix_states[i][1],nextstatenum,xl,xb,xw,xc,reqsent,reprec,reqrec,repsent]	
				#print(matrix_states[i])
			#else:			
		#matrix_states[0]=[matrix_states[0][0],matrix_states[0][1],9999,0,0,0,0,0,0,0,0]
		#matrix_states[0][2]=[9999]
		#print(matrix_states[2999,0])
		#print(nodes)
		#print(edges)
		#print(payload)
		matrix_states=matrix_states.astype(object)
		matrix_states=np.column_stack((iterationA,matrix_states,payload))
		np.savetxt(output_file, matrix_states, fmt='%s', delimiter=" ")
		##output(matrix_states,graphfile,ite,payload,iterationA)
	print (" -- h2 --")
	print (" -- good termination --")
    
## --------------------------
## Execution starts.
if __name__ == '__main__':
		
    if (len(sys.argv) != 5):
    	print ("  Error.  Incorrect usage.")
    	print ("  usage: exec infile outfile.")
    	print ("  Halt.")
    	quit()
    	
    graphfile=sys.argv[1]
    nodes=sys.argv[2]
    edges=sys.argv[3]
    timeSeconds=sys.argv[4]
    main(nodes,edges,timeSeconds)
    