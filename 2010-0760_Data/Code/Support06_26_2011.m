%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Social Networks          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read in the data about the village so we can build the full mapping
% between internal ID and HHIDPID

cd('~/ProgramsAndData/Data/Raw_csv')


support=zeros(77,6,16);
supportInv=zeros(77,6,16);
totsize=0;
suppglob=zeros(6,16);
suppglob2=zeros(6,16);
denom=zeros(1,16);
suppglobInv=zeros(6,16);
suppglob2Inv=zeros(6,16);
denomInv=zeros(1,16);


for x=1:77
	% two missing villages
	if x==13|x==22
		continue;
	end
    	w=x

	filename=strcat('village',num2str(w),'.csv');
	village = csvread(filename);

	persons = 100*village(:,1) + village(:,2);
	numPersons = length(persons);

	% vids = floor( village(:,1) / 1000 );
	% vid = vids(1); % This is the village id. We only use one at a time

	% Household ID's and Person ID's
	hhids = mod( village(:,1), 1000 );
	pids = village(:,2);
	numHouseholds = max(hhids);
	
	% These structures will map between the "person ID" (as given in the cells of the file)
	% and their "ID"  (as used in the adjacency matrix)
	% Since "person ID" could look like 500124, and we don't want an adjacency matrix that is 500K x 500K,
	% each person wil be associated wtih an ID that counts from 1, so that row ID(person) and col ID(person)
	% of the adjacency matrix will define the links for that person
	personToID = {};
	IDtoPerson = [];
	
	% This structure will be used to show that members within a household have a relationship
	% households(i,j) = ID( v i j ) if household i has pid j (in village v), 0 otherwise
	households = zeros(numHouseholds,9);
	
	% Populate the mappings and household structures
	for curID=1:numPersons
		person = persons(curID);
	  	hhid = hhids(curID);
	  	pid = pids(curID);
	  
	  	% It turns out that octave/matlab will not let the field of a structure begin with a number
	  	% So we add letters before the numbers
	  	personField = ['P',num2str(person)];
	  
	  	% Update the maps
	  	personToID.(personField) = curID;
	  	IDtoPerson(curID) = person;
	  
  		% Update the household structure
  		households(hhid,pid) = curID;
  
	end

	% Now we will build the "base" adjacency matrix, in which nodes which belong
	% to the same household are connected
	baseAdjMat = zeros(numPersons,numPersons);

	housesize=zeros(numHouseholds,1);
	% Look through the households 
	for h = 1:numHouseholds
  
  		% Within the household, we want all the pairs of PID's
  		% (i,j) for i != j
  		for i = 1:max(village(:,2))
    
    			% No more ID's left in this household
    			if ~households(h,i)
      				housesize(h,1)=i-1;
      				break;
    			end
    
    			basePerson = households(h,i);
    
    			% Match i with its pairing, j
    			for j = i+1:max(village(:,2))
      
      				% No more ID's to pair with
      				if ~households(h,j)
        				break;
      				end
      
      				curPerson = households(h,j);
      
      				% Add vertices
      				baseAdjMat(basePerson,curPerson) = 0;
      				baseAdjMat(curPerson,basePerson) = 0;
      
    			end
    
  		end
  		if i==max(village(:,2))&housesize(h,1)==0
      			housesize(h,1)=max(village(:,2));
  		end

	end



	% Now we must build adjacency matrices for every relationship

	relationships = ...
  	{
    	strcat('visitgo',num2str(w));
    	strcat('visitcome',num2str(w));
    	strcat('nonrel',num2str(w));
    	strcat('medic',num2str(w));
    	strcat('keroricego',num2str(w));
    	strcat('keroricecome',num2str(w));
    	strcat('borrowmoney',num2str(w));
    	strcat('lendmoney',num2str(w));
    	strcat('helpdecision',num2str(w));
    	%strcat('locleader',num2str(w));
    	strcat('giveadvice',num2str(w));
    	strcat('templecompany',num2str(w));
    	strcat('rel',num2str(w));
  	};

	numRelationships = size(relationships,1);

	villageRelationships = zeros(numPersons,numPersons,numRelationships);

	% Walk through each relationship
	for relationshipID=1:numRelationships
  		curRelationship = relationships{relationshipID};
  
  		% Read in the relationship info as a List of Lists (LoL)
  		LoL = csvread([ curRelationship, '.csv' ]);
  
  		% Start building the Adjacency Matrix
  		adjMat = zeros(numPersons,numPersons);
  
  		for i = 1:size(LoL,1)
    
    		% The first cell in a row is the ID of the base person
    			basePerson = LoL(i,1);
    			if (basePerson == 0)
      				continue;
    			end
    
    		% Find the ID mapping
    			basePersonField = ['P',num2str(basePerson)];
    			if ~isfield(personToID,basePersonField)
      				%disp(['Warning: When processing relationship ', curRelationship, ...
          			%  ' could not find HHIDPID ', basePersonField, ' in village table']);
        			%disp(['...occured on row ', num2str(i), ', column ', num2str(1), ...
            			%  ' of file ', curRelationship, '.csv']);
      				continue;
    			end
    
    			baseID = personToID.(basePersonField);

    			% The remaining cells in the row are the links to other people that were
    			% given by the base person
    			for j = 2:size(LoL,2)

      				curPerson = LoL(i,j);
      			if curPerson==0 % If it's a zero there is nothing to process
        			continue
		      end	
      
      			% Find the ID mapping for the person
     				 curPersonField = ['P',num2str(curPerson)];
      			if ~isfield(personToID,curPersonField)
        			%disp(['Warning: When processing relationship ', curRelationship, ...
            			%  ' could not find HHIDPID ', curPersonField, ' in village table']);
        			%disp(['...occured on row ', num2str(i), ', column ', num2str(j), ...
            			%  ' of file ', curRelationship, '.csv']);
        			continue;
      			end
      			curID = personToID.(curPersonField);
      
      			% Make the relevant updates to the adjacency matrix
      			adjMat(baseID,curID) = 1;
      			%adjMat(curID,baseID) = 1;
      
    		end
  	end
  
  	% Put the matrix into the village relationships structure
  	villageRelationships(:,:,relationshipID) = adjMat;
  
  	% Now let's make a CSV file of this, for viewing purposes
  
  	% Add row labels
  	adjMat = [ IDtoPerson', adjMat ];
  	% Add column labels
  	adjMat = [ [0,IDtoPerson]; adjMat ];

  	%csvwrite([num2str(w),'-',curRelationship,'.csv'],adjMat);
  	%dlmwrite(['adjmat-d-',curRelationship,'.csv'],adjMat(2:length(adjMat-1),2:length(adjMat-1)),'delimiter','\t');
  
	end

	villageRelationships(:,:,12)=max(baseAdjMat,villageRelationships(:,:,12));
	for indi=1:12
    		for i=1:numPersons
        		villageRelationships(i,i,indi)=0;
    		end
	end


	% input survey persons
	%%%01-04-2010 Needed to move this up here in order to compute amount of reciprocity and distribution of support for surveyed people

	surveyperson=csvread(strcat('survey',num2str(w),'.csv'));
	surveyid=surveyperson(:,1)*100+surveyperson(:,2);

	survey=zeros(length(surveyid),1);
	for i=1:length(surveyid)
    		for j=1:numPersons
    			if IDtoPerson(j)==surveyid(i);
        			survey(i)=j;
        			break;
    			end
    		end
	end

	isurvey=zeros(numPersons,1);
	for i=1:length(survey)
    		isurvey(survey(i))=i;
	end


	k=0;
	for i=1:numPersons
    		if isurvey(i)==0
			villageRelationships(i-k,:,:)=[];
			villageRelationships(:,i-k,:)=[];
        		k=k+1;
    		end
	end





	%create Physical Favor(PS), IF and Social
	%and Measure reciprocity

	tempsize=size(villageRelationships(:,:,1));
	tempsize=tempsize(1);
	tempOnes=ones(tempsize,1);

	villageRelationships=0+villageRelationships;
	V=max(villageRelationships(:,:,1),villageRelationships(:,:,2)');
	V=min(V,V');

	F=max(villageRelationships(:,:,3),villageRelationships(:,:,3)');

	Med=max(villageRelationships(:,:,4),villageRelationships(:,:,4)');

	K=max(villageRelationships(:,:,5),villageRelationships(:,:,6)');
	K=min(K,K');

	M=max(villageRelationships(:,:,7),villageRelationships(:,:,8)');
	M=min(M,M');

	A=max(villageRelationships(:,:,9),villageRelationships(:,:,10)');
	A=min(A,A');

	R=max(villageRelationships(:,:,12),villageRelationships(:,:,12)');


	PF=max(M,K);
	IF=max(A,Med);
	Fav=max(PF,IF);
	S=max(V,F);
	All1=PF|IF|S;
	All2=PF|IF|S|villageRelationships(:,:,11)|villageRelationships(:,:,12);
	All2=max(All2,All2');
	
	for i=1:length(PF)
		for i=1:length(PF)
			PF(i,i)=0;
			IF(i,i)=0;
			Fav(i,i)=0;
			S(i,i)=0;
			All1(i,i)=0;
			All2(i,i)=0;	
		end
	end


	%calculate support measures
	count=zeros(6,16);
	countInv=zeros(6,16);
	basePF=PF^2;
	baseIF=IF^2;
	baseFav=Fav^2;
	baseS=S^2;
	baseAll1=All1^2;
	baseAll2=All2^2;

	
	
	for i=1:length(PF)-1
		for j=i+1:length(PF)
		        if PF(i,j)==1&basePF(i,j)~=0
				count(6,1)=count(6,1)+1;
		        end
		        if PF(i,j)==1&baseAll1(i,j)~=0
				count(6,2)=count(6,2)+1;
		        end
	        	if IF(i,j)==1&baseIF(i,j)~=0
	            		count(6,3)=count(6,3)+1;
	        	end
	        	if IF(i,j)==1&baseAll1(i,j)~=0
	            		count(6,4)=count(6,4)+1;
	       		end
	        	if S(i,j)==1&baseS(i,j)~=0
	            		count(6,5)=count(6,5)+1;
	        	end
	        	if S(i,j)==1&baseAll1(i,j)~=0
	            		count(6, 6)=count(6,6)+1;
	        	end
	        	if All1(i,j)==1&baseAll1(i,j)~=0
	            		count(6, 7)=count(6, 7)+1;
	        	end
		 	if PF(i,j)==1&baseAll2(i,j)~=0
	            		count(6,8)=count(6,8)+1;
	    		end
			if IF(i,j)==1&baseAll2(i,j)~=0
	            		count(6,9)=count(6,9)+1;
	        	end
		 	if S(i,j)==1&baseAll2(i,j)~=0
	            		count(6, 10)=count(6,10)+1;
	        	end
		 	if All2(i,j)==1&baseAll2(i,j)~=0
	            		count(6, 11)=count(6, 11)+1;
	        	end
			if Fav(i,j)==1&baseFav(i,j)~=0
	            		count(6,12)=count(6,12)+1;
	        	end
	 		if Fav(i,j)==1&baseAll1(i,j)~=0
            			count(6, 13)=count(6,13)+1;
	       		end
			if Fav(i,j)==1&baseAll2(i,j)~=0
	         		count(6, 14)=count(6, 14)+1;
	        	end
			if PF(i,j)==1&baseFav(i,j)~=0
            			count(6, 15)=count(6,15)+1;
	       		end
			if IF(i,j)==1&baseFav(i,j)~=0
	         		count(6, 16)=count(6, 16)+1;
	        	end
		end
	end

	for z=1:5
		for i=1:length(PF)-1		
			for j=i+1:length(PF)
		        	if PF(i,j)==1&basePF(i,j)==z
		       			count(z,1)=count(z,1)+1;
		       		end
		        	if PF(i,j)==1&baseAll1(i,j)==z
		       	    		count(z,2)=count(z,2)+1;
		       		end
		       		if IF(i,j)==1&baseIF(i,j)==z
		           	   		count(z,3)=count(z,3)+1;
		            	end
		       		if IF(i,j)==1&baseAll1(i,j)==z
		          		count(z,4)=count(z,4)+1;
		        	end
		       	 	if S(i,j)==1&baseS(i,j)==z
		     			count(z,5)=count(z,5)+1;
		       		end
	        		if S(i,j)==1&baseAll1(i,j)==z
	           			count(z,6)=count(z,6)+1;
	       			end
	       			if All1(i,j)==1&baseAll1(i,j)==z
	            			count(z,7)=count(z,7)+1;
	        		end
				if PF(i,j)==1&baseAll2(i,j)==z
		           		count(z,8)=count(z,8)+1;
	    			end
				if IF(i,j)==1&baseAll2(i,j)==z
	            			count(z,9)=count(z,9)+1;
	        		end
	 			if S(i,j)==1&baseAll2(i,j)==z
            				count(z, 10)=count(z,10)+1;
	        		end
	 			if All2(i,j)==1&baseAll2(i,j)==z
	           		    	count(z, 11)=count(z, 11)+1;
	        		end
				if Fav(i,j)==1&baseFav(i,j)==z
	            			count(z,12)=count(z,12)+1;
	        		end
	 			if Fav(i,j)==1&baseAll1(i,j)==z
            				count(z, 13)=count(z,13)+1;
	        		end
	 			if Fav(i,j)==1&baseAll2(i,j)==z
	           		    	count(z, 14)=count(z, 14)+1;
	        		end
				if PF(i,j)==1&baseFav(i,j)==z
	            			count(z, 15)=count(z,15)+1;
		       		end
				if IF(i,j)==1&baseFav(i,j)==z
		         		count(z, 16)=count(z, 16)+1;
		        	end
			
	  		end
		end
	end
	count(:,1)=count(:,1)*2/sum(sum(PF));
	count(:,2)=count(:,2)*2/sum(sum(PF));
	count(:,3)=count(:,3)*2/sum(sum(IF));
	count(:,4)=count(:,4)*2/sum(sum(IF));
	count(:,5)=count(:,5)*2/sum(sum(S));
	count(:,6)=count(:,6)*2/sum(sum(S));
	count(:,7)=count(:,7)*2/sum(sum(All1));
	count(:,8)=count(:,8)*2/sum(sum(PF));
	count(:,9)=count(:,9)*2/sum(sum(IF));
	count(:,10)=count(:,10)*2/sum(sum(S));
	count(:,11)=count(:,11)*2/sum(sum(All2));
	count(:,12)=count(:,12)*2/sum(sum(Fav));
	count(:,13)=count(:,13)*2/sum(sum(Fav));
	count(:,14)=count(:,14)*2/sum(sum(Fav));
	count(:,15)=count(:,15)*2/sum(sum(PF));
	count(:,16)=count(:,16)*2/sum(sum(IF));

	support(x,:,:)=count;
	
	
	suppglob=suppglob+tempsize*count;
	
	suppglob2(:,1)=suppglob2(:,1)+ sum(sum(PF))*count(:,1);
	denom(1,1)=denom(1,1)+sum(sum(PF));
	suppglob2(:,2)=suppglob2(:,2)+ sum(sum(PF))*count(:,2);
	denom(1,2)=denom(1,2)+sum(sum(PF));
	suppglob2(:,3)=suppglob2(:,3)+ sum(sum(IF))*count(:,3);
	denom(1,3)=denom(1,3)+sum(sum(IF));
	suppglob2(:,4)=suppglob2(:,4)+ sum(sum(IF))*count(:,4);
	denom(1,4)=denom(1,4)+sum(sum(IF));
	suppglob2(:,5)=suppglob2(:,5)+ sum(sum(S))*count(:,5);
	denom(1,5)=denom(1,5)+sum(sum(S));
	suppglob2(:,6)=suppglob2(:,6)+ sum(sum(S))*count(:,6);
	denom(1,6)=denom(1,6)+sum(sum(S));
	suppglob2(:,7)=suppglob2(:,7)+ sum(sum(All1))*count(:,7);
	denom(1,7)=denom(1,7)+sum(sum(All1));
	suppglob2(:,8)=suppglob2(:,8)+ sum(sum(PF))*count(:,8);
	denom(1,8)=denom(1,8)+sum(sum(PF));
	suppglob2(:,9)=suppglob2(:,9)+ sum(sum(IF))*count(:,9);
	denom(1,9)=denom(1,9)+sum(sum(IF));
	suppglob2(:,10)=suppglob2(:,10)+ sum(sum(S))*count(:,10);
	denom(1,10)=denom(1,10)+sum(sum(S))	;
	suppglob2(:,11)=suppglob2(:,11)+ sum(sum(All2))*count(:,11);
	denom(1,11)=denom(1,11)+sum(sum(All2));
	suppglob2(:,12)=suppglob2(:,12)+ sum(sum(Fav))*count(:,12);
	denom(1,12)=denom(1,12)+sum(sum(Fav));
	suppglob2(:,13)=suppglob2(:,13)+ sum(sum(Fav))*count(:,13);
	denom(1,13)=denom(1,13)+sum(sum(Fav));
	suppglob2(:,14)=suppglob2(:,14)+ sum(sum(Fav))*count(:,14);
	denom(1,14)=denom(1,14)+sum(sum(Fav));
	suppglob2(:,15)=suppglob2(:,15)+ sum(sum(PF))*count(:,15);
	denom(1,15)=denom(1,15)+sum(sum(PF));
	suppglob2(:,16)=suppglob2(:,16)+ sum(sum(IF))*count(:,16);
	denom(1,16)=denom(1,16)+sum(sum(IF));
	
	


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%We Compute the Benchmark Support levels: Specifically, for each relationship of focus we measure the support  enjoyed by "non-relationships" among pairs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

	numRelsPF=PF*tempOnes;
	numRelsIF=IF*tempOnes;
	numRelsS=S*tempOnes;
	numRelsFav=Fav*tempOnes;
	numRelsAll1=All1*tempOnes;
	numRelsAll2=All2*tempOnes;

	boolePF=zeros(tempsize,0);
	booleIF=zeros(tempsize,0);
	booleS=zeros(tempsize,0);
	booleFav=zeros(tempsize,0);
	booleAll=zeros(tempsize,0);
	booleAll2=zeros(tempsize,0);

	invertMat=ones(tempsize,tempsize);
	invPF= invertMat-PF;
	invIF=invertMat-IF;
	invS=invertMat-S;
	invFav=invertMat-Fav;
	invAll1=invertMat-All1;
	invAll2=invertMat-All2;	

	for i=1:length(PF)
		for i=1:length(PF)
			invPF(i,i)=0;
			invIF(i,i)=0;
			invFav(i,i)=0;
			invS(i,i)=0;
			invAll1(i,i)=0;
			invAll2(i,i)=0;	
		end
	end

	
	for i=1:length(PF)-1
		for j=i+1:length(PF)
			%%%%%%%  PF %%%%%%%%%%%%%%%%%
			if invPF(i,j)==1&basePF(i,j)~=0
				countInv(6,1)=countInv(6,1)+1;
		       	end
		        if invPF(i,j)==1&baseAll1(i,j)~=0
				countInv(6,2)=countInv(6,2)+1;
		        end
			if invPF(i,j)==1&baseAll2(i,j)~=0
	            		countInv(6,8)=countInv(6,8)+1;
	    		end
			if invPF(i,j)==1&baseFav(i,j)~=0
            			countInv(6, 15)=countInv(6,15)+1;
	       		end
			
			%%%%%%%  IF %%%%%%%%%%%%%%%%%		
	        	if invIF(i,j)==1&baseIF(i,j)~=0
	 	         	countInv(6,3)=countInv(6,3)+1;
	 		end
	        	if invIF(i,j)==1&baseAll1(i,j)~=0
	         	 	countInv(6,4)=countInv(6,4)+1;
	       		end
			if invIF(i,j)==1&baseAll2(i,j)~=0
	            		countInv(6,9)=countInv(6,9)+1;
	        	end
			if invIF(i,j)==1&baseFav(i,j)~=0
	         		countInv(6, 16)=countInv(6, 16)+1;
	        	end

			%%%%%%%  S %%%%%%%%%%%%%%%%%
			if invS(i,j)==1&baseS(i,j)~=0
	           		countInv(6,5)=countInv(6,5)+1;
	        	end
	        	if invS(i,j)==1&baseAll1(i,j)~=0
	            		countInv(6, 6)=countInv(6,6)+1;
	        	end
			if invS(i,j)==1&baseAll2(i,j)~=0
	            		countInv(6, 10)=countInv(6,10)+1;
	        	end
			
			%%%%%%%  All1 %%%%%%%%%%%%%%%%%	
	        	if invAll1(i,j)==1&baseAll1(i,j)~=0
	            		countInv(6, 7)=countInv(6, 7)+1;
	        	end
		
		 	
			%%%%%%%  All2 %%%%%%%%%%%%%%%%%	
			if invAll2(i,j)==1&baseAll2(i,j)~=0
	            		countInv(6, 11)=countInv(6, 11)+1;
	        	end
	
			%%%%%%%  Fav %%%%%%%%%%%%%%%%%	
			if invFav(i,j)==1&baseFav(i,j)~=0
	            		countInv(6,12)=countInv(6,12)+1;
	        	end
	 		if invFav(i,j)==1&baseAll1(i,j)~=0
            			countInv(6, 13)=countInv(6,13)+1;
	       		end
			if invFav(i,j)==1&baseAll2(i,j)~=0
	         		countInv(6, 14)=countInv(6, 14)+1;
	        	end
		end
	end



	for z=1:5
		for i=1:length(PF)-1
			for j=i+1:length(PF)
				%%%%%%%  PF %%%%%%%%%%%%%%%%%
				if invPF(i,j)==1&basePF(i,j)==z
					countInv(z,1)=countInv(z,1)+1;
		       		end
		        	if invPF(i,j)==1&baseAll1(i,j)==z
					countInv(z,2)=countInv(z,2)+1;
		        	end
				if invPF(i,j)==1&baseAll2(i,j)==z
	            			countInv(z,8)=countInv(z,8)+1;
	    			end
				if invPF(i,j)==1&baseFav(i,j)==z
            				countInv(z, 15)=countInv(z,15)+1;
	       			end
			
				%%%%%%%  IF %%%%%%%%%%%%%%%%%		
	        		if invIF(i,j)==1&baseIF(i,j)==z
	 	        	 	countInv(z,3)=countInv(z,3)+1;
	 			end
	        		if invIF(i,j)==1&baseAll1(i,j)==z
	         		 	countInv(z,4)=countInv(z,4)+1;
	       			end
				if invIF(i,j)==1&baseAll2(i,j)==z
	            			countInv(z,9)=countInv(z,9)+1;
	        		end
				if invIF(i,j)==1&baseFav(i,j)==z
	         			countInv(z, 16)=countInv(z, 16)+1;
	        		end

				%%%%%%%  S %%%%%%%%%%%%%%%%%
				if invS(i,j)==1&baseS(i,j)==z
	           			countInv(z,5)=countInv(z,5)+1;
	        		end
	        		if invS(i,j)==1&baseAll1(i,j)==z
	            			countInv(z, 6)=countInv(z,6)+1;
	        		end
				if invS(i,j)==1&baseAll2(i,j)==z
	            			countInv(z, 10)=countInv(z,10)+1;
	        		end
			
				%%%%%%%  All1 %%%%%%%%%%%%%%%%%	
	        		if invAll1(i,j)==1&baseAll1(i,j)==z
	            			countInv(z, 7)=countInv(z, 7)+1;
	        		end
			
		 	
				%%%%%%%  All2 %%%%%%%%%%%%%%%%%	
				if invAll2(i,j)==1&baseAll2(i,j)==z
	            			countInv(z, 11)=countInv(z, 11)+1;
	        		end
		
				%%%%%%%  Fav %%%%%%%%%%%%%%%%%	
				if invFav(i,j)==1&baseFav(i,j)==z
	            			countInv(z,12)=countInv(z,12)+1;
	        		end
	 			if invFav(i,j)==1&baseAll1(i,j)==z
            				countInv(z, 13)=countInv(z,13)+1;
	       			end
				if invFav(i,j)==1&baseAll2(i,j)==z
	         			countInv(z, 14)=countInv(z, 14)+1;
	        		end
			end	
		end	
	end



	countInv(:,1)=countInv(:,1)*2/sum(sum(invPF));
	countInv(:,2)=countInv(:,2)*2/sum(sum(invPF));
	countInv(:,3)=countInv(:,3)*2/sum(sum(invIF));
	countInv(:,4)=countInv(:,4)*2/sum(sum(invIF));
	countInv(:,5)=countInv(:,5)*2/sum(sum(invS));
	countInv(:,6)=countInv(:,6)*2/sum(sum(invS));
	countInv(:,7)=countInv(:,7)*2/sum(sum(invAll1));
	countInv(:,8)=countInv(:,8)*2/sum(sum(invPF));
	countInv(:,9)=countInv(:,9)*2/sum(sum(invIF));
	countInv(:,10)=countInv(:,10)*2/sum(sum(invS));
	countInv(:,11)=countInv(:,11)*2/sum(sum(invAll2));
	countInv(:,12)=countInv(:,12)*2/sum(sum(invFav));
	countInv(:,13)=countInv(:,13)*2/sum(sum(invFav));
	countInv(:,14)=countInv(:,14)*2/sum(sum(invFav));
	countInv(:,15)=countInv(:,15)*2/sum(sum(invPF));
	countInv(:,16)=countInv(:,16)*2/sum(sum(invIF));

	supportInv(x,:,:)=countInv;
	
	
	suppglobInv=suppglobInv+tempsize*countInv;
	
	suppglob2Inv(:,1)=suppglob2Inv(:,1)+ sum(sum(invPF))*countInv(:,1);
	denomInv(1,1)=denomInv(1,1)+sum(sum(invPF));
	suppglob2Inv(:,2)=suppglob2Inv(:,2)+ sum(sum(invPF))*countInv(:,2);
	denomInv(1,2)=denomInv(1,2)+sum(sum(invPF));
	suppglob2Inv(:,3)=suppglob2Inv(:,3)+ sum(sum(invIF))*countInv(:,3);
	denomInv(1,3)=denomInv(1,3)+sum(sum(invIF));
	suppglob2Inv(:,4)=suppglob2Inv(:,4)+ sum(sum(invIF))*countInv(:,4);
	denomInv(1,4)=denomInv(1,4)+sum(sum(invIF));
	suppglob2Inv(:,5)=suppglob2Inv(:,5)+ sum(sum(invS))*countInv(:,5);
	denomInv(1,5)=denomInv(1,5)+sum(sum(invS));
	suppglob2Inv(:,6)=suppglob2Inv(:,6)+ sum(sum(invS))*countInv(:,6);
	denomInv(1,6)=denomInv(1,6)+sum(sum(invS));
	suppglob2Inv(:,7)=suppglob2Inv(:,7)+ sum(sum(invAll1))*countInv(:,7);
	denomInv(1,7)=denomInv(1,7)+sum(sum(invAll1));
	suppglob2Inv(:,8)=suppglob2Inv(:,8)+ sum(sum(invPF))*countInv(:,8);
	denomInv(1,8)=denomInv(1,8)+sum(sum(invPF));
	suppglob2Inv(:,9)=suppglob2Inv(:,9)+ sum(sum(invIF))*countInv(:,9);
	denomInv(1,9)=denomInv(1,9)+sum(sum(invIF));
	suppglob2Inv(:,10)=suppglob2Inv(:,10)+ sum(sum(invS))*countInv(:,10);
	denomInv(1,10)=denomInv(1,10)+sum(sum(invS));
	suppglob2Inv(:,11)=suppglob2Inv(:,11)+ sum(sum(invAll2))*countInv(:,11);
	denomInv(1,11)=denomInv(1,11)+sum(sum(invAll2));
	suppglob2Inv(:,12)=suppglob2Inv(:,12)+ sum(sum(invFav))*countInv(:,12);
	denomInv(1,12)=denomInv(1,12)+sum(sum(invFav));
	suppglob2Inv(:,13)=suppglob2Inv(:,13)+ sum(sum(invFav))*countInv(:,13);
	denomInv(1,13)=denomInv(1,13)+sum(sum(invFav));
	suppglob2Inv(:,14)=suppglob2Inv(:,14)+ sum(sum(invFav))*countInv(:,14);
	denomInv(1,14)=denomInv(1,14)+sum(sum(invFav));
	suppglob2Inv(:,15)=suppglob2Inv(:,15)+ sum(sum(invPF))*countInv(:,15);
	denomInv(1,15)=denomInv(1,15)+sum(sum(invPF));
	suppglob2Inv(:,16)=suppglob2Inv(:,16)+ sum(sum(invIF))*countInv(:,16);
	denomInv(1,16)=denomInv(1,16)+sum(sum(invIF));
	
	totsize=totsize+tempsize;
	clear villageRelationships;
	

end
	
suppglob=suppglob/totsize;
suppglobInv=suppglobInv/totsize;


for i=1:16
	suppglob2(:,i)=suppglob2(:,i)/denom(1,i);
end

for i=1:16
	suppglob2Inv(:,i)=suppglob2Inv(:,i)/denomInv(1,i);
end
	
	

cd('~/ProgramsAndData/Data')
csvwrite('Support_Measures06_26_2011.csv',suppglob);
	
csvwrite('Support_Measures_Glob_06_26_2011.csv',suppglob2);

	
for z=1:6
	namefile=strcat('Support_',num2str(z), '_Vlevel06_26_2011.csv'); 
	csvwrite(namefile, support(:,z,:));
end	    
	
	

csvwrite('Support_MeasuresInv06_26_2011.csv',suppglobInv);
	
csvwrite('Support_Measures_GlobInv_06_26_2011.csv',suppglob2Inv);

	
for z=1:6
	namefile=strcat('SupportInv_',num2str(z), '_Vlevel06_26_2011.csv'); 
	csvwrite(namefile, supportInv(:,z,:));
end
	
	

    

    
    
    
    
