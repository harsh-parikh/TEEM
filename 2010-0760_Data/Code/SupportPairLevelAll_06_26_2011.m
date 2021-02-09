%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Social Networks          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read in the data about the village so we can build the full mapping
% between internal ID and HHIDPID

clear all;

cd('~/../../../j/a/jacksonm/indianvillages/Data/Raw_csv')
SupportAndCovariates=zeros(1,68);


startVil=1;
endVil=77; 
numPair=0;
for x=startVil:endVil
	% two missing villages
	if x==13|x==22
		continue;
	end
    	w=x;

	filename=strcat('village',num2str(w),'.csv');
	village = csvread(filename);
    
     gps=csvread(strcat('gps',num2str(w),'.csv'));

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
      
      				% Add vertices %%%In this version we are not adding
      				% them!
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

	surveyperson=csvread(strcat('survey_extended',num2str(w),'.csv'));
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

	disp('Finished creating matrices');
	
	for i=1:length(PF)-1
		for j=i+1:length(PF)
                numPair=numPair+1;
                    %%%Link statistics
                    %Support
                if basePF(i,j)~=0 
                    SupportAndCovariates(numPair,1)=1; 
                end 
                    %Number of Supporting links
                    SupportAndCovariates(numPair,6+1)=basePF(i,j);
                
                if baseIF(i,j)~=0 
                    SupportAndCovariates(numPair,2)=1; 
                end 
                    %Number of Supporting links
                    SupportAndCovariates(numPair,6+2)=baseIF(i,j);
                    
               if baseFav(i,j)~=0 
                    SupportAndCovariates(numPair,3)=1; 
                end 
                    %Number of Supporting links
                    SupportAndCovariates(numPair,6+3)=baseFav(i,j);     
                    
                if baseS(i,j)~=0 
                    SupportAndCovariates(numPair,4)=1; 
                end 
                    %Number of Supporting links
                    SupportAndCovariates(numPair,6+4)=baseS(i,j);
                    
                if baseAll1(i,j)~=0 
                    SupportAndCovariates(numPair,5)=1; 
                end 
                    %Number of Supporting links
                    SupportAndCovariates(numPair,6+5)=baseAll1(i,j); 
                    
                     if baseAll2(i,j)~=0 
                    SupportAndCovariates(numPair,6)=1; 
                end 
                    %Number of Supporting links
                    SupportAndCovariates(numPair,6+6)=baseAll2(i,j); 
                        
                    %link presence  PhysicalF
                    SupportAndCovariates(numPair, 13)=PF(i,j); 
                     %link presence  IntangibleF
                    SupportAndCovariates(numPair, 14)=IF(i,j);
                     %link presence  Favor
                    SupportAndCovariates(numPair, 15)=Fav(i,j);
                     %link presence  Hedonic
                    SupportAndCovariates(numPair, 16)=S(i,j);
                     %link presence  All1
                    SupportAndCovariates(numPair, 17)=All1(i,j);
                     %link presence  ALl2
                    SupportAndCovariates(numPair, 18)=All2(i,j);
                    
                    %% Personal Statistics
                    %gender 
                    SupportAndCovariates(numPair, 19)=surveyperson(i,25);
                    SupportAndCovariates(numPair, 20)=surveyperson(j,25);
           
                    %age
                   SupportAndCovariates(numPair, 21)=surveyperson(i,3);
                   SupportAndCovariates(numPair, 22)=surveyperson(j,3);
                   
                   %education
                   SupportAndCovariates(numPair, 23)=surveyperson(i,4);
                   SupportAndCovariates(numPair, 24)=surveyperson(j,4);
                   
                  %Village Native
                   SupportAndCovariates(numPair, 25)=surveyperson(i,5);
                   SupportAndCovariates(numPair, 26)=surveyperson(j,5);
                           
                   %subcaste
                   SupportAndCovariates(numPair, 27)=surveyperson(i,7);
                   SupportAndCovariates(numPair, 28)=surveyperson(j,7);
                   
                  %Occupation
                   SupportAndCovariates(numPair, 29)=surveyperson(i,9);
                   SupportAndCovariates(numPair, 30)=surveyperson(j,9);
                   
                   %SHG
                   SupportAndCovariates(numPair, 31)=surveyperson(i,12);
                   SupportAndCovariates(numPair, 32)=surveyperson(j,12);
                   
                   %Ration Card
                   SupportAndCovariates(numPair, 33)=surveyperson(i,16);
                   SupportAndCovariates(numPair, 34)=surveyperson(j,16);

                   %MF
                   SupportAndCovariates(numPair, 35)=surveyperson(i,18);
                   SupportAndCovariates(numPair, 36)=surveyperson(j,18);
                   
                   %Size of the Household
                   SupportAndCovariates(numPair, 37)=surveyperson(i,19);
                   SupportAndCovariates(numPair, 38)=surveyperson(j,19);
                   
                   %Number of rooms per person
                   SupportAndCovariates(numPair, 39)=surveyperson(i,20);
                   SupportAndCovariates(numPair, 40)=surveyperson(j,20);
                   
                   %MF elegibility
                   SupportAndCovariates(numPair, 41)=surveyperson(i,24);
                   SupportAndCovariates(numPair, 42)=surveyperson(j,24);
                   
                   %Household ID
                   SupportAndCovariates(numPair, 43)=surveyperson(i,1);
                   SupportAndCovariates(numPair, 44)=surveyperson(j,1);
                   
                   %Caste
                   SupportAndCovariates(numPair, 45)=surveyperson(i,6);
                   SupportAndCovariates(numPair, 46)=surveyperson(j,6);
                   
                   %Work
                   SupportAndCovariates(numPair, 47)=surveyperson(i,8);
                   SupportAndCovariates(numPair, 48)=surveyperson(j,8);
                   
                   %WorkPrivPub
                   SupportAndCovariates(numPair, 49)=surveyperson(i,10);
                   SupportAndCovariates(numPair, 50)=surveyperson(j,10);
                   
                   %WorkOutside
                   SupportAndCovariates(numPair, 51)=surveyperson(i,11);
                   SupportAndCovariates(numPair, 52)=surveyperson(j,11);

                   %OutsLoans
                   SupportAndCovariates(numPair, 53)=surveyperson(i,13);
                   SupportAndCovariates(numPair, 54)=surveyperson(j,13);
                   
                   %SavingsAccount
                   SupportAndCovariates(numPair, 55)=surveyperson(i,14);
                   SupportAndCovariates(numPair, 56)=surveyperson(j,14);
                   
                   %ElectionCard
                   SupportAndCovariates(numPair, 57)=surveyperson(i,15);
                   SupportAndCovariates(numPair, 58)=surveyperson(j,15);
                   
                   %RationCardCol
                   SupportAndCovariates(numPair, 59)=surveyperson(i,17);
                   SupportAndCovariates(numPair, 60)=surveyperson(j,17);
                   
                   %BedsPerPerson
                   SupportAndCovariates(numPair, 61)=surveyperson(i,21);
                   SupportAndCovariates(numPair, 62)=surveyperson(j,21);
                   
                   %Electricity
                   SupportAndCovariates(numPair, 63)=surveyperson(i,22);
                   SupportAndCovariates(numPair, 64)=surveyperson(j,22);
                  
                   %Latrine
                   SupportAndCovariates(numPair, 65)=surveyperson(i,23);
                   SupportAndCovariates(numPair, 66)=surveyperson(j,23);
                   
                  %PersonId
                   SupportAndCovariates(numPair, 67)=surveyperson(i,2);
                   SupportAndCovariates(numPair, 68)=surveyperson(j,2);
                   
                
                   
                   
                   
		end
	end

	
end


%Add it using CAt in unix shell



headers={'SuppPF', 'SuppIF', 'SuppFav', 'SuppHed', 'SuppAll1', 'SuppAll2',...
    'NumSuppPF', 'NumSuppIF', 'NumSuppFav', 'NumSuppHed', 'NumSuppAll1', ... 
    'NumSuppAll2', 'PFav', 'IFav', 'Fav', 'Hed', 'All1', 'All2', 'Gender1',...
    'Gender2', 'Age1', 'Age2', 'educ1', 'educ2', 'native1', 'native2',...
    'subcaste1', 'subcaste2', 'Occup1', 'Occup2', 'Shg1', 'Shg2', 'Ration1',...
    'Ration2', 'Mf1', 'Mf2', 'HhSize1', 'Hhsize2', 'Roomspp1', 'Roomspp2',...
    'MfElig1', 'MfElig2', 'HouseId1', 'HouseId2', 'Caste1', 'Caste2', 'Work1',...
    'Work2', 'WorkPrivPub1', 'WorkPrivPub2', 'WorkOut1', 'WorkOut2',...
    'Loans1', 'Loans2', 'Savings1', 'Savings2', 'ElecCard1', 'ElecCard2', 'RationColor1', 'RationColor2',...
    'Bedspp1', 'Bedspp2', 'Electr1', 'Electr2', 'Latr1', 'Latr2', 'indid1', 'indid2'} ;

cd('~/../../../j/a/jacksonm/indianvillages/Data')


filename=['SupportAndCovariatesVil_', num2str(startVil), 'to', num2str(endVil), '_02_21_2011.csv']
cell2csv(filename, headers);
dlmwrite(filename, SupportAndCovariates, '-append', 'delimiter', ',');
	
quit();
    

    
    
