%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          Social Networks          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read in the data about the village so we can build the full mapping
% between internal ID and HHIDPID

cd('~/ProgramsAndData/Data/Raw_csv')

cluster=zeros(77,16);

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
	All1=1*All1;
	All2=1*All2;

	count=zeros(1,16);
	average=zeros(1,16);
	totdeg=zeros(1,16);



	%clustering

	degree=sum(PF);
	clus=PF*PF*PF;
	up=0;down=0;
	for i=1:length(PF)
    		up=up+clus(i,i);
    		down=down+degree(i)*(degree(i)-1);
	end
	count(1,1)=up/down;
	average(1,1)=average(1,1)+count(1,1)*down;
	totdeg(1,1)=totdeg(1,1)+down;	

	degree=sum(PF);
	clus=PF*All1*PF;
	up=0;down=0;
	for i=1:length(PF)
    		up=up+clus(i,i);
    		down=down+degree(i)*(degree(i)-1);
	end
	count(1,2)=up/down;
	average(1,2)=average(1,2)+count(1,2)*down;
	totdeg(1,2)=totdeg(1,2)+down;	
    
	degree=sum(IF);
	clus=IF*IF*IF;
	up=0;down=0;
	for i=1:length(IF)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,3)=up/down;
	average(1,3)=average(1,3)+count(1,3)*down;
	totdeg(1,3)=totdeg(1,3)+down;	

	degree=sum(IF);
	clus=IF*All1*IF;
	up=0;down=0;
	for i=1:length(All1)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,4)=up/down;
	average(1,4)=average(1,4)+count(1,4)*down;
	totdeg(1,4)=totdeg(1,4)+down;	


	degree=sum(S);
	clus=S*S*S;
	up=0;down=0;
	for i=1:length(S)
    		up=up+clus(i,i);
    		down=down+degree(i)*(degree(i)-1);
	end
	count(1,5)=up/down;
	average(1,5)=average(1,5)+count(1,5)*down;
	totdeg(1,5)=totdeg(1,5)+down;	

	degree=sum(S);
	clus=S*All1*S;
	up=0;down=0;
	for i=1:length(S)
    		up=up+clus(i,i);
    		down=down+degree(i)*(degree(i)-1);
	end
	count(1,6)=up/down;
	average(1,6)=average(1,6)+count(1,6)*down;
	totdeg(1,6)=totdeg(1,6)+down;	
    
	degree=sum(All1);
	clus=All1*All1*All1;
	up=0;down=0;
	for i=1:length(All1)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,7)=up/down;
	average(1,7)=average(1,7)+count(1,7)*down;
	totdeg(1,7)=totdeg(1,7)+down;	

	degree=sum(PF);
	clus=PF*All2*PF;
	up=0;down=0;
	for i=1:length(PF)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,8)=up/down;
	average(1,8)=average(1,8)+count(1,8)*down;
	totdeg(1,8)=totdeg(1,8)+down;	

	degree=sum(IF);
	clus=IF*All2*IF;
	up=0;down=0;
	for i=1:length(IF)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,9)=up/down;
	average(1,9)=average(1,9)+count(1,9)*down;
	totdeg(1,9)=totdeg(1,9)+down;	

	degree=sum(S);
	clus=S*All2*S;
	up=0;down=0;
	for i=1:length(S)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,10)=up/down;
	average(1,10)=average(1,10)+count(1,10)*down;
	totdeg(1,10)=totdeg(1,10)+down;	

	degree=sum(All2);
	clus=All2*All2*All2;
	up=0;down=0;
	for i=1:length(All2)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,11)=up/down;
	average(1,11)=average(1,11)+count(1,11)*down;
	totdeg(1,11)=totdeg(1,11)+down;	

	degree=sum(Fav);
	clus=Fav*Fav*Fav;
	up=0;down=0;
	for i=1:length(Fav)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,12)=up/down;
	average(1,12)=average(1,12)+count(1,12)*down;
	totdeg(1,12)=totdeg(1,12)+down;	


	degree=sum(Fav);
	clus=Fav*All1*Fav;
	up=0;down=0;
	for i=1:length(Fav)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,13)=up/down;
	average(1,13)=average(1,13)+count(1,13)*down;
	totdeg(1,13)=totdeg(1,13)+down;	


	degree=sum(Fav);
	clus=Fav*All2*Fav;
	up=0;down=0;
	for i=1:length(Fav)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,14)=up/down;
	average(1,14)=average(1,14)+count(1,14)*down;
	totdeg(1,14)=totdeg(1,14)+down;	

	degree=sum(PF);
	clus=PF*Fav*PF;
	up=0;down=0;
	for i=1:length(PF)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,15)=up/down;
	average(1,15)=average(1,15)+count(1,15)*down;
	totdeg(1,15)=totdeg(1,15)+down;	

	degree=sum(IF);
	clus=IF*Fav*IF;
	up=0;down=0;
	for i=1:length(IF)
    		up=up+clus(i,i);
	    	down=down+degree(i)*(degree(i)-1);
	end
	count(1,16)=up/down;
	average(1,16)=average(1,16)+count(1,16)*down;
	totdeg(1,16)=totdeg(1,16)+down;	



cluster(x,:)=count;
        
clear villageRelationships;

end


for i=1:16
	average(1,i)=average(1,i)/totdeg(1,i)
end

cd('~/ProgramsAndData/Data')	
namefile=strcat('Clustering06_26_2011.csv'); 
csvwrite(namefile, cluster(:,:));

namefile=strcat('Average_Clustering06_26_2011.csv'); 
csvwrite(namefile, average(:,:));	    
	
    






    

    
    
    
    
