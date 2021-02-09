% Generate village level regression variables %

cd('~\ProgramsAndData\Data\Raw_csv')

vilreg=zeros(1,9);
count=1;
mfset=[1,2,3,4,12,19,20,21,23,24,25,28,29,31,32,33,36,39,42,43,45,46,47,50,51,52,55,57,59,62,65,67,68,70,71,72,73,75];

for x=1:length(mfset)
    
    w=mfset(x);



% import data
village=csvread(strcat('village',num2str(w),'.csv'));
gps=csvread(strcat('gps',num2str(w),'.csv'));
survey=csvread(strcat('survey',num2str(w),'.csv'));
bss=csvread(strcat('bss',num2str(w),'.csv'));

% population in village and household
vilreg(x,1)=length(village);
for i=1:length(village)-1
    if village(i+1,2)<=village(i,2)
        house=mod(village(i,1),1000);
        gps(house,8)=village(i,2);
    end
end
house=mod(village(i+1,1),1000);
gps(house,8)=village(i,2);

% education (0-15)
edu=survey(:,4);
k=0;
for i=1:length(edu)
    if edu(i-k)==0
        edu(i-k)=[];
        k=k+1;
        continue;
    end
    if edu(i-k)==16
        edu(i-k)=0;
    end
end
vilreg(x,2)=mean(edu);

% wealth (room per person, bed for person, electricity (0-2),latrine(0-2))
k=0;
for i=1:length(gps)
    if gps(i-k,2)==0|gps(i-k,8)==0
        gps(i-k,:)=[];
        k=k+1;
    end
end
gps(:,4)=gps(:,4)./gps(:,8);
gps(:,5)=gps(:,5)./gps(:,8);
vilreg(x,3:6)=mean(gps(:,4:7));
vilreg(x,5:6)=3-vilreg(x,5:6);

%micro finance
for i=1:length(village)
    if village(i,3)==2&village(i,4)>14
        gps(mod(village(i,1),1000),9)=1;
    end
end
vilreg(x,7)=length(bss)/sum(gps(:,9));

end

cd('~\ProgramsAndData\Data')

support=csvread('Support_6_Vlevel06_26_2011.csv');

for x=1:length(mfset)
    w=mfset(x);
    vilreg(x,8)=support(w,14);
    vilreg(x,9)=support(w,10);
end



    
    
    
    



