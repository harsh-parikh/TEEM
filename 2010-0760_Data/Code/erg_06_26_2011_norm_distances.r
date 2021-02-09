library(network)
library(ergm)





#########Program Code#################################
measurecount<-14
resultsmat<-array(0,dim=c(4,16))	
while(measurecount<=14){
	if(measurecount==1){
		name_base="PFavor"
		name_context="basePF"
        }	
	if(measurecount==2){
		name_base="PFavor"
		name_context="baseAll1"
        }	
	if(measurecount==3){
		name_base="PFavor"
		name_context="baseAll2"
        }	
	if(measurecount==4){
		name_base="IFavor"
		name_context="baseIF"
        }	
	if(measurecount==5){
		name_base="IFavor"
		name_context="baseAll1"
        }	
	if(measurecount==6){
		name_base="IFavor"
		name_context="baseAll2"
        }	
	if(measurecount==7){
		name_base="Social"
		name_context="baseS"
        }	
	if(measurecount==8){
		name_base="Social"
		name_context="baseAll1"
        }
	if(measurecount==9){
		name_base="Social"
		name_context="baseAll2"
        }	
	if(measurecount==10){
		name_base="All1"
		name_context="baseAll1"
        }	
	if(measurecount==11){
		name_base="All2"
		name_context="baseAll2"
        }	
	if(measurecount==12){
		name_base="Favors"
		name_context="baseFavors"
        }	
	if(measurecount==13){
		name_base="Favors"
		name_context="baseAll1"
        }
	if(measurecount==14){
		name_base="Favors"
		name_context="baseAll2"
        }	
	if(measurecount==15){
		name_base="PFavor"
		name_context="baseFavors"
        }	
	if(measurecount==16){
		name_base="IFavor"
		name_context="baseFavors"
        }
	

	resultsmat<-array(0,dim=c(6,77))	
	villagenum<-1
	while (villagenum<=77){	
		
		if((villagenum!=13)&(villagenum!=22)){	

			print(paste("Currently Analyzing:", name_base, "-", name_context, "_", villagenum, sep=""))
			
			setwd("~/ProgramsAndData/Data/Relations/Long")
			temp1<-read.table(paste(villagenum, "-", name_base, ".csv", sep=""),  sep = ",", header=FALSE)	
			mat1<-data.matrix(temp1)
	
			temp2<-read.table(paste(villagenum, "-", name_context, ".csv", sep=""),  sep = ",", header=FALSE)
			mat2<-data.matrix(temp2)

			temp3<-read.table(paste(villagenum, "-Dist_Surv.csv", sep=""),  sep = ",", header=FALSE)
			mat3<-data.matrix(temp3)
			mat3<-mat3/mean(mat3)
	##############Turn it into a 0, 1 matrix once more########################
			temp<-(mat2>=1)	
			mat2<-temp*1
	
			net1 <- network(mat1)
			net2 <- network(mat2)
			
			model1<-ergm(net1~edges+edgecov(net2)+edgecov(mat3))
			resultsmat[1,villagenum]<-summary(model1)$coefs[1,1]
			resultsmat[2,villagenum]<-summary(model1)$coefs[2,1]
			resultsmat[3,villagenum]<-summary(model1)$coefs[3,1]
			resultsmat[4,villagenum]<-summary(model1)$coefs[1,2]
			resultsmat[5,villagenum]<-summary(model1)$coefs[2,2]
			resultsmat[6,villagenum]<-summary(model1)$coefs[3,2]
		}
		villagenum<-villagenum+1
	}
	setwd("~/ProgramsAndData/Results")
	resultsname=paste("ergm-distnorm-", name_base, "_", name_context, ".csv", sep="")
	write.csv(resultsmat, file=resultsname, row.names=FALSE, quote=FALSE)
	measurecount<-measurecount+1
}

setwd("~/ProgramsAndData/Code")
		















