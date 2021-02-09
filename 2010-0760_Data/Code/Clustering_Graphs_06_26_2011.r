library(statnet)


#########Program Code#################################
setwd("~/ProgramsAndData/Data")
name1="Support_6_Vlevel06_26_2011"
name2="Clustering06_26_2011"

temp1<-read.table(paste(name1,".csv", sep=""),  sep = ",", header=FALSE)
mat1<-data.matrix(temp1)
mat1<-mat1[-13,]
mat1<-mat1[-21,]

temp2<-read.table(paste(name2,".csv", sep=""),  sep = ",", header=FALSE)
mat2<-data.matrix(temp2)
mat2<-mat2[-13,]
mat2<-mat2[-21,]


measurecount<-1
while(measurecount<=14){
	if(measurecount==1){
		name_base="PFavor"
		name_context="PFavor"
        }	
	if(measurecount==2){
		name_base="PFavor"
		name_context="All1"
        }	
	if(measurecount==8){
		name_base="PFavor"
		name_context="All2"
        }	
	if(measurecount==3){
		name_base="IFavor"
		name_context="IFavor"
        }	
	if(measurecount==4){
		name_base="IFavor"
		name_context="All1"
        }	
	if(measurecount==9){
		name_base="IFavor"
		name_context="All2"
        }	
	if(measurecount==5){
		name_base="Social"
		name_context="Social"
        }	
	if(measurecount==6){
		name_base="Social"
		name_context="All1"
        }
	if(measurecount==10){
		name_base="Social"
		name_context="All2"
        }	
	if(measurecount==7){
		name_base="All1"
		name_context="All1"
        }	
	if(measurecount==11){
		name_base="All2"
		name_context="All2"
        }
	if(measurecount==12){
		name_base="Fav"
		name_context="Fav"
        }	
	if(measurecount==13){
		name_base="Fav"
		name_context="All1"
        }	
	if(measurecount==14){
		name_base="Fav"
		name_context="All2"
        }	
	if(measurecount==15){
		name_base="PF"
		name_context="Fav"
        }	
	if(measurecount==16){
		name_base="IF"
		name_context="Fav"	
        }

	
	setwd("~/ProgramsAndData/Results")
	
	clust<-mat2[,measurecount]	
	supp<-mat1[,measurecount]
	
	both<-array(0, dim=c(75,2))
	both.sorted.smoothed<-array(0, dim=c(75,2))

	both[,1]=clust<-mat1[,measurecount]
	both[,2]=clust<-mat2[,measurecount]	
	
	print(paste("Plotting...", name_base, "-", name_context))
	clust.sorted <- sort(clust, decreasing=FALSE)
	supp.sorted <- sort(supp, decreasing=FALSE)
	both.sorted<-both[order(both[,1], decreasing=FALSE),] 

	xax<-c(1:75)
	xax=xax/75	
	
        clust.smoothed<-loess(clust.sorted~xax, span=0.5)$fitted
	supp.smoothed<-loess(supp.sorted~xax, span=0.5)$fitted

	both.sorted.smoothed[,1]<-loess(both.sorted[,1]~xax, span=0.5)$fitted
	both.sorted.smoothed[,2]<-loess(both.sorted[,2]~xax, span=0.5)$fitted

	name<-paste(name_base,"-", name_context, sep="") 
	plotname=paste("/ProgramsAndData/Results/Clust_Supp_",name,".jpeg", sep="")

	plot(xax,supp.smoothed, type="l", lty=2, lwd=2, sub="Clustering vs. Support (Inverse CDF)", 
		main=name, ylab="", xlab="", col='blue', ylim=c(0,1))
	lines(xax,clust.smoothed, col='black',lty=1, lwd=2)
	legend("topleft",  c("Support", "Clustering"), col=c(4,1), lty=c(2,1), ncol = 1, cex = 0.8)
	savePlot(filename=plotname , type ="jpeg");
	

	name<-paste(name_base,"-", name_context, sep="") 
	plotname=paste("~/ProgramsAndData/Results/Supp_Ord_",name,".jpeg", sep="")

	plot(xax,both.sorted.smoothed[,1], type="l", lty=2, lwd=2, sub="Clustering vs. Support (Ordered by Support)", 
		main=name, ylab="", xlab="", col='blue', ylim=c(0,1))
	lines(xax,both.sorted.smoothed[,2], col='black',lty=1, lwd=2)
	legend("topleft",  c("Support", "Clustering"), col=c(4,1), lty=c(2,1), ncol = 1, cex = 0.8)
	savePlot(filename=plotname , type ="jpeg");
	




	measurecount<-measurecount+1
}
setwd("~/ProgramsAndData/Code")
















