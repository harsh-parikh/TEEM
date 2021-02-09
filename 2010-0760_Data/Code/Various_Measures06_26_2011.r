library(statnet)


#########Program Code#################################
setwd("~/ProgramsAndData/Data")
name1="Support_6_Vlevel06_26_2010"
name2="Support_1_Vlevel06_26_2010"
name3="Support_2_Vlevel06_26_2010"
name4="Support_3_Vlevel06_26_2010"
name5="Support_4_Vlevel06_26_2010"
name6="Support_5_Vlevel06_26_2010"

temp1<-read.table(paste(name1,".csv", sep=""),  sep = ",", header=FALSE)
mat1<-data.matrix(temp1)
mat1<-mat1[-13,]
mat1<-mat1[-21,]

temp2<-read.table(paste(name2,".csv", sep=""),  sep = ",", header=FALSE)
mat2<-data.matrix(temp2)
mat2<-mat2[-13,]
mat2<-mat2[-21,]

temp3<-read.table(paste(name3,".csv", sep=""),  sep = ",", header=FALSE)
mat3<-data.matrix(temp3)
mat3<-mat3[-13,]
mat3<-mat3[-21,]

temp4<-read.table(paste(name4,".csv", sep=""),  sep = ",", header=FALSE)
mat4<-data.matrix(temp4)
mat4<-mat4[-13,]
mat4<-mat4[-21,]

temp5<-read.table(paste(name5,".csv", sep=""),  sep = ",", header=FALSE)
mat5<-data.matrix(temp5)
mat5<-mat5[-13,]
mat5<-mat5[-21,]

temp6<-read.table(paste(name6,".csv", sep=""),  sep = ",", header=FALSE)
mat6<-data.matrix(temp6)
mat6<-mat6[-13,]
mat6<-mat6[-21,]


measurecount<-1
while(measurecount<=16){
	if(measurecount==1){
		name_base="PFavors"
		name_context="PFavors"
        }	
	if(measurecount==2){
		name_base="PFavors"	
		name_context="Hedonic (or) Favors"	
        }	
	if(measurecount==8){
		name_base="PFavors"
		name_context="All"
        }	
	if(measurecount==3){
		name_base="IFavors"
		name_context="IFavors"
        }	
	if(measurecount==4){
		name_base="IFavors"
		name_context="Hedonic (or) Favors"
        }	
	if(measurecount==9){
		name_base="IFavors"
		name_context="All"
        }	
	if(measurecount==5){
		name_base="Hedonic"
		name_context="Hedonic"
        }	
	if(measurecount==6){
		name_base="Hedonic"
		name_context="Hedonic (or) Favors"
        }
	if(measurecount==10){
		name_base="Hedonic"
		name_context="All"
        }	
	if(measurecount==7){
		name_base="Hedonic (or) Favors"
		name_context="Hedonic (or) Favors"
        }	
	if(measurecount==11){
		name_base="All"
		name_context="All"
        }	
	if(measurecount==12){
		name_base="Favors"
		name_context="Favors"
        }
	if(measurecount==13){
		name_base="Favors"
		name_context="Hedonic (or) Favors"
        }	
	if(measurecount==14){
		name_base="Favors"
		name_context="All"
        }	
	if(measurecount==15){
		name_base="PFavors"
		name_context="Favors"
        }	
	if(measurecount==16){
		name_base="IFavors"
		name_context="Favors"	
        }	


	if(name_base=="Favors"){
		name_base1="Fav"	
	}
	if(name_base=="PFavors"){
		name_base1="PF"	
	}
	if(name_base=="IFavors"){
		name_base1="IF"	
	}
	if(name_base=="All"){
		name_base1="All"	
	}
	if(name_base=="Hedonic (or) Favors"){
		name_base1="H(or)F"	
	}
	if(name_base=="Hedonic"){
		name_base1="Hed"	
	}

	if(name_context=="Favors"){
		name_context1="Fav"	
	}
	if(name_context=="PFavors"){
		name_context1="PF"	
	}
	if(name_context=="IFavors"){
		name_context1="IF"	
	}
	if(name_context=="All"){
		name_context1="All"	
	}
	if(name_context=="Hedonic (or) Favors"){
		name_context1="H(or)F"	
	}
	if(name_context=="Hedonic"){
		name_context1="Hed"	
	}


	setwd("~/ProgramsAndData/Results")
	
	
	supp<-mat1[,measurecount]
	
	all<-array(0, dim=c(75,6))
	all.sorted.smoothed<-array(0, dim=c(75,6))

	all[,1]<-mat1[,measurecount]
	all[,2]<-mat2[,measurecount]
	all[,3]<-mat3[,measurecount]
	all[,4]<-mat4[,measurecount]	
	all[,5]<-mat5[,measurecount]
	all[,6]<-mat6[,measurecount]		
	
	print(paste("Plotting...", name_base, "-", name_context))
	all.sorted<-all[order(all[,1], decreasing=FALSE),] 

	xax<-c(1:75)
	xax=xax/75	

	all.sorted.smoothed[,1]<-loess(all.sorted[,1]~xax, span=0.5)$fitted
	all.sorted.smoothed[,2]<-loess(all.sorted[,2]~xax, span=0.5)$fitted
	all.sorted.smoothed[,3]<-loess(all.sorted[,3]~xax, span=0.5)$fitted
	all.sorted.smoothed[,4]<-loess(all.sorted[,4]~xax, span=0.5)$fitted
	all.sorted.smoothed[,5]<-loess(all.sorted[,5]~xax, span=0.5)$fitted
	all.sorted.smoothed[,6]<-loess(all.sorted[,6]~xax, span=0.5)$fitted

	name<-paste("g'=",name_base,", g=", name_context, sep="") 
	name2=paste(name_base1,"-", name_context1, sep="")
	plotname=paste("~/ProgramsAndData/Results/Supp_Lev",name2,".jpeg", sep="")

	plot(xax,all.sorted.smoothed[,1], type="l", lty=1, lwd=2, 
		main=name, ylab="", xlab="", col='black', ylim=c(0,1))
	lines(xax,all.sorted.smoothed[,2], col='blue',lty=2, lwd=2)
	lines(xax,all.sorted.smoothed[,3], col='red',lty=3, lwd=2)
	lines(xax,all.sorted.smoothed[,4], col='green',lty=4, lwd=2)
	lines(xax,all.sorted.smoothed[,5], col='black',lty=5, lwd=2)
	lines(xax,all.sorted.smoothed[,6], col='black',lty=6, lwd=2)

	legend("topleft",  c("Supported", "by 1", "by 2", "by 3", "by 4", "by 5"), col=c(1,4,2,3,1,1), lty=c(1,2,3,4,5,6), ncol = 1, cex = 0.8)
	print(paste("Writing...", plotname))	
	savePlot(filename=plotname , type ="jpeg");
	
	measurecount<-measurecount+1
}
setwd("~/ProgramsAndData/Code")
















