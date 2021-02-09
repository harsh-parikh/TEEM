library(plotrix)


#########Program Code#################################
measurecount<-14
while(measurecount<=14){
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
		name_basefile="Favors"	
	}
	if(name_base=="PFavors"){
		name_base1="PF"
		name_basefile="PFavor"	
	}
	if(name_base=="IFavors"){
		name_base1="IF"	
		name_basefile="IFavor"	
	}
	if(name_base=="All"){
		name_base1="All"
		name_basefile="All2"	
	}
	if(name_base=="Hedonic (or) Favors"){
		name_base1="H(or)F"
		name_basefile="All1"	
	}
	if(name_base=="Hedonic"){
		name_base1="Hed"
		name_basefile="Social"	
	}

	if(name_context=="Favors"){
		name_context1="Fav"
		name_contextfile="baseFavors"	
	}
	if(name_context=="PFavors"){
		name_context1="PF"
		name_contextfile="basePF"		
	}
	if(name_context=="IFavors"){
		name_context1="IF"	
		name_contextfile="baseIF"	
	}
	if(name_context=="All"){
		name_context1="All"	
		name_contextfile="baseAll2"	
	}
	if(name_context=="Hedonic (or) Favors"){
		name_context1="H(or)F"	
		name_contextfile="baseAll1"	
	}
	if(name_context=="Hedonic"){
		name_context1="Hed"	
		name_contextfile="baseS"	
	}

	setwd("~/ProgramsAndData/Results/")
	name1=paste("ergm-distnorm-4sd-", name_basefile, "_", name_contextfile, sep="")
	temp1<-read.table(paste(name1,".csv", sep=""),  sep = ",", header=TRUE)
	mat1<-data.matrix(temp1)
	mat1<-mat1[,-13]
	mat1<-mat1[,-21]
	
	
	supp<-mat1[,measurecount]
	
	all1<-array(0, dim=c(2,75))
	all2<-array(0, dim=c(2,75))
	all3<-array(0, dim=c(2,75))
	

	all1[1,]<-mat1[1,]
	all1[2,]<-2.64*mat1[4,]

	all2[1,]<-mat1[2,]
	all2[2,]<-2.64*mat1[5,]
	
	all3[1,]<-mat1[3,]
	all3[2,]<-2.64*mat1[6,]

	
	
	
	print(paste("Plotting...", name_base, "-", name_context))
	all.sorted<-all[order(all[,1], decreasing=FALSE),] 

	xax<-c(1:75)	


	name<-paste("g'=",name_base,", g=", name_context, sep="") 
	name2=paste(name_base1,"-", name_context1, sep="")
	plotname=paste("~/ProgramsAndData/Results/Supp_Coefnorm4sd",name2,".jpeg", sep="")

	plotCI(xax, all2[1,], all2[2,], lty=1, lwd=2, sub="(Ergm) Support Coefficient", 
		main=name, ylab="", xlab="Village", col='red', scol="blue", ylim=c(0,4))
	#lines(xax,all.sorted.smoothed[,2], col='blue',lty=2, lwd=2)
	#lines(xax,all.sorted.smoothed[,3], col='red',lty=3, lwd=2)
	#lines(xax,all.sorted.smoothed[,4], col='green',lty=4, lwd=2)
	#lines(xax,all.sorted.smoothed[,5], col='black',lty=5, lwd=2)
	#lines(xax,all.sorted.smoothed[,6], col='black',lty=6, lwd=2)

	#legend("topleft",  c("Supported", "by 1", "by 2", "by 3", "by 4", "by 5"), col=c(1,4,2,3,1,1), lty=c(1,2,3,4,5,6), ncol = 1, cex = 0.8)
	print(paste("Writing...", plotname))	
	savePlot(filename=plotname , type ="jpeg");
	
	measurecount<-measurecount+1
}
setwd("~/ProgramsAndData/Code")
















