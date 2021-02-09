clear all
set more off
set memory 4g
cd ~/ProgramsAndData/Data 
set more off


foreach base in fav{
	foreach context in suppall2{
		foreach property in SCasteDyad educDyad wealthDyad1 wealthDyad2 genderDyad SHGDyad ageDyad {
			clear all
			use SupportAndCovariatesVil_All_06_26_2011
			display "Calculating Stats for `base' in `context' decomposing by `property'"   
			collapse (mean) mean=`context' (count) numObs=houseid1, by(`property' `base' villageid)			
			reshape wide mean numObs, i(villageid `property')  j(`base')
			reshape wide mean0  mean1 numObs0 numObs1, i(villageid)  j(`property')
			outsheet villageid mean0* mean1* numObs0* numObs1*  using `property'_`base'_`context'.csv, c replace
		}
	
	}
}

**************************************MicroFinance************************************************************************************************************************	
foreach base in fav{
	foreach context in suppall2{
		foreach property in MFDyad {
			clear all
			use SupportAndCovariatesVil_All_06_26_2011
			display "Calculating Stats for `base' in `context' decomposing by `property'"   
			collapse (mean) mean=`context' (count) numObs=houseid1 if mfelig1==1 & mfelig2==1 , by(`property' `base' villageid)			
			reshape wide mean numObs, i(villageid `property')  j(`base')
			reshape wide mean0  mean1 numObs0 numObs1, i(villageid)  j(`property')
			outsheet villageid mean0* mean1* numObs0* numObs1*  using `property'_`base'_`context'.csv, c replace
		}
	
	}
}



	
