
clear all
set memory 4g

/*We first need to create the new dataset: Relations by individual*/
cd ~/ProgramsAndData/Data  
use SupportAndCovariatesVil_All_06_26_2011
set more off


generate gendertemp=gender1
replace gender1=gender2
replace gender2=gendertemp
drop gendertemp

generate agetemp=age1
replace age1=age2
replace age2=agetemp
drop agetemp

generate eductemp=educ1
replace educ1=educ2
replace educ2=eductemp
drop eductemp

generate nativetemp=native1
replace native1=native2
replace native2=nativetemp
drop nativetemp

generate subcastetemp=subcaste1
replace subcaste1=subcaste2
replace subcaste2=subcastetemp
drop subcastetemp

generate occuptemp=occup1
replace occup1=occup2
replace occup2=occuptemp
drop occuptemp

generate shgtemp=shg1
replace shg1=shg2
replace shg2=shgtemp
drop shgtemp

generate rationtemp=ration1
replace ration1=ration2
replace ration2=rationtemp
drop rationtemp

generate mftemp=mf1
replace mf1=mf2
replace mf2=mftemp
drop mftemp

generate hhsizetemp=hhsize1
replace hhsize1=hhsize2
replace hhsize2=hhsizetemp
drop hhsizetemp

generate roomspptemp=roomspp1
replace roomspp1=roomspp2
replace roomspp2=roomspptemp
drop roomspptemp

generate mfeligtemp=mfelig1
replace mfelig1=mfelig2
replace mfelig2=mfeligtemp
drop mfeligtemp

generate housetemp=houseid1
replace houseid1=houseid2
replace houseid2=housetemp
drop housetemp

generate castetemp=caste1
replace caste1=caste2
replace caste2=castetemp
drop castetemp

generate worktemp=work1
replace work1=work2
replace work2=worktemp
drop worktemp

generate workprivpubtemp=workprivpub1
replace workprivpub1=workprivpub2
replace workprivpub2=workprivpubtemp
drop workprivpubtemp

generate workouttemp=workout1
replace workout1=workout2
replace workout2=workouttemp
drop workouttemp

generate loanstemp=loans1
replace loans1=loans2
replace loans2=loanstemp
drop loanstemp

generate savingstemp=savings1
replace savings1=savings2
replace savings2=savingstemp
drop savingstemp

generate eleccardtemp=eleccard1
replace eleccard1=eleccard2
replace eleccard2=eleccardtemp
drop eleccardtemp

generate rationcolortemp=rationcolor1
replace rationcolor1=rationcolor2
replace rationcolor2=rationcolortemp
drop rationcolortemp

generate bedspptemp=bedspp1
replace bedspp1=bedspp2
replace bedspp2=bedspptemp
drop bedspptemp

generate electrtemp=electr1
replace electr1=electr2
replace electr2=electrtemp
drop electrtemp

generate latrtemp=latr1
replace latr1=latr2
replace latr2=latrtemp
drop latrtemp

generate indidtemp=indid1
replace indid1=indid2
replace indid2=indidtemp
drop indidtemp

generate villageidtemp=villageid1
replace villageid1=villageid2
replace villageid2=villageidtemp
drop villageidtemp

save SupportAndCovariatesVilInv_All_06_26_2011, replace

clear all

use SupportAndCovariatesVil_All_06_26_2011
append using SupportAndCovariatesVilInv_All_06_26_2011
save individuals_All_06_26_2011, replace




clear all
set memory 4g 
set more off

use individuals_All_06_26_2011
duplicates drop houseid1 indid1, force
keep houseid1 indid1 gender1 age1 educ1 native1 subcaste1 occup1 ///
shg1 ration1 mf1 hhsize1 roomspp1 mfelig1 houseid1 caste1 work1 workprivpub1 ///
workout1 loans1 savings1 eleccard1 rationcolor1 bedspp1 electr1 latr1 villageid1

sort houseid1 indid1
save ind_data_06_26_2011, replace

/*The following two regressions correspond to the Table 5*/


clear
use individuals_All_06_26_2011
probit suppall2 gender1 age1 educ1 roomspp1 hhsize1 if fav==1
outtex
probit suppall2 gender1 age1 educ1 roomspp1 hhsize1 if fav==0
outtex



/*Running the regressions using the entire sample of pairs (linked and unlinked) is
not computationally possible in many systems due to the matsize limits of STATA 
as we have roughly 75*(600*599/2) such pairs. 
A valid alternative  for such cases is to rely in an unbiased sample instead. 
The sampling  method below is used to prevent biases due to differences degree 
(i.e If we pick the dyads at random (when looking at linked pairs) we would be much more likely 
to end with many links from individuals that have a high degree. 
So for each individual in each household we pick 3 random partners.
The sample size can be adjusted in order to check for robustness!*/

clear
use individuals_All_06_26_2011
sample 3, count by(houseid1 indid1)
save sample_06_26_2011, replace 

clear
use individuals_All_06_26_2011
drop if fav!=1
sample 3, count by(houseid1 indid1)
save sampleLinked_06_26_2011, replace


clear
use individuals_All_06_26_2011
drop if fav!=0
sample 3, count by(houseid1 indid1)
save sampleNotLinked_06_26_2011, replace



use sampleLinked_06_26_2011
probit suppall2 gender1 age1 educ1 roomspp1 hhsize1 if fav==1
outtex

use sampleNotLinked_06_26_2011
probit suppall2 gender1 age1 educ1 roomspp1 hhsize1 if fav==0
outtex































