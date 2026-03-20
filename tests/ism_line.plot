PAPERFORMAT A4H BBROTATE

PLOT   : Absolute Fluxes
*
\LETTERSIZE 0.25
* 
\OFS  2.0 21.7
\INBOX
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
*2:orange
\DEFINECOLOR 2  0.93, 0.62, 0.18
*3:red
\DEFINECOLOR 3  0.73, 0.25, 0.25
*4:blue
\DEFINECOLOR 4  0.36, 0.49, 0.79
*5:green
\DEFINECOLOR 5  0.1 0.75 0.45
*6: pink
\DEFINECOLOR 6  0.9, 0.5, 0.8
*7:purple
\DEFINECOLOR 7  0.5, 0.3, 0.6

\INSTRUCTION EXPORT
*********************
****  variables  ****
*********************
*
\VAR STAR  = "SMC 006894, O9V"
\VAR EBV = 0.08
\VAR DM =  18.7
\VAR SHIFT = 0.00

\VAR vmac0 = 0 km/s 
\VAR vmac10 = 10 km/s 
\VAR vmac100 = 50 km/s 

\VAR vrot0 =  0
\VAR vrot100 =  100
\VAR vrot500 =  500

\VAR bin =0.1
\VAR Z_sol = 1.307e-3

\VAR CONVOL_HST  = "GAUSS=0.1"
* For HR spectra: FWHM = 0.2 Angstrom 
* For LR specta: FWHM = 0.6 Angstrom 
\VAR CONVOL_VLT = "GAUSS=0.6"


****  models  ****
 

\VAR PATH = test_data/

\VAR MODEL1 = T33_logg4.0_Mdot-8.5_logL4.72_v1994_vmic8_Z0.1
\VAR OUT_ISM = test_data/ism_lorentz.dat
\VAR OUT_ISM2 = test_data/ism_voigt.dat
\VAR OUT_ISM3 = test_data/ism_hlymana.dat
\VAR OUT_ISM4 = test_data/ism_hlymana_dens.dat

\EXPR MODEL1 = $PATH // $MODEL1
\EXPR FNAME1     = $MODEL1 // /formal.plot
\EXPR FNAMEOUT1  = $MODEL1 // /formal.out
\EXPR FNAMEFLUX1 = $MODEL1 // /wruniq.plot

  
\VAR INCKEY_OPT = "* OPT" 

****  IDENTS  **** 
\VAR ident_file = ~eschoesser/projects/bridge/idents/ident_O.dat
*
\INSTRUCTION NO-EXPORT

***********************************************************************
***********************************************************************
***********************************************************************
 
*PLOT   : HST1.1

\LETTERSIZE 0.25
\OFS  2.0 15
\INBOX
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\PEN=2
\LINUN XMIN 1. XMAX 1.  0. 0.

***********************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT IUE SHORT"
 \LINUNLAB 1560 1.55 1780 1.55 0 0 0 0.27 &EFe IV
\LINUNLAB 1830 1.65 2100 1.65 0 0 0 0.27 &EFe III
 HEADER :
 X-ACHSE:\CENTER\#l# / \A
 Y-ACHSE:\CENTER\Normalized flux  
     MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
 X:  18.0CM     960.0      1220.0       5.00     10.00     0.0000     
 Y:  5.5CM      0       1.8        0.500     0.5     0.0000     

*****  MODELS  *****


*N=? SYMBOL=9 SIZE=0.1 COLOR=5 PEN=3  
*COMMAND SETNAME MODEL9
*COMMAND ISMLINE COLDENS=-1.0E15 L0=1608.451 GAMM=320000000 F=0.58 VRAD=-45
*COMMAND WRITE FILE=$OUT_ISM DATASET=MODEL9
*COMMAND INCLUDE $FNAME1

*N=? SYMBOL=9 SIZE=0.1 COLOR=6 PEN=3  
*COMMAND SETNAME MODEL1
*COMMAND ISMLINE COLDENS=-1.0E13 L0=1608.451 GAMM=500000000 F=0.7 VTURB=10 A=55.845 T=8000 VRAD=60
*COMMAND WRITE FILE=$OUT_ISM2 DATASET=MODEL1
*COMMAND INCLUDE $FNAME1

*N=? SYMBOL=9 SIZE=0.1 COLOR=6 PEN=3  
*COMMAND SETNAME MODEL2
*COMMAND HLYMANA EBV=-0.1 
*COMMAND WRITE FILE=$OUT_ISM3 DATASET=MODEL2
*COMMAND INCLUDE $FNAME1

N=? SYMBOL=9 SIZE=0.1 COLOR=6 PEN=3  
COMMAND SETNAME MODEL3
COMMAND HLYMANA COLDENS=-1.0E18 VTURB=10 T=80000 VRAD=60
COMMAND WRITE FILE=$OUT_ISM4 DATASET=MODEL3
COMMAND INCLUDE $FNAME1

END
