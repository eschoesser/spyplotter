PAPERFORMAT A4H BBROTATE
MULTIPLOT START
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
*
*
\INSTRUCTION EXPORT
*********************
****  variables  ****
*********************
*
\VAR STAR  = "O9-9.5 V"
\VAR EBV = 0.08
\VAR DM =  18.7
\VAR SHIFT = 0.03
\VAR SHIFTUV = 0.0
\VAR vrot =  8 km/s 
\VAR vmac= 8 km/s 
\VAR VRAD =  163
\VAR convol = 0.2
\VAR convol2 = 0.6
\VAR bin =0.2
*

****  PHOTOMETRY  ****
\VAR FUV   = 14.11
\VAR NUV   = 14.29
\VAR B     = 14.95
\VAR V     = 15.16
\VAR R     = 15.5
\VAR G     = 15.2
\VAR J     = 15.75
\VAR H     = 15.86
\VAR K     = 15.89
\VAR IRAC1 = 15.91
\VAR IRAC2 = 15.89
*
\VAR VTURBHL= 80.0 
\VAR VRADHL = 50.0
\VAR clight = 300000
*

\VAR VRADUV = 170.
\VAR BINHST= 0.2
\VAR CONVOL_HST  = "GAUSS=0.1"
\VAR VTURBHL= 50.0 
\VAR VRADHL = 50.0

****  MODEL FILES  ****
 
\VAR PATH =  ./test_data/
\VAR MODEL = T33_logg4.0_Mdot-8.5_logL4.72_v1994_vmic8_Z0.1

\EXPR MODEL    = $PATH // $MODEL
\EXPR FNAME    = $MODEL // /formal.plot
\EXPR FNAMEOUT = $MODEL // /formal.out
\EXPR FNAMEFLUX = $MODEL // /wruniq.plot

  
\VAR INCKEY_OPT = "* OPT" 

****  IDENTS  **** 
\VAR ident_file =  ~varshar/science/ullyses/ident_O.dat
*
****  Reddening is split into galactic and SMC contribution! ****
\VAR REDD = $EBV
\EXPR NEGREDD =  -1  * $REDD
\VAR  REDD_GAL = 0.04
\EXPR REDD = $REDD - $REDD_GAL
\EXPR REDD = $REDD //  " SMC" 
*
****  radial velocity  **** 
\CALC RADFAC = $VRAD / 300000.
\CALC RADFAC = 1. - $RADFAC
\CALC RADFACUV = $VRADUV / 300000. 
\CALC RADFACUV = 1. - $RADFACUV 
 
**** rotational velocity  ****
*
\VAR clight = 300000
 
\CALC dlamrot = LOG(1. + $vrot / $clight)
\CALC dlammac = LOG(1. + $vmac / $clight)

****  distance modulus  ****
*
\EXPR DILUTE  =  0.4 * $DM
\EXPR WRUNIQOUT  = $MODEL // /wruniq.out
\EXPR INFO = $MODEL // /modinfo.kasdefs
\VAR-LIST
\INCLUDE $INFO
\INSTRUCTION NO-EXPORT
\BGRLUN COLOR=4


******************************* GALEX ******************************************
* GALEX uses AB-Magniutes, conversion following:
* http://galexgi.gsfc.nasa.gov/docs/galex/FAQ/counts_background.html
* VAR names FUV NUV

\CALC logflambda = -0.4 * $FUV -7.322
\LUN LOG1565. $logflambda M.0 M.0 0.2 &0$FUV

\CALC logflambda = -0.4 * $NUV - 7.654
\LUN LOG2301. $logflambda M.0 M.0 0.2 &0$NUV

**************** GAIA DR 2 (Vega magnitudes) **************
* center wavelengths and zeropoints (ZP) from
* http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=GAIA&gname2=GAIA2
* VAR names : G Gbp Grp
* note: G is very broad (3300 - 10000 Ang and covers Gbp and Grp)
* lambdas are lambda_eff (Vega)

\VAR lambda = 5845.67
\CALC logflambda = -0.4 * $G - 8.604
\LUN LOG$lambda $logflambda M.0 M.0 0.2 &0$G

*\VAR lambda = 5017.28
*\CALC logflambda = -0.4 * $Gbp - 8.393
*\LUN LOG$lambda $logflambda M.0 M.0 0.2 &0G&Tbp&M

*\VAR lambda = 7593.40
*\CALC logflambda = -0.4 * $Grp - 8.900
*\LUN LOG$lambda $logflambda M.0 M.0 0.2 &0G&Trp&M

*
*
* Johnson U
* \IF $U .NE. ?
*\CALC lambda = 3971
*\CALC logflambda = -8.434 -0.4*$U
*\LUN LOG$lambda $logflambda M.0 M.0 0.2 &0$U
*\ENDIF
*
* Johnson B
\IF $B .NE. ?
\CALC lambda = 4481
\CALC logflambda = -8.184 - 0.4 * $B
\LUN LOG$lambda $logflambda M.0 M.0 0.2 &0$B
\ENDIF
* *
* * Johnson V
\IF $V .NE. ?
\CALC lambda = 5423
\CALC logflambda = -8.420 - 0.4 * $V
\LUN LOG$lambda $logflambda M.0 M.0 0.2 &0$V
\ENDIF
* *
* * Johnson R
\IF $R .NE. ?
\CALC lambda = 6441
\CALC logflambda = -8.643 -0.4 * $R
\LUN LOG$lambda $logflambda M.0 M.0 0.2 &0$R
\ENDIF
*
* Johnson I
*\IF $I .NE. ?
*\CALC lambda = 8071
*\CALC logflambda = -8.951 -0.4 * $I
*\LUN LOG$lambda $logflambda M.0 M.0 0.2 &0$I
*\ENDIF
*
\CALC logflambda = -9.519 -  0.4 * $J
\LUN LOG1.26E4 $logflambda M.0 M.0 0.2 &0$J
\CALC logflambda = -9.8998 -  0.4 * $H
\LUN LOG1.60E4 $logflambda M.0 M.0 0.2 &0$H
\CALC logflambda = -10.392 -  0.4 * $K
\LUN LOG2.22E4 $logflambda M.0 M.0 0.2 &0$K
* zero magnitude attributes from Jarret et al. 2011, ApJ 735, 112
* VAR names: W1 W2 W3 W4
* *
* \CALC logflambda = -11.0873 - 0.4 * $W1
* \LUN LOG3.3526E4 $logflambda M.0 M.0 0.2 &0$W1
* \CALC logflambda = -11.6171 - 0.4 * $W2
* \LUN LOG4.6028E4 $logflambda M.0 M.0 0.2 &0$W2
*\CALC logflambda = -13.1861 - 0.4 * $W3
*\LUN LOG11.5608E4 $logflambda M.0 M.0 0.2 &0$W3
*\CALC logflambda = -14.2933 - 0.4 * $W4
*\LUN LOG22.0883E4 $logflambda M.0 M.0 0.2 &0$W4
*
********* IRAC *********************************
* from Cohen et al. 2003, AJ, 125, 2645, Table 11
* VAR names: IRAC1 IRAC2 IRAC3 IRAC4
\CALC logflambda = -11.18 - 0.4 * $IRAC1
\LUN LOG3.6E4 $logflambda M.0 M.0 0.2 &0$IRAC1
CALC logflambda = -11.58 - 0.4 * $IRAC2
\LUN LOG4.5E4 $logflambda M.0 M.0 0.2 &0$IRAC2
*\CALC logflambda = -11.96 - 0.4 * $IRAC3
*\LUN LOG5.8E4 $logflambda M.0 M.0 0.2 &0$IRAC3
*\CALC logflambda = -12.52 - 0.4 * $IRAC4
*\LUN LOG8.0E4 $logflambda M.0 M.0 0.2 &0$IRAC4
\BGRLUN OFF
*
*******************************************************
********* START PLOTS *********************************
*******************************************************
 
\PEN=2
\LINUN XMIN 1. XMAX 1.  0. 0.
*\LUN  1600. YMIN  0.0 0.4  0.15  &E10x
*\LINUN  1900. 1.  3200. 1.  0. 0.
*\LINUN  1520. .1  1680. .1  0. 0.
*\LINUN  1650. 1.  2000. 1.  0. 0.
*IDENTS
***********************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT IUE SHORT"

HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux  
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     1130.0      1230.0       5.00     10.00     0.0000     
Y:  5.5CM      0       1.8        0.500     0.5     0.0000     

*****  MODELS  *****
N=? SYMBOL=5 SIZE=0.1 COLOR=2 PEN=3  
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND BINNING $bin
COMMAND CONVOL $CONVOL_HST
COMMAND INCLUDE $FNAME INCKEY="* IUE SHORT"

 
END

***********************************************
***********************************************
***********************************************
    
PLOT   : HST 

\LETTERSIZE 0.25
\OFS  2.0 15
\INBOX
*\NOCOPYRIGHT
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\DEFINECOLOR 7  1.0 0.58 0.0
\PEN=2
\LINUN XMIN 1. XMAX 1.  0. 0.
*\LUN  1600. YMIN  0.0 0.4  0.15  &E10x
*\LINUN  1900. 1.  3200. 1.  0. 0.
*\LINUN  1520. .1  1680. .1  0. 0.
*\LINUN  1650. 1.  2000. 1.  0. 0.
*IDENTS
***********************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT IUE SHORT"
 
HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux  
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     1230.0      1330.0        5.00    20.00     0.0000     
Y:  5.5CM      0.2       1.6        0.500     0.5     0.0000     

*****  MODELS  *****

N=? SYMBOL=5 SIZE=0.1 COLOR=2 PEN=3  
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND BINNING $bin
COMMAND CONVOL $CONVOL_HST
COMMAND INCLUDE $FNAME INCKEY="* IUE SHORT"
 
END

***********************************************
***********************************************
***********************************************
    
PLOT   : HST 

\LETTERSIZE 0.25
\OFS  2 8.5
\INBOX
*\NOCOPYRIGHT
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\DEFINECOLOR 7  1.0 0.58 0.0
\PEN=2
\LINUN XMIN 1. XMAX 1.  0. 0.
*\LUN  1600. YMIN  0.0 0.4  0.15  &E10x
*\LINUN  1900. 1.  3200. 1.  0. 0.
*\LINUN  1520. .1  1680. .1  0. 0.
*\LINUN  1650. 1.  2000. 1.  0. 0.
*IDENTS
***********************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT IUE SHORT"
 
HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux  
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     1330.0      1430.0        5.00    20.00     0.0000     
Y:  5.5CM      0.2       1.6        0.500     0.5     0.0000     

*****  MODELS  *****

N=? SYMBOL=5 SIZE=0.1 COLOR=2 PEN=3  
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND BINNING $bin
COMMAND CONVOL $CONVOL_HST
COMMAND INCLUDE $FNAME INCKEY="* IUE SHORT"
 
END

MULTIPLOT END
PAPERFORMAT A4H BBROTATE
MULTIPLOT START

PLOT   : HST 

\LETTERSIZE 0.25
\OFS  2.0 21.7
\INBOX
*\NOCOPYRIGHT
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\DEFINECOLOR 7  1.0 0.58 0.0
\PEN=2
\LINUN XMIN 1. XMAX 1.  0. 0.
*\LUN  1600. YMIN  0.0 0.4  0.15  &E10x
*\LINUN  1900. 1.  3200. 1.  0. 0.
*\LINUN  1520. .1  1680. .1  0. 0.
*\LINUN  1650. 1.  2000. 1.  0. 0.
*IDENTS

***********************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT IUE SHORT"
 
HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux  
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     1430.0      1550.0       5.00     10.00     0.0000     
Y:  5.5CM      0       1.8        0.500     0.5     0.0000     

*****  MODELS  *****


N=? SYMBOL=5 SIZE=0.1 COLOR=2 PEN=3  
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND BINNING $bin
COMMAND CONVOL $CONVOL_HST
COMMAND INCLUDE $FNAME INCKEY="* IUE SHORT"

 
END

***********************************************
***********************************************
***********************************************
    
PLOT   : HST 

\LETTERSIZE 0.25
\OFS  2.0 15
\INBOX
*\NOCOPYRIGHT
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\DEFINECOLOR 7  1.0 0.58 0.0
\PEN=2
\LINUN XMIN 1. XMAX 1.  0. 0.
*\LUN  1600. YMIN  0.0 0.4  0.15  &E10x
*\LINUN  1900. 1.  3200. 1.  0. 0.
*\LINUN  1520. .1  1680. .1  0. 0.
*\LINUN  1650. 1.  2000. 1.  0. 0.
*IDENTS
***********************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT IUE SHORT"
 
HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux  
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     1550.0      1670.0        5.00    20.00     0.0000     
Y:  5.5CM      0.2       1.6        0.500     0.5     0.0000     

*****  MODELS  *****


N=? SYMBOL=5 SIZE=0.1 COLOR=2 PEN=3  
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND BINNING $bin
COMMAND CONVOL $CONVOL_HST
COMMAND INCLUDE $FNAME INCKEY="* IUE SHORT"
 
END

***********************************************
***********************************************
***********************************************
    
PLOT   : HST 

\LETTERSIZE 0.25
\OFS  2 8.5
\INBOX
*\NOCOPYRIGHT
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\DEFINECOLOR 7  1.0 0.58 0.0
\PEN=2
\LINUN XMIN 1. XMAX 1.  0. 0.
*\LUN  1600. YMIN  0.0 0.4  0.15  &E10x
*\LINUN  1900. 1.  3200. 1.  0. 0.
*\LINUN  1520. .1  1680. .1  0. 0.
*\LINUN  1650. 1.  2000. 1.  0. 0.
*IDENTS
***********************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT IUE SHORT"
 
HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux  
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     1670.0      1790.0        5.00    20.00     0.0000     
Y:  5.5CM      0.2       1.6        0.500     0.5     0.0000     

*****  MODELS  *****


N=? SYMBOL=5 SIZE=0.1 COLOR=2 PEN=3  
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND BINNING $bin
COMMAND CONVOL $CONVOL_HST
COMMAND INCLUDE $FNAME INCKEY="* IUE SHORT"
 
END

MULTIPLOT END
PAPERFORMAT A4H BBROTATE

MULTIPLOT START


***********************************************************************
***********************************************************************
*
PLOT   : OPT-BLUE+MID
*
\LETTERSIZE 0.25
* 
\OFS  2.0 21.7
\INBOX
\NOCOPYRIGHT
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\PEN=1
\LINUN  XMIN 1.  XMAX 1.     0. 0.
*
**** IDENTS ************************************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT BLUE" 
************************************************************************
*
HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux  
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     4180.0     4480.0       20.00     100.00     4500
Y:  5.5CM      0.500       1.4         0.500     .500     0.0000  

*****  MODELS  *****

N=? SYMBOL=5 SIZE=0.08 COLOR=2 PEN=4
COMMAND SETNAME MODEL
COMMAND BINNING $bin
COMMAND CONVOL GAUSS $convol
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND INCLUDE $FNAME INCKEY=$INCKEY_OPT

END

*
PLOT   : OPT-BLUE+MID
*
\LETTERSIZE 0.25
* 
\OFS  2.0 15
\INBOX
\NOCOPYRIGHT
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\PEN=1
\LINUN  XMIN 1.  XMAX 1.     0. 0.
*
**** IDENTS ************************************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT BLUE" 
************************************************************************
*
HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux  
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     4480.0     4780.0       20.00     100.00     4500
Y:  5.5CM      0.500       1.4         0.500     .500     0.0000  
 
*****  MODELS  *****

N=? SYMBOL=5 SIZE=0.08 COLOR=2 PEN=4
COMMAND SETNAME MODEL
COMMAND BINNING $bin
COMMAND CONVOL GAUSS $convol
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND INCLUDE $FNAME INCKEY=$INCKEY_OPT

END

*
PLOT   : OPT-BLUE+MID
*
\LETTERSIZE 0.25
* 
\OFS  2.0 8.5
\INBOX
\NOCOPYRIGHT
\PENDEF=3
\DEFAULTCOLOR=4
\FONT=TIMES
\SET_NDATMAX = 600000
\PEN=1
\LINUN  XMIN 1.  XMAX 1.     0. 0.
*
**** IDENTS ************************************************************
\PEN=2
\ID_INBOX
\IDLENG 1
\IDSIZE 0.25
\IDY 1.3U
\INCLUDE $ident_file INCKEY="* IDENT BLUE" 
************************************************************************
*
HEADER :
X-ACHSE:\CENTER\#l# / \A
Y-ACHSE:\CENTER\Normalized flux   
    MASSTAB    MINIMUM    MAXIMUM    TEILUNGEN  BESCHRIFT. DARUNTER
X:  18.0CM     4780.0     5080.0       20.00     100.00     4500
Y:  5.5CM      0.500       1.4         0.500     .500     0.0000  
 
*****  MODELS  *****

N=? SYMBOL=5 SIZE=0.08 COLOR=2 PEN=4
COMMAND SETNAME MODEL
COMMAND BINNING $bin
COMMAND CONVOL GAUSS $convol
COMMAND XLOG
COMMAND CONVOL ROT $dlamrot
COMMAND CONVOL MACRO-RT $dlammac
COMMAND XDEX
COMMAND INCLUDE $FNAME INCKEY=$INCKEY_OPT

END

*
*
*
MULTIPLOT END