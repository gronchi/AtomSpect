"""Estimate hyperfine A constants from hand-written atomic input dictionaries.

Prints single-level estimates and an example of coupling two electrons.
These semi-empirical calculations are research examples, not validated atomic data.
Original example by Leo Nofs.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AtomSpect.hfs_estimate import Ahfs


def main():
    #%%ManualDictTests
    domanualDicts = True
    if domanualDicts:


        #%%Li72s12

        Li72s12 = {'Z' : 3  , #Atomic number
                'AMU': 7  , #Atomic mass in amu
                'E_ionize': 5.391715 , #Ionization energy [eV]
                'E1':0 , #Energy of electron [eV]
                'n1': 2     , #Princple quantum number
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 3.256427 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  : 0   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question.
                'j1'  : .5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'ds' , #Correction factors for the hyperfine calculation. Possible flags: 'F H R del eps ds'
                'E2':3.37313 , #Energy of the second level for dsdn correction (Only used for s states typically)
                'n2':3 ,  #Principle quantum number of the second level for dsdn correction (Only used for s states typically)
                'A_Exp': 401 , #Experimental A in MHZ for easy comparison
             }

        Li72s12A = Ahfs(Li72s12)
        print(f'Li72s12 : Calc = {Li72s12A}, Exp = {Li72s12["A_Exp"]}')
        #%%Li2p12
        Li72p12 = {'Z' : 3  , #Atomic number
                'AMU': 7  , #Atomic mass in amu
                'E_ionize': 5.391715 , #Ionization energy [eV]
                'E1':1.847846  , #Energy of electron [eV]
                'n1': 2     , #Princple quantum number
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 3.256427 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  : 1   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : -0.5 ,  #Spin of the electron in question.
                'j1'  : .5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'FRHdelepsds' , #Correction factors for the hyperfine calculation. Possible flags: 'F H R del eps ds'
                'E2': 3.834258 , #Energy of the second level for dsdn correction (Only used for s states typically)
                'n2':3 ,  #Principle quantum number of the second level for dsdn correction (Only used for s states typically)
                'A_Exp': 45 , #Experimental A in MHZ for easy comparison
             }

        Li72p12A = Ahfs(Li72p12)
        print(f'Li72p12 : Calc = {Li72p12A}, Exp = {Li72p12["A_Exp"]}')


        #%%Li2p32
        Li72p32 = {'Z' : 3  , #Atomic number
                'AMU': 7  , #Atomic mass in amu
                'E_ionize': 5.391715 , #Ionization energy [eV]
                'E1':1.847846  , #Energy of electron [eV]
                'n1': 2     , #Princple quantum number
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 3.256427 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  : 1   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question.
                'j1'  : 1.5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'FRHdelepsds' , #Correction factors for the hyperfine calculation. Possible flags: 'F H R del eps ds'
                'E2': 3.834258 , #Energy of the second level for dsdn correction (Only used for s states typically)
                'n2':3 ,  #Principle quantum number of the second level for dsdn correction (Only used for s states typically)
                'A_Exp': -3 , #Experimental A in MHZ for easy comparison
             }

        Li72p32A = Ahfs(Li72p32)
        print(f'Li72p32 : Calc = {Li72p32A}, Exp = {Li72p32["A_Exp"]}')


      # %%Li73p12


        Li73p12 = {'Z' : 3 , #Atomic number
                'AMU': 7  , #Atomic mass in amu
                'E_ionize':5.391715 , #Ionization energy [eV]
                'E1': 3.834258  , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 3     , #Princple quantum number.
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 3.256427 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  : 1   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : -0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  :.5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'FRHdelepsds' , #Correction factors for the hyperfine calculation.
                'E2': 4.521648  ,#Energy of the second level for dsdn correction (Only used for s states typically)
                'n2':4 ,  #Principle quantum number of the second level for dsdn correction (Only used for s states typically)
                'A_Exp': 13.5,  #Experimental A in MHZ for easy comparison

             }
        Li73p12A = Ahfs(Li73p12)
        print(f'Li73p12 : Calc = {Li73p12A}, Exp = {Li73p12["A_Exp"]}')
    # %%Li73p32


        Li73p32 = {'Z' : 3 , #Atomic number
                'AMU': 7  , #Atomic mass in amu
                'E_ionize':5.391715 , #Ionization energy [eV]
                'E1': 3.834258  , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 3     , #Princple quantum number.
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 3.256427 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  : 1   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  :1.5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'FHdelepsds' , #Correction factors for the hyperfine calculation.
                'E2': 4.521648  ,#Energy of the second level for dsdn correction (Only used for s states typically)
                'n2':4 ,  #Principle quantum number of the second level for dsdn correction (Only used for s states typically)
                'A_Exp': -0.96,  #Experimental A in MHZ for easy comparison

             }
        Li73p32A = Ahfs(Li73p32)
        print(f'Li73p32 : Calc = {Li73p32A}, Exp = {Li73p32["A_Exp"]}')

        #%%Na3s12
        Na3s12 = {'Z' :  11, #Atomic number
                'AMU': 23  , #Atomic mass in amu
                'E_ionize':5.13907696 , #Ionization energy [eV]

                'E1': 0    , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 3     , #Princple quantum number.
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI':2.217522 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  :  0   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : .5   , #J = L +/- S coupled angular momentum
                'E2': 3.191353,


                'n2':4 ,

                'A_Exp': 885,  #Came from Allegrini

             }


        Na3s12A = Ahfs(Na3s12)
        print(f'Na3s12 : Calc = {Na3s12A}, Exp = {Na3s12["A_Exp"]}')
        #%%Na3d52

        Na3d52 = {'Z' :  11, #Atomic number
                'AMU': 23  , #Atomic mass in amu
                'E_ionize':5.13907696 , #Ionization energy [eV]

                'E1': 3.6169733    , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 3     , #Princple quantum number.
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 2.217522 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  :  2   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : 2.5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'RFHdelepsds' , #Correction factors for the hyperfine calculation.
                'E2': 4.283498,


                'n2':4 ,

                'A_Exp': 0.108 ,  #Came from Allegrini

             }


        Na3d52A = Ahfs(Na3d52)
        print(Na3d52A)
        #%%Na3d32

        Na3d32 = {'Z' :  11, #Atomic number
                'AMU': 23  , #Atomic mass in amu
                'E_ionize':5.13907696 , #Ionization energy [eV]

                'E1': 3.6169733    , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 3     , #Princple quantum number.
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 2.217522 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  :  2   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : -0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : 1.5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'RFHdelepsds' , #Correction factors for the hyperfine calculation.
                'E2': 4.283498,


                'n2':4 ,

                'A_Exp': 0.527,

             }


        Na3d32A = Ahfs(Na3d32)
        print(Na3d32A)


          #%%Na3p32

        Na3p32 = {'Z' : 11   , #Atomic number
                'AMU': 23  , #Atomic mass in amu
                'E_ionize':5.13907696 , #Ionization energy [eV]

                'E1': 2.103719    , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 3     , #Princple quantum number.
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 2.217522 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  :  1   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : 1.5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'FHdelepsds' , #Correction factors for the hyperfine calculation.
                'E2': 3.753091,


                'n2':4 ,

                'A_Exp': 18.8 ,

             }


        Na3p32A = Ahfs(Na3p32)
        print(Na3p32A)


    #%%Na3p12
        Na3p12 = {'Z' : 11   , #Atomic number
                'AMU': 23  , #Atomic mass in amu
                'E_ionize':5.13907696 , #Ionization energy [eV]

                'E1': 2.103719    , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 3     , #Princple quantum number.
                'I' : 1.5    , #Nuclear spin angular momentum
                'muI': 2.217522 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  :  1   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : .5   , #J = L +/- S coupled angular momentum
                'Corrflags': 'RFHdelepsds' , #Correction factors for the hyperfine calculation.
                'E2': 3.753091,


                'n2':4 ,

                'A_Exp': 94.44,  #If I calculate this the same way I do for the other l-1/2, I get a much different answer...


             }


        Na3p12A = Ahfs(Na3p12)
        print(Na3p12A)


        #%%Rb85Corr
        Rb5s12 = {'Z' : 37   , #Atomic number
                'AMU': 85  , #Atomic mass in amu
                'E_ionize': 4.177128 , #Ionization energy [eV]
                'E1': 0   , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 5     , #Princple quantum number.
                'I' : 2.5    , #Nuclear spin angular momentum
                'muI': 1.35298 ,    #Nuclear dipole moment (in units of mu_N)

                'l1'  :  0   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : .5   , #J = L +/- S coupled angular momentum

                'E2' :  2.496113 , #Optional. For s electrons, needed for qdefect correction factor
                'n2' :  6   , #Optional, n2>n1. Needed for qdefect correction factor of s orbitals.
                'Corrflags' : 'FHepsdelds' ,
                'A_Exp': 1011,

                    }

        Rb5s12A = Ahfs(Rb5s12)
        print(Rb5s12A)


        Rb5p12 = {'Z' : 37   , #Atomic number
                'AMU': 85  , #Atomic mass in amu
                'E_ionize': 4.177128 , #Ionization energy [eV]

                'E1': 1.57923     , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 5     , #Princple quantum number.
                'I' : 2.5    , #Nuclear spin angular momentum
                'muI': 1.35298 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  : 1   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : .5   , #J = L +/- S coupled angular momentum
                'E2' :  2.946702 , #Optional. For s electrons, needed for qdefect correction factor
                'n2' :  6    ,#Optional, n2>n1. Needed for qdefect correction factor of s orbitals.
                'A_Exp': 120,
                    }

        Rb5p12A = Ahfs(Rb5p12)
        print(Rb5p12A)


        Rb5p32 = {'Z' : 37   , #Atomic number
                'AMU': 85  , #Atomic mass in amu
                'E_ionize': 4.177128 , #Ionization energy [eV]

                'E1': 1.57923     , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 5     , #Princple quantum number.
                'I' : 2.5    , #Nuclear spin angular momentum
                'muI': 1.35298 ,    #Nuclear dipole moment (in units of mu_N)
                'l1'  : 1   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 0.5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : 1.5   , #J = L +/- S coupled angular momentum
                'E2' :  2.946702 , #Optional. For s electrons, needed for qdefect correction factor
                'n2' :  6    ,#Optional, n2>n1. Needed for qdefect correction factor of s orbitals.
                'Corrflags' : 'Hdelepsds' ,
                'A_Exp': 25.009,
                    }
        RBp32 = Ahfs(Rb5p32)
        print(RBp32)


        W7SOut = {'Z' : 74   , #Atomic number
                'AMU': 183  , #Atomic mass in amu
                'E_ionize': 7.86403 , #Ionization energy [eV]
                'E1': -.18667     , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 6     , #Princple quantum number.
                'I' : 0.5    , #Nuclear spin angular momentum
                'muI': 0.1095569 ,    #Nuclear dipole moment (in units of mu_N)

                'l1'  :  0   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' :  .5 ,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : .5   , #J = L +/- S coupled angular momentum

               'E2' :5.310495 ,
               'n2' : 7 ,
                'Corrflags': 'Fdelepsds' , #Correction factors for the hyperfine calculation.

                'A_Exp': 496,

             }

        W7SOutA = Ahfs(W7SOut)
        print(W7SOutA)
        W7Sin = {'Z' : 74   , #Atomic number
                'AMU': 183  , #Atomic mass in amu

                'E_ionize': 16.37 , #Ionization energy [eV]
                'E1': .40624084     , #Term energy for calculation [eV]. For l=0, this has the lower principle quantum number. Used for qdefect calc and correction.
                'n1': 5    , #Princple quantum number.
                'I' : 0.5    , #Nuclear spin angular momentum
                'muI': 0.1095569 ,    #Nuclear dipole moment (in units of mu_N)

                'l1'  :  0   , #Orbital Angular Momentum S=0, L=1, P=2, D=3, F=4...etc
                's1' : 2.5,  #Spin of the electron in question. For coupling two electrons, s1 is the "inner" and s2 is "outer/excited".
                'j1'  : 2.5   , #J = L +/- S coupled angular momentum


                'Corrflags': 'FHRdeleps' , #Correction factors for the hyperfine calculation.


                'A_Exp': 496,

             }

        W7SinA = Ahfs(W7Sin)
        print(W7SinA)
    def GenCouple(A,B,C,D):
        #Couples (A(A+1) + B(B+1) - C(C+1)) /(2D(D+1)) since it is such a common calculation, esp for testing.
        return (A*(A+1) + B*(B+1) - C*(C+1)) /(2*D*(D+1))
    C1 = GenCouple(3,2.5,.5,3)
    C2 = GenCouple(3,.5,2.5,3)

    print(C1, C1*W7SinA['A'])
    print(C2, C2*W7SOutA['A'])

    A2 = (C1*W7SinA['A'] + C2*W7SOutA['A'])
    print(A2)


if __name__ == "__main__":
    main()
