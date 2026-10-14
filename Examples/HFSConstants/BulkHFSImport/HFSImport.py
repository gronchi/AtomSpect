"""Estimate hyperfine A constants from CSV inputs and compare measured values.

The semi-empirical estimator is a research aid; agreement for these inputs does
not validate other elements or states. Constants and comparison values are MHz.
Original example by Leo Nofs.
"""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from AtomSpect.hfs_estimate import Ahfs, pererror, read_HFSInput, powerset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path(__file__).with_name("HFSInputs.csv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent,
                        help="Directory for HFSOutput2.csv (default: this script's folder).")
    parser.add_argument("--no-show", action="store_true", help="Run without opening windows.")
    parser.add_argument("--scan-corrections", action="store_true",
                        help="Compare all combinations of correction flags.")
    parser.add_argument("--verbose", action="store_true", help="Print cases with large errors.")
    args = parser.parse_args(argv)
    if args.no_show:
        plt.switch_backend("Agg")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    HFSInputs = read_HFSInput(str(args.input))

    #%%CalculateInputs
    #Make it so that ds is included for s states,
    Corrections = 'FHdelepsds'

    for HFScalcs in HFSInputs:
            if 'Corrflags'  in HFScalcs:
                HFScalcs['Corrflags'] = HFScalcs['Corrflags'] + Corrections
            else:
                HFScalcs['Corrflags'] = Corrections


            HFScalcs['LongLable'] = f'{HFScalcs["Label"]}; {HFScalcs["Corrflags"]}'
            HFScalcs['A_Calc'] = Ahfs(HFScalcs)['A']
            HFScalcs['pererror'] = pererror(HFScalcs['A_Exp'],HFScalcs['A_Calc'])


    verboseresults = args.verbose
    if verboseresults:
        for HFScalcs in HFSInputs:
            if HFScalcs['pererror']>20: #Print out all the cases of error being more than 30%.

                print(f'Label: {HFScalcs["Label"]}')
                print(f'A_Exp: {HFScalcs["A_Exp"]}')
                print(f'A_Calc: {HFScalcs["A_Calc"]}')
                print(f'%Error: {HFScalcs["pererror"]}')
                print(f'%Corrs: {HFScalcs["Corrflags"]}')

                print('')


    with (args.output_dir / 'HFSOutput2.csv').open(mode='w', newline='', encoding='utf-8') as readme:
        fieldnames = HFSInputs[0].keys()

        writer = csv.DictWriter(readme, fieldnames = fieldnames)
        writer.writeheader()
        for things in HFSInputs:
            try:
                trow = {k: v for k, v in things.items()}
                writer.writerow(trow)
            except:
                pass


    #%%PlotInputDirectly

    plt.close('all')


    plt.figure()
    plt.title('Calculated Hyperfine A Constants')
    newcolors = plt.cm.rainbow(np.linspace(0, 1, len(HFSInputs)))
    plt.scatter(HFSInputs[0]['Label'], HFSInputs[0]['A_Exp'], color='black',alpha=0.75,marker='x', label='Exp')

    for t, things in enumerate(HFSInputs):
            plt.scatter(things['Label'],things['A_Calc'], s=100,color=newcolors[t],  label=things['LongLable'])
            plt.scatter(things['Label'], things['A_Exp'], color='black',alpha=0.75,marker='x')
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",borderaxespad=0)
            plt.xticks(rotation=90)


    plt.figure()
    plt.title('Percent Error for Input File')
    newcolors = plt.cm.rainbow(np.linspace(0, 1, len(HFSInputs)))

    for t, things in enumerate(HFSInputs):

            plt.scatter(things['Label'],things['pererror'], s=100,color=newcolors[t],  label=things['LongLable'])
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncols=2,borderaxespad=0)
            plt.xticks(rotation=90)


    #%%InputSpan
    #This section spans some input configurations to look for trends in calculation.
    #The goal was to try to determine where calculations might be going wrong, as well as what plays the most impact.
    Corrflaglist = ['F','H','R','del','eps','ds']

    Couple = ['None']
    spancorrflags = powerset(Corrflaglist)
    resultslist = []

    doinputspan = args.scan_corrections
    if doinputspan==True:

        minerror = []
        for HFScalcs in HFSInputs:
            tempminerror = {'Label' : HFScalcs['Label']      ,
                            'A_Exp' : HFScalcs['A_Exp']
                            }
            tmin = 100000

            for cc in Couple:
                HFScalcs['Coupling'] = cc
                for corrs in spancorrflags:
                    results = {}
                    HFScalcs['Corrflags'] = corrs #Set the flag for the correction factor for fast comparison of multiple settings.
                    results['A_Exp'] = HFScalcs['A_Exp']
                    results['A_Calc'] = Ahfs(HFScalcs)['A']
                    results['Label'] = HFScalcs['Label']
                    results['Atom'] = HFScalcs['Atom']
                    results['Corrflags'] = HFScalcs['Corrflags']
                    results['Coupling'] = cc

                    ttmin = pererror(HFScalcs['A_Exp'],results['A_Calc'])
                    results['pererror'] = ttmin
                    if ttmin<tmin:
                        tmin = np.min([tmin,ttmin])
                        tempminerror['A_Calc'] = Ahfs(HFScalcs)['A']
                        tempminerror['minerror'] = tmin
                        tempminerror['Corrflags'] = corrs
                        tempminerror['Coupling'] = cc
                        tempminerror['LongLable'] = f'{HFScalcs["Label"]}, {corrs}'

                    resultslist.append(results)
            minerror.append(tempminerror)


        plt.figure()
        plt.title('Corrections corresponding to minimum error')
        newcolors = plt.cm.rainbow(np.linspace(0, 1, len(minerror)))

        for tt, thingsa in enumerate(minerror):
            plt.scatter(thingsa['Label'],thingsa['A_Calc'], s=100,color=newcolors[tt], marker = 'o', label=thingsa['LongLable'])

            plt.scatter(thingsa['Label'],thingsa['A_Exp'], color='black',alpha=0.75,marker='x')

            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left",borderaxespad=0)
            plt.xticks(rotation=90)

        plt.figure()
        plt.title('Min Percent Error')
        newcolors = plt.cm.rainbow(np.linspace(0, 1, len(minerror)))

        for t, things in enumerate(minerror):
            plt.scatter(things['Label'],things['minerror'], s=100,color=newcolors[t],  label=things['LongLable'])
            plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncols=2,borderaxespad=0)
            plt.xticks(rotation=90)


    if verboseresults and doinputspan:
        for HFScalcs in minerror:
            if HFScalcs['minerror']>30: #Print out all the cases of error being more than 30%.

                print(f'Label: {HFScalcs["Label"]}')
                print(f'A_Exp: {HFScalcs["A_Exp"]}')
                print(f'A_Calc: {HFScalcs["A_Calc"]}')
                print(f'%Error: {HFScalcs["minerror"]}')
                print(f'%Corrs: {HFScalcs["Corrflags"]}')

                print('')


    #%%PlotAs
    PlotAllInOne = False
    if PlotAllInOne:
        #Let the x axis be the different states/labels. Each label should be unique. Then will use different markers/colors for the different correction factors.
        plt.close('all')
        plt.figure()
        newcolors = plt.cm.rainbow(np.linspace(0, 1, len(spancorrflags)+1))

        for i,res in enumerate(resultslist):
            if i<len(spancorrflags):
                plt.scatter(res['Label'],res['A_Calc'], s=3,color=newcolors[i%len(spancorrflags)],  label=res['Corrflags'])
            else:
                plt.scatter(res['Label'],res['A_Calc'], s=3,color=newcolors[i%len(spancorrflags)])

        ii=0
        for HFSins in HFSInputs:
            if ii==0:
                plt.scatter(HFSins['Label'], HFSins['A_Exp'], color='black',alpha=0.75,marker='x', label='Exp')
                ii+=1
            else:
                plt.scatter(HFSins['Label'], HFSins['A_Exp'], color='black',alpha=0.75,marker='x')

        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", ncols=2,borderaxespad=0)
        plt.tight_layout()

    #%%PerLevel (LSJ)
    #Each column will now have a zoomed in with all of the individual correction factors
    PlotSingleLevel = False
    if PlotSingleLevel:

        plt.close('all')

        Ncorr = len(spancorrflags)
        newcolors = plt.cm.rainbow(np.linspace(0, 1, len(spancorrflags)+1))


        for jj,res in enumerate(resultslist):
            if jj%Ncorr==0:
                plt.figure()
                plt.title(f'{res["Label"]}, Coupling={res["Coupling"]} , A={res["A_Exp"]} MHz')
                plt.hlines(res['A_Exp'], spancorrflags[0],Ncorr , color = 'black',label='Exp')
                plt.scatter(res['Corrflags'],res['A_Calc'], color = newcolors[jj%Ncorr])

            else:
                plt.scatter(res['Corrflags'],res['A_Calc'], color = newcolors[jj%Ncorr])
            plt.xticks(rotation=90)
            plt.grid(True,axis='x')

    print(f"Calculated {len(HFSInputs)} levels; wrote {args.output_dir / 'HFSOutput2.csv'}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
