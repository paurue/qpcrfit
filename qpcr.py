import sys,os,string,time,random
from datetime import datetime,date
from collections import OrderedDict 
import numpy as np
import pandas as pd
from scipy.optimize import minimize,curve_fit,leastsq,anneal,basinhopping
import jinja2 as j2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_valid_chars = "-_.()%s%s" % (string.ascii_letters, string.digits)

def get_valid_fname(fname):
    return ''.join(c if c in _valid_chars else '_' for c in fname)

def mak2(n,par=None,D0=1e-2,k=.5,Fb=0):
    """ MAK2 model from Boogy and Wolf 2010"""
    if par is not None:
        D0,k,Fb,slope=par
    D=np.zeros(n)
    D[0]=D0
    for i in range(n-1):
        D[i+1]=D[i]+k*np.log(1+D[i]/k)
    F=D+Fb
    E[1:]=k*np.log(1+D[:-1]/k)/D[:-1]
    return F,D,E

def mak3(n,par=None,D0=1e-2,k=.5,Fb=0,slope=0):
    """MAK2 model from Boogy and Wolf 2010, modified to include the slope effect"""
    if par is not None:
        D0,k,Fb,slope=par
    D=np.zeros(n)
    E=np.ones(n)
    D[0]=D0
    for i in range(n-1):
        D[i+1]=D[i]+k*np.log(1+D[i]/k)        
    F=D+Fb+np.arange(n)*slope
    E[1:]=k*np.log(1+D[:-1]/k)/D[:-1]
    return F,D,E

def mak3q(n,par=None,logD0=1e-2,logk=.5,Fb=0,slope=0):
    """MAK2 model from Boogy and Wolf 2010, modified to include the slope effect"""
    if par is not None:
        logD0,logk,Fb,slope=par
    D0=10**logD0
    k=10**logk
    D=np.zeros(n)
    E=np.ones(n)
    D[0]=D0
    for i in range(n-1):
        D[i+1]=D[i]+k*np.log(1+D[i]/k)        
    F=D+Fb+np.arange(n)*slope
    E[1:]=k*np.log(1+D[:-1]/k)/D[:-1]
    return F,D,E

class qpcrAnalysis:
    _html_template="""<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN">
<html lang="en">
<head>
  <title>qPCR Analysis - AMALab webtools</title>
    <link rel="stylesheet" type="text/css" media="all" href="/css/qpcrfit.css" />
</head>
<body>
  <div class="container">
      <div class="content">
        <div class="page-header">
          <div class="row">
            <p><h1>AMALab webtools - qPCR Analysis</h1></p>
          </div>
        </div>
    <h2>Summary</h2>
      <table>
        <tr><td>Job ID:</td>      <td>{{ ID }}</td></tr>    
        <tr><td>Experiment:</td>  <td>{{ expname }}</td></tr>    
        <tr><td>Submitted by:</td><td>{{ user }}</td></tr>    
        <tr><td>Date:</td>        <td>{{ date }}</td></tr>    
        <tr><td>Input file:</td>  <td>{{ ofn }}</td></tr>    
      </table>
    <h2>Summary results</h2>
      <h3>Initial concentrations (fitted) </h3>
        <img src="{{ figname_initialConcentration }}" height=250/>
        <p><a href={{ csvname_initialConcentration }}>CSV file</a></p>
      <h3> Details of fitting process </h3>
        <h4>Fitting error</h4>
        <img src="{{ figname_fittingError }}" height=250 />
        <h4>Kinetic constant (k)</h4>
        <img src="{{ figname_k }}" height=250 />
        <h4>Background fluorescence (Fb)</h4>
        <img src="{{ figname_Fb }}" height=250 />
        <h4>Fluorescence slope (s)</h4>
        <img src="{{ figname_slope }}" height=250 />
   <h2>Samples analysis</h2>
      {% for k,v in samples.items() recursive %}
        <h3>{{ k }}</h3>
        {%- if v %}                                                         
          {% for kk,vv in v.items() recursive %}
            <h4>{{ kk }}</h4>
            <img src="{{ vv }}" height=200/>
          {% endfor %}
        {%- endif %}
        <hr> 
      {% endfor %}
   <h2>Log</h2>
     <code>
      {{ log }}
     </code>
  </div>
</body>
</html>
"""
    def __init__(self,fname,expname,username=None,originalfname=None):
        self._start_time = time.time()
        self.info=dict()
        if originalfname is not None:
            self.info['ofn']=originalfname # original file name
        self.info['ifn']=fname # input file name
        self._log=""
        self._message("qPCR Analysis started.")
        if username is not None:
            self.info['username']=username
        else:
            self.info['username']=''
        self.info['expname']=expname
        # Generate an ID to prefix all generated files
        t=date.today()
        str_date="%04d%02d%02d"%(t.year,t.month,t.day)
        self.info['date']="%04d/%02d/%02d"%(t.year,t.month,t.day)
        str_exp=expname.replace(' ','_')

        str_code=''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(3))
        self.ID="%s_%s_%s_%s"%(str_date,str_exp,originalfname,str_code)
        self.info['ID']=self.ID
        self.info['resultsdir']="results/res_%s"%get_valid_fname(self.ID)
        os.mkdir("%s"%self.info['resultsdir'])

        d=pd.read_csv(fname)
        d=d.dropna(axis=0)
        nr,nc=d.shape
        nsamples=int((nc-1)/2)
        self.nvalues=nr
        #self._message("%d samples loaded, with %d cycles each."%(nsamples,self.nvalues))           
        # remove empty samples
        samples=[d.columns[i] for i in range(2,nc,2)]
        print(samples)
        aux=[d.columns[i].split(':')[1].strip() for i in range(2,nc,2)]
        samples_ok=[len(s)>0 for s in aux]
        self.samples=[samples[i] for i,ok in enumerate(samples_ok) if ok ]
        self.nsamples=len(self.samples)
        #print(nsamples,self.nsamples)
        self.info['samples']=OrderedDict()
        self.long_fitting={}
        for s in self.samples:
            self.info['samples'][s]=OrderedDict()
            cycles_with_detectable_signal=np.where(d.ix[:,s]>1)[0]
            if len(cycles_with_detectable_signal)==0:
                self.long_fitting[s]=True
                self._message("Warning: (%s) requires a long fitting procedure."%s)      
            elif np.min(cycles_with_detectable_signal)>33:     
                self.long_fitting[s]=True
                self._message("Warning: (%s) requires a long fitting procedure."%s)      
            else:
                self.long_fitting[s]=False
        self.samplesdata=d.ix[:,self.samples].as_matrix()
        self._fitted=False

    def __str__(self):
        text="qpcr Analysis: %s (ID: %s)\n" % (self.info['expname'],self.ID)
        text+="%d samples:\n\t"%len(self.info['samples'])+'\n\t'.join(self.info['samples'])
        return text
    def _compute_cutoffs(self):
        """Find the region to be fitted for each sample"""
        self._cutoffidx=np.zeros(self.nsamples,dtype=np.int)
        # Find the inlfection point
        # TODO: check robustness of this method against fluctuations in the data
        self.samplesdatadiff=np.diff(self.samplesdata,axis=0)
        flex=np.argmax(self.samplesdatadiff,axis=0)
        # if the detected cycles is the last one, then the flex has not yet been reached, warn.
        for i,f in enumerate(flex):
            #self._message("(%s) Preanalysis - detection of inflection point."%(self.samples[i]))           
            if f==(self.nvalues-1):
                self._cutoffidx[i]=f
                self._message("Warning: (%s) Inflection point not detected. Using all fluorescent values available (%d cycles)."%(self.samples[i],f))           
            elif f<10:
                self._message("Warning: (%s) Early inflection point (cycle %d)."%(self.samples[i],f))
            else:               
                self._cutoffidx[i]=np.minimum(f+2,self.nvalues)
                #self._message("(%s) Inflection point found at cycle %d)."%(self.samples[i],f))           

    def _prefit(self,costfunc,sample):
        """ Perform an exhaustive grid search to find good initial conditions"""
        # Here we obtain 10 initial candidates to best fitting
        #opt=100.
        if self.long_fitting[sample]:
            l1=np.linspace(-20,0,16)
            l2=np.linspace(-10,-0.3,16)
            l3=np.linspace(-0.5,0.5,16)
            l4=np.linspace(-0.015,0.015,16)
        else:
            l1=np.linspace(-20,0,5)
            l2=np.linspace(-10,-0.3,5)
            l3=np.linspace(-0.5,0.5,5)
            l4=np.linspace(-0.015,0.015,5)
        l1,l2,l3,l4=np.meshgrid(l1,l2,l3,l4)
        l1,l2,l3,l4=l1.ravel(),l2.ravel(),l3.ravel(),l4.ravel()
        #self._message("(%s) Prefitting data (grid search) - %d tests."%(sample,l1.size))
        costs=np.ones_like(l1)
        #optpar=np.array((l1[8],l2[8],l3[8],l4[8]))
        for j in range(l1.size):
            ll=np.array((l1[j],l2[j],l3[j],l4[j]))
            costs[j]=costfunc(ll)
        idx=np.argsort(costs)
        opt=costs[idx]
        optpars=np.column_stack((l1[idx],l2[idx],l3[idx],l4[idx]))
        #self._message(str(optpars.shape))
        #self._message("(%s) Prefitting done. Error achieved after prefitting: %s."%(sample,str(opt)))
        #self._message("(%s) Prefitting done. Error achieved after prefitting: %g."%(sample,opt[0]))
        if opt[0]>=1:
            self._message("Warning: (%s) Prefitting did not find a good initial condition (%g)."%(sample,opt[0]))
        return optpars,opt

    def fit(self):
        self._compute_cutoffs()
        self.mak3fluorescence=OrderedDict()
        self.mak3concentration=OrderedDict()
        self.mak3efficiency=OrderedDict()
        self.mak3fpre=OrderedDict()
        self.mak3cpre=OrderedDict()
        self.mak3epre=OrderedDict()
        self.initialConcentration=OrderedDict()
        self.k=OrderedDict()
        self.Fb=OrderedDict()
        self.slope=OrderedDict()
        self.fitting_error=OrderedDict()
        for i,s in enumerate(self.samples):
            x=self.samplesdata[:self._cutoffidx[i],i]
            print(i,s,self._cutoffidx[i])
            shift=x.min()
            x=x-shift
            def costfunc(q):
                Fest,Dest,Eest=mak3q(n=self._cutoffidx[i],par=q)
                sqdist=((x-Fest)**2).sum()
                return sqdist
            def costfunc0(q):
                Fest,Dest,Eest=mak3q(n=self._cutoffidx[i],par=q)
                dif=(x-Fest)
                return dif
            # Prefit
            q0,err0=self._prefit(costfunc,s)
            #self.mak3fpre[s],self.mak3cpre[s],self.mak3epre[s]=mak3q(n=self._cutoffidx[i],par=q0)
            #self.mak3fpre[s]+=shift

            # Optimize with Least-Squares
            #self._message("(%s) Least Squares fitting."%(s))
            q1=np.zeros_like(q0)
            err1=np.ones_like(err0)
            for j in range(100):
                #self._message("(%s) Fitting started. Error achieved after fitting:%g."%(s,err0[j]))
                q1[j,:],pc=leastsq(costfunc0,q0[j,:],maxfev=5000)
                err1[j]=costfunc(q1[j,:])
                #self._message("(%s) Fitting done. Error achieved after fitting:%g."%(s,err1[j]))
            idx=np.argsort(err1)
            erropt=err1[idx[0]]
            qopt=q1[idx[0],:]
            self.fitting_error[s]=erropt
            #self._message("(%s) Fitting done. Error achieved after fitting:%g."%(s,erropt))

            self.initialConcentration[s]=10**qopt[0]
            self.k[s]=10**qopt[1]
            self.Fb[s]=qopt[2]+shift
            self.slope[s]=qopt[3]
            self.mak3fluorescence[s],self.mak3concentration[s],self.mak3efficiency[s]=mak3q(n=self._cutoffidx[i],par=qopt)
            self.mak3fluorescence[s]+=shift
        self._fitted=True

    def to_csv(self):
        """Generates an Excel report of the fitting"""
        if not self._fitted:
            self.fit()
        #self._message("Saving results into a csv (comma separated values) file.")
        v=np.array([list(self.initialConcentration.values()),
           list(self.fitting_error.values()),
           list(self.k.values()),
           list(self.Fb.values()),
           list(self.slope.values())]).T
        k=list(self.initialConcentration.keys())
        d=pd.DataFrame(v,columns=['Initial Concentration','Fitting Error','k','Fb','Slope'],index=k)
        fn=get_valid_fname(self.ID)
        self.csvname="%s_initial_concentrations.csv"%(fn)
        self.fullcsvname="%s/%s_initial_concentrations.csv"%(self.info['resultsdir'],fn)
        self.info['csvname_initialConcentration']=self.csvname
        print(self.csvname)
        d.to_csv('%s/%s'%(self.info['resultsdir'],self.csvname))

    def save_figs(self):
        """Generates an HTML report of the fitting"""
        if not self._fitted:
            self.fit()
        #self._message("Saving plots...")
        # 1. Generate the required PNG plots
        # 1.1 Truncation plots
        for i,s in enumerate(self.samples):
            fig,ax=plt.subplots(1,2,figsize=(8,4))
            cyct=np.arange(self.nvalues)
            cycf=np.arange(self._cutoffidx[i])
            cycd=0.5*(cyct[1:]+cyct[:-1])
            ax[0].plot(cyct,self.samplesdata[:,i],'k.-',linewidth=0.5,label="Full series")
            ax[0].plot(cycf,self.samplesdata[:self._cutoffidx[i],i],'r-',linewidth=1,label="Truncated")
            ax[0].set_xlim([0,self.nvalues-1])
            ax[0].set_ylim([0,self.samplesdata.max()*1.1])
            ax[0].set_xlabel("Cycle")
            ax[0].set_ylabel("Fluorescence (a.u.)")
            ax[0].set_title("Detected fluorescence")
            plt.legend(loc='upper left',frameon=False)
            # First derivative
            ax[1].plot(cycd,self.samplesdatadiff[:,i],'k.-',linewidth=0.5)
            ax[1].axvline(self._cutoffidx[i],color='r')
            ax[1].set_xlim([0,self.nvalues-1])
            ax[1].set_ylim([self.samplesdatadiff.min()*1.1,self.samplesdatadiff.max()*1.1])
            ax[1].set_xlabel("Cycle")
            ax[1].set_ylabel("dF/dCycle (a.u.)")
            ax[1].set_title("Fluorescence rate")
            plt.tight_layout()
            fn=get_valid_fname(self.samples[i])
            figname="%s_%s_%s.svg"%(self.ID,"01truncation",fn)
            self.info['samples'][s]['Data truncation for fitting']=figname
            plt.savefig('%s/%s'%(self.info['resultsdir'],figname))
            plt.close()        
        # 1.2 Fitting plots
        for i,s in enumerate(self.samples):
            fig,ax=plt.subplots(1,3,figsize=(12,4))
            cyct=np.arange(self.nvalues)
            cycf=np.arange(self._cutoffidx[i])
            ax[0].plot(cyct,self.samplesdata[:,i],'k:',linewidth=0.5,label="Full series")
            ax[0].plot(cycf,self.samplesdata[:self._cutoffidx[i],i],'r.-',linewidth=0.5,label="Truncated")
            #ax[0].plot(cycf,self.mak3fpre[s],'y-',linewidth=1,label="prefit")
            ax[0].plot(cycf,self.mak3fluorescence[s],'g-',linewidth=1,label="MAK3 fit")
            ax[0].axvline(self._cutoffidx[i],color='k')
            ax[0].set_xlim([0,self.nvalues-1])
            ax[0].set_ylim([0,self.samplesdata.max()*1.1])
            ax[0].set_xlabel("Cycle")
            ax[0].set_ylabel("Fluorescence (a.u.)")
            ax[0].set_title("Detected fluorescence")
            ax[0].legend(loc='upper left',frameon=False)
            # DNA levels
            ax[1].plot(cycf,self.mak3concentration[s],'g-',linewidth=1,label="MAK3")
            ax[1].axvline(self._cutoffidx[i],color='k')
            ax[1].set_xlim([0,self.nvalues-1])
            ax[1].set_ylim([0,self.mak3concentration[s].max()*1.1])
            ax[1].set_xlabel("Cycle")
            ax[1].set_ylabel("concentration (a.u.)")
            ax[1].set_title("estimated cDNA levels")
            # Efficiency
            ax[2].plot(cycf,self.mak3efficiency[s],'b-',linewidth=1,label="MAK3")
            ax[2].axvline(self._cutoffidx[i],color='k')
            ax[2].set_xlim([0,self.nvalues-1])
            ax[2].set_ylim([0,1.1])
            ax[2].set_xlabel("Cycle")
            ax[2].set_ylabel("Efficiency")
            ax[2].set_title("Amplification efficiency")         
            plt.tight_layout()
            fn=get_valid_fname(self.samples[i])
            figname="%s_%s_%s.svg"%(self.ID,"02mak3",fn)
            self.info['samples'][s]['MAK3 Fitting']=figname
            plt.savefig('%s/%s'%(self.info['resultsdir'],figname))
            plt.close()
        # 2 Initial concentrations
        figwdth=np.maximum(5,0.4*self.nsamples+1)
        fig,ax=plt.subplots(1,1,figsize=(figwdth,7))
        v=list(self.initialConcentration.values())
        k=list(self.initialConcentration.keys())
        ax.bar(0.75+np.arange(self.nsamples),v,facecolor='k',width=0.5)
        ax.set_xticks(1+np.arange(self.nsamples))
        ax.set_xticklabels(k,rotation=90)
        ax.set_xlim([0,self.nsamples+1])
        plt.tight_layout()
        figname="%s_%s_.svg"%(self.ID,"00initialConcentration")
        self.info['figname_initialConcentration']=figname
        plt.savefig('%s/%s'%(self.info['resultsdir'],figname))
        plt.close()
        # 3 Fitting Error
        fig,ax=plt.subplots(1,1,figsize=(figwdth,7))
        v=list(self.fitting_error.values())
        k=list(self.fitting_error.keys())
        ax.bar(0.75+np.arange(self.nsamples),v,facecolor='k',width=0.5)
        ax.set_xticks(1+np.arange(self.nsamples))
        ax.set_xticklabels(k,rotation=90)
        ax.set_xlim([0,self.nsamples+1])
        ax.set_ylim([0,1e-2])
        plt.tight_layout()
        figname="%s_%s_.svg"%(self.ID,"00fittingError")
        self.info['figname_fittingError']=figname
        plt.savefig('%s/%s'%(self.info['resultsdir'],figname))
        # 4 kinetic constant
        fig,ax=plt.subplots(1,1,figsize=(figwdth,7))
        v=list(self.k.values())
        k=list(self.k.keys())
        ax.bar(0.75+np.arange(self.nsamples),v,facecolor='k',width=0.5)
        ax.set_xticks(1+np.arange(self.nsamples))
        ax.set_xticklabels(k,rotation=90)
        ax.set_xlim([0,self.nsamples+1])
        plt.tight_layout()
        figname="%s_%s_.svg"%(self.ID,"00kineticConstant")
        self.info['figname_k']=figname
        plt.savefig('%s/%s'%(self.info['resultsdir'],figname))
        # 5 background fluorescence
        fig,ax=plt.subplots(1,1,figsize=(figwdth,7))
        v=list(self.Fb.values())
        k=list(self.Fb.keys())
        ax.bar(0.75+np.arange(self.nsamples),v,facecolor='k',width=0.5)
        ax.set_xticks(1+np.arange(self.nsamples))
        ax.set_xticklabels(k,rotation=90)
        ax.set_xlim([0,self.nsamples+1])
        plt.tight_layout()
        figname="%s_%s_.svg"%(self.ID,"00bkgFluorescence")
        self.info['figname_Fb']=figname
        plt.savefig('%s/%s'%(self.info['resultsdir'],figname))
        # 6 slope
        fig,ax=plt.subplots(1,1,figsize=(figwdth,7))
        v=list(self.slope.values())
        k=list(self.slope.keys())
        ax.bar(0.75+np.arange(self.nsamples),v,facecolor='k',width=0.5)
        ax.set_xticks(1+np.arange(self.nsamples))
        ax.set_xticklabels(k,rotation=90)
        ax.set_xlim([0,self.nsamples+1])
        ax.set_ylim([0,0.025])
        plt.tight_layout()
        figname="%s_%s_.svg"%(self.ID,"00fluorescenceSlope")
        self.info['figname_slope']=figname
        plt.savefig('%s/%s'%(self.info['resultsdir'],figname))

    def _message(self,message):
        self._log+=str(datetime.now()) + "| %s<br/>"%message

    def to_html(self):
        self.to_csv()
        self.save_figs()
        elapsed=int(time.time() - self._start_time)
        self._message("qPCR Analysis ended. Time elapsed:%d minutes and %d seconds" %(int(elapsed/60),elapsed%60))
        self.info['log']=self._log
        self.htmlfname='%s/%s.html'%(self.info['resultsdir'],self.info['ID'])
        template=j2.Template(self._html_template)
        self.html_string=template.render(self.info)
        f=open(self.htmlfname,'w')
        f.write(self.html_string)
        f.close()        
