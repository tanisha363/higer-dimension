from django.shortcuts import render,redirect
from django.http import HttpResponse
import os, sys, getopt, pdb
from .forms import *
from numpy import *
from numpy.linalg import *
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.random_projection import SparseRandomProjection
from sklearn.random_projection import GaussianRandomProjection
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def home(request):
    if request.method == 'POST':
        form = FileForm(request.POST, request.FILES)
        filn = request.POST['file_name']
        if form.is_valid():
            form.save()
            context = {'img_cond':request.POST['selection']}
            
            mds(filn)
            sprndprojection(filn)
            gaurndprojection(filn)
            isomap(filn)
            Tsne(filn)
            return render(request,'result.html',context)  
    else:
        form = FileForm()    
        context={'form':form}
        return render(request,'home.html',context) 


def mds(filn):
    df = pd.read_csv('media/'+str(filn))
 
    standardized_data = StandardScaler().fit_transform(df)
    standardized_data.shape

    mds = MDS(n_components=2, n_init=4, max_iter=1, metric=True, n_jobs=1, random_state=0)
    reduced_data = mds.fit_transform(standardized_data)


    reduced_df = np.vstack((reduced_data.T)).T
    reduced_df = pd.DataFrame(data=reduced_df, columns=["X", "Y"])

    reduced_df.head() 
    reduced_df.dtypes
    g = sns.FacetGrid(reduced_df, height=6).map(plt.scatter, 'X', 'Y').add_legend()
    plt.savefig('static/imgs/fig2.png')
def sprndprojection(filn):
    
    df = pd.read_csv('media/'+str(filn))
 
    standardized_data = StandardScaler().fit_transform(df)
    standardized_data.shape

    srp = SparseRandomProjection(n_components=2,density='auto',eps=0.5, random_state=0)
    reduced_data = srp.fit_transform(standardized_data)


    reduced_df = np.vstack((reduced_data.T)).T
    reduced_df = pd.DataFrame(data=reduced_df, columns=["X", "Y"])

    reduced_df.head()
    
    reduced_df.dtypes
    g = sns.FacetGrid(reduced_df, height=6).map(plt.scatter, 'X', 'Y').add_legend()
    plt.savefig('static/imgs/fig3.png')
def gaurndprojection(filn):
    
    df = pd.read_csv('media/'+str(filn))
 
    standardized_data = StandardScaler().fit_transform(df)
    standardized_data.shape

    grp = GaussianRandomProjection(n_components=2,eps=0.5, random_state=0)
    reduced_data = grp.fit_transform(standardized_data)

    reduced_df = np.vstack((reduced_data.T)).T
    reduced_df = pd.DataFrame(data=reduced_df, columns=["X", "Y"])

    reduced_df.head()
    
    reduced_df.dtypes
    g = sns.FacetGrid(reduced_df, height=6).map(plt.scatter, 'X', 'Y').add_legend()
    plt.savefig('static/imgs/fig4.png')
def isomap(filn):
    
    df = pd.read_csv('media/'+str(filn))
 
    standardized_data = StandardScaler().fit_transform(df)
    standardized_data.shape

    isomap = Isomap(n_components=2,n_jobs=1,n_neighbors=1)
    reduced_data = isomap.fit_transform(standardized_data)

    reduced_df = np.vstack((reduced_data.T)).T
    reduced_df = pd.DataFrame(data=reduced_df, columns=["X", "Y"])

    reduced_df.head()
    
    reduced_df.dtypes
    g = sns.FacetGrid(reduced_df, height=6).map(plt.scatter, 'X', 'Y').add_legend()
    plt.savefig('static/imgs/fig5.png')
def Tsne(filn):
    
    df = pd.read_csv('media/'+str(filn))
 
    standardized_data = StandardScaler().fit_transform(df)
    standardized_data.shape

    tsne = TSNE(random_state=0, perplexity=30, learning_rate=200, n_iter=1000)
    reduced_data = tsne.fit_transform(standardized_data)

    reduced_df = np.vstack((reduced_data.T)).T
    reduced_df = pd.DataFrame(data=reduced_df, columns=["X", "Y"])

    reduced_df.head()
    
    reduced_df.dtypes
    g = sns.FacetGrid(reduced_df, height=6).map(plt.scatter, 'X', 'Y').add_legend()
    plt.savefig('static/imgs/fig6.png')


