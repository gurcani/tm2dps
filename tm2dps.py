#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 20:35:36 2025

@author: ogurcan
"""

import numpy as np
import cupy as xp
from mlsarray.mlsarray import mlsarray,slicelist,init_kspace_grid,rfft2
from mlsarray.gensolver import gensolver,save_data
import h5py as h5
import os

filename='out.h5'
Npx,Npy=1024,1024
t0,t1=0.0,300.0
wecontinue=False
Nx,Ny=2*int(np.floor(Npx/3)),2*int(np.floor(Npy/3))
Lx,Ly=100,100
dkx,dky=2*np.pi/Lx,2*np.pi/Ly
sl=slicelist(Nx,Ny)
slbar=np.s_[int(Ny/2)-1:int(Ny/2)*int(Nx/2)-1:int(Nx/2)]

lkx,lky=init_kspace_grid(sl)
kx,ky=lkx*dkx,lky*dky
ksqr=kx**2+ky**2
Nk=kx.size
rtol,atol=1e-9,1e-12

w=10.0
phik=1e-6*xp.exp(-lkx**2/2/w**2-lky**2/2/w**2)*xp.exp(1j*2*np.pi*xp.random.rand(lkx.size).reshape(lkx.shape))
nk=1e-6*xp.exp(-lkx**2/2/w**2-lky**2/2/w**2)*xp.exp(1j*2*np.pi*xp.random.rand(lkx.size).reshape(lkx.shape))
phik[slbar]=0
nk[slbar]=0
zk=np.hstack((phik,nk))

nu=1e-2
D=1e-2
tau=1.0
kap=1.0
C=0.2
g=1e-2
nuH,nuL=0.0,0.0

if(wecontinue):
    fl=h5.File(filename,'r+',libver='latest')
    fl.swmr_mode = True
    zk=fl['last/zk'][()]
    t=fl['last/t'][()]
else:
    if os.path.exists(filename):
        os.remove(filename)
    fl=h5.File(filename,'w',libver='latest')
    fl.swmr_mode = True
    t=t0
    save_data(fl,'data',ext_flag=False,kx=kx.get(),ky=ky.get())
    save_data(fl,'params',ext_flag=False,C=C,kap=kap,g=g,tau=tau,nu=nu,D=D,Lx=Lx,Ly=Ly)
    save_data(fl,'last',ext_flag=False,zk=zk.get(),t=t)

def irft(uk):
    utmp=mlsarray(Npx,Npy)
    utmp[sl]=uk
    utmp[-1:-int(Nx/2):-1,0]=utmp[1:int(Nx/2),0].conj()
    utmp.irfft2()
    return utmp.view(dtype=float)[:,:-2]

def rft(u):
    uk=rfft2(u,norm='forward',overwrite_x=True).view(type=mlsarray)
    return xp.hstack(uk[sl])

def save_last(t,y):
    zk=y.view(dtype=complex)
    save_data(fl,'last',ext_flag=False,zk=zk.get(),t=t)

def save_real_fields(t,y):
    zk=y.view(dtype=complex)
    phik=zk[:Nk]
    om=irft(-ksqr*phik)
    nk=zk[Nk:]
    n=irft(nk)
    save_data(fl,'fields',ext_flag=True,om=om.get(),n=n.get(),t=t)

def save_fluxes(t,y):
    zk=y.view(dtype=complex)
    phik=zk[:Nk]
    nk=zk[Nk:]
    dyphi=irft(1j*ky*phik)
    om=irft(-ksqr*phik)
    n=irft(nk)
    Gam=np.mean(-dyphi*n,1)
    Pi=np.mean(-dyphi*om,1)
    save_data(fl,'fluxes',ext_flag=True,Gam=Gam.get(),Pi=Pi.get(),t=t)

def save_zonal(t,y):
    zk=y.view(dtype=complex)
    phik=zk[0:Nk]
    nk=zk[Nk:]
    vy=irft(-1j*kx*phik)
    om=irft(-ksqr*phik)
    n=irft(nk)
    save_data(fl,'fields/zonal/',ext_flag=True,vbar=xp.mean(vy,1).get(),ombar=xp.mean(om,1).get(),nbar=xp.mean(n,1).get(),t=t)

def fshow(t,y):
    zk=y.view(dtype=complex)
    phik=zk[0:Nk]
    nk=zk[Nk:]
    dyphi=irft(1j*ky*phik)
    n=irft(nk)
    Gam=np.mean(-dyphi*n)
    print('Gam=',Gam.get())

def rhs(t,y):
    zk=y.view(dtype=complex)
    dzkdt=xp.zeros_like(zk)
    phik,nk=zk[:Nk],zk[Nk:]
    dphikdt,dnkdt=dzkdt[:Nk],dzkdt[Nk:]
    dxphi=irft(1j*kx*phik)
    dyphi=irft(1j*ky*phik)
    dxn=irft(1j*kx*nk)
    dyn=irft(1j*ky*nk)
    sigk=xp.sign(ky)
    om=irft(-ksqr*phik)
    
    dphikdt[:]=(
        -(C*(1+tau*ksqr)+1j*tau*(g-kap)*ky*ksqr)*sigk*phik
        +(1j*ky*g*(1+tau*(1+ksqr))+C*(1+tau*ksqr))*sigk*nk
        +(1j*kx*rft(dyphi*om)-1j*ky*rft(dxphi*om))
        +(kx**2*rft(dxphi*dyn)-ky**2*rft(dyphi*dxn)+kx*ky*rft(dyphi*dyn-dxphi*dxn))
        )/ksqr-(nu*ksqr+nuH*ksqr**2+nuL/ksqr**2)*phik*sigk

    dnkdt[:]=-(1j*ky*g+C)*sigk*nk+(1j*ky*(g-kap)+C)*sigk*phik\
        +rft(dyphi*dxn-dxphi*dyn)\
        -(D*ksqr+nuH*ksqr**2+nuL/ksqr**2)*nk*sigk
        
    return dzkdt.view(dtype=float)

fsave=[save_last, save_fluxes, save_zonal, save_real_fields]
dtsave=[0.1,0.1,0.1,0.1]
dtstep,dtshow=0.1,0.1
r=gensolver('cupy_ivp.DOP853',rhs,t,zk.view(dtype=float),t1,fsave=fsave,fshow=fshow,dtstep=dtstep,dtshow=dtshow,dtsave=dtsave,dense=False,rtol=rtol,atol=atol)
r.run()
fl.close()
