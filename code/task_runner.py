import numpy as np, math
from decimal import Decimal
from math import comb
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import root_scalar, minimize, root
from scipy.integrate import simpson, solve_ivp

def jacobi(A,b,tol=1e-8,limit=500):
    D=np.diag(A); R=A-np.diagflat(D); x=np.zeros_like(b); errors=[]
    for _ in range(limit):
        xn=(b-R@x)/D; e=np.linalg.norm(xn-x,np.inf); errors.append(e)
        if e<tol: return xn,errors
        x=xn
    return x,errors

def gauss_seidel(A,b,tol=1e-8,limit=500):
    x=np.zeros_like(b); errors=[]
    for _ in range(limit):
        old=x.copy()
        for i in range(len(b)): x[i]=(b[i]-A[i,:i]@x[:i]-A[i,i+1:]@old[i+1:])/A[i,i]
        e=np.linalg.norm(x-old,np.inf); errors.append(e)
        if e<tol: return x,errors
    return x,errors

def run_task(k):
    if k=="A":
        x=np.array([62,64,66,67,69,70,68,64,58,52,47,44,42,41,43,46,51,55,58,60,61,62,63,63.])
        m=x>65
        print("Greenhouse humidity monitoring"); print(f"Mean: {x.mean():.2f}% | Min: {x.min():.0f}% | Max: {x.max():.0f}%")
        print("Hours above 65%:",np.where(m)[0].tolist(),"count:",m.sum()); print(f"Mean during affected hours: {x[m].mean():.2f}%")
    elif k=="B":
        x=np.random.default_rng(1).uniform(50,20000,1_000_000); net=x/1.16; vat=x-net
        print("VAT on one million retail transactions"); print(f"Gross: KES {x.sum():,.2f}"); print(f"Net: KES {net.sum():,.2f}"); print(f"VAT: KES {vat.sum():,.2f}"); print(f"VAT share: {vat.sum()/x.sum():.4%}")
    elif k=="C":
        x=0.
        for _ in range(1_000_000): x+=.10
        print("Banking fees and floating-point arithmetic"); print("Naive:",repr(x)); print("math.fsum:",math.fsum(.10 for _ in range(1_000_000)))
        print("Decimal:",Decimal("0.10")*1_000_000); print("Naive error:",abs(x-100000))
    elif k=="D":
        p=np.array([[0,0],[12,0],[12,8],[0,8.]])
        a=np.deg2rad(30); R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        M=np.eye(3); M[:2,:2]=R@np.diag([.5,.5]); M[:2,2]=[5,3]
        out=(np.c_[p,np.ones(4)]@M.T)[:,:2]
        print("Floor plan on a map"); print("Homogeneous matrix:\\n",np.round(M,4)); print("Transformed corners:\\n",np.round(out,3))
        print(f"12 m side becomes {np.linalg.norm(out[1]-out[0]):.2f} map units")
    elif k=="E":
        A=np.array([[10.,-4],[-4,9]]); b=np.array([12.,-6]); x=np.linalg.solve(A,b)
        print("Two-loop electrical circuit"); print("Mesh currents (A):",x); print("Shared-resistor current (A):",x[0]-x[1]); print("Residual:",np.linalg.norm(b-A@x))
    elif k=="F":
        A=np.array([[3.,-1,-1,0],[-1,4,-2,-1],[-1,-2,5,-1],[0,-1,-1,3]])
        lu,p=lu_factor(A); loads=[np.eye(4)[i] for i in range(3)]+[np.array([1.,1,0,0]),np.array([2.,0,0,1])]
        print("Resistive network with five load scenarios")
        for i,b in enumerate(loads,1):
            x=lu_solve((lu,p),b); print(f"Scenario {i}: {np.round(x,4)} residual={np.linalg.norm(b-A@x):.2e}")
    elif k=="G":
        n=7; A=2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1); b=np.zeros(n); b[0]=80; b[-1]=30
        xj,ej=jacobi(A,b,1e-6); xg,eg=gauss_seidel(A,b,1e-6)
        print("Temperature along a heated rod"); print("Jacobi:",np.round(xj,3)); print("Gauss-Seidel:",np.round(xg,3)); print("Direct:",np.round(np.linalg.solve(A,b),3)); print("Iterations:",len(ej),len(eg))
    elif k=="H":
        names=["Router","Server1","Server2","Printer","Laptop"]; A=np.zeros((5,5))
        for i,j in [(0,1),(0,2),(0,3),(0,4),(1,2),(2,4)]: A[i,j]=A[j,i]=1
        w,v=np.linalg.eig(A); i=np.argmax(w.real); s=np.abs(v[:,i].real); s/=s.sum()
        print("Office network node importance"); print("Dominant eigenvalue:",round(w[i].real,4))
        for name,val in sorted(zip(names,s),key=lambda z:-z[1]): print(f"{name}: {val:.4f}")
    elif k=="I":
        f=lambda q:50*q-(2000+20*q+.05*q*q)
        lo=root_scalar(f,bracket=[0,200],method="bisect").root; hi=root_scalar(f,bracket=[300,700],method="bisect").root
        print("Business break-even production"); print(f"Lower break-even: {lo:.4f} units"); print(f"Upper break-even: {hi:.4f} units"); print(f"Maximum profit at 300 units: KES {f(300):.2f}")
    elif k=="J":
        # KES 100,000 is received and repaid as eight monthly instalments of KES 14,000.
        cashflows=[100000]+[-14000]*8
        npv=lambda r: sum(c/(1+r)**t for t,c in enumerate(cashflows))
        sol=root_scalar(npv,x0=.01,x1=.05,method="secant",xtol=1e-12)
        rate=sol.root
        print("True interest rate of a loan")
        print(f"Monthly rate: {rate:.6%}")
        print(f"Nominal annual rate: {12*rate:.4%}")
        print(f"Effective annual rate: {(1+rate)**12-1:.4%}")
        print("Total repaid: KES 112000 | Total interest: KES 12000")
    elif k=="K":
        f=lambda z:[z[0]**2+z[1]**2-25,(z[0]-6)**2+z[1]**2-16]
        print("Overlapping cell-tower coverage")
        for guess in ([3.,3.],[3.,-3.]):
            sol=root(f,guess)
            print("Guess",guess,"-> intersection (km):",np.round(sol.x,5),"residuals:",np.round(f(sol.x),10))
    elif k=="L":
        from scipy.interpolate import interp1d, CubicSpline
        day=np.arange(1,11,dtype=float); moisture=np.array([32,30.5,29,np.nan,26.5,25,24.5,30,29.5,28])
        good=~np.isnan(moisture)
        linear=interp1d(day[good],moisture[good]); cubic=CubicSpline(day[good],moisture[good])
        print("Missing soil-moisture reading")
        print(f"Day 4 linear estimate: {float(linear(4)):.2f}%")
        print(f"Day 4 cubic-spline estimate: {float(cubic(4)):.2f}%")
    elif k=="M":
        from scipy.optimize import curve_fit
        weeks=np.arange(1,9,dtype=float); demand=np.array([120,138,160,185,215,250,290,335.])
        model=lambda x,a,b:a*np.exp(b*x)
        params,_=curve_fit(model,weeks,demand,p0=(100,.1))
        fitted=model(weeks,*params); rmse=np.sqrt(np.mean((demand-fitted)**2))
        print("Product demand forecasting")
        print(f"Fitted model: y = {params[0]:.2f} * exp({params[1]:.4f} * week)")
        print(f"RMSE: {rmse:.2f} units")
        print(f"Weekly growth: {100*(np.exp(params[1])-1):.1f}%")
        print(f"Forecast for week 10: {model(10,*params):.1f} units")
    elif k=="N":
        t=np.array([0,2,4,6,8,10.]); v=np.array([0,5,12,20,27,31.])
        acceleration=np.gradient(v,t)
        print("Acceleration from velocity readings")
        print("Acceleration estimates (m/s^2):",np.round(acceleration,3))
        print(f"Peak acceleration: {acceleration.max():.2f} m/s^2 at t={t[acceleration.argmax()]:.0f} s")
    elif k=="O":
        t=np.array([0,30,60,90,120,150,180.]); v=np.array([0,12,18,22,20,10,0.])
        print("Distance from speed readings"); print(f"Trapezoid: {np.trapezoid(v,t):.1f} m"); print(f"Simpson: {simpson(v,x=t):.1f} m"); print(f"Average speed: {np.trapezoid(v,t)/180*3.6:.2f} km/h")
    elif k=="P":
        r,K,P0,h=.4,1000.,50.,1.; t=np.arange(31); y=np.zeros(31); y[0]=P0
        for i in range(30): y[i+1]=y[i]+h*r*y[i]*(1-y[i]/K)
        exact=K/(1+(K-P0)/P0*np.exp(-r*t))
        print("Logistic population growth")
        for d in (0,5,10,15,20,30): print(f"Day {d}: Euler {y[d]:.1f}, exact {exact[d]:.1f}")
    elif k=="Q":
        beta,gamma,N=.4,.1,10000.
        def sir(t,y):
            S,I,R=y; return [-beta*S*I/N,beta*S*I/N-gamma*I,gamma*I]
        sol=solve_ivp(sir,(0,160),[N-10,10,0],t_eval=np.linspace(0,160,641),rtol=1e-8)
        I=sol.y[1]; print("SIR epidemic model"); print(f"Peak infections: {I.max():.0f} at day {sol.t[I.argmax()]:.1f}"); print(f"Ever infected: {N-sol.y[0,-1]:.0f} ({(N-sol.y[0,-1])/N:.1%})")
    elif k=="R":
        best=None
        for m in range(30):
            for b in range(20):
                seats=14*m+33*b; cost=8000*m+15000*b
                if seats>=200 and (best is None or cost<best[0]): best=(cost,m,b,seats)
        print("Minimum-cost vehicle fleet"); print(f"Cost: KES {best[0]:,}; matatus: {best[1]}; buses: {best[2]}; seats: {best[3]}")
    elif k=="S":
        rng=np.random.default_rng(7); N=200_000; p=.05; fails=rng.random((N,5))<p; down=(fails.sum(axis=1)>=3)
        exact=sum(comb(5,j)*p**j*(1-p)**(5-j) for j in range(3,6))
        print("Server-cluster reliability"); print(f"Monte Carlo: {down.mean():.5%}"); print(f"Exact binomial: {exact:.5%}")
        fails2=rng.random((N,5))<.10; print(f"With 10% individual failure probability: {(fails2.sum(axis=1)>=3).mean():.4%}")
    else: raise ValueError("Task letter must be A-S")
