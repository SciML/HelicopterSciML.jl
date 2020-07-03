using Pkg
cd(@__DIR__)
Pkg.activate("../"); Pkg.instantiate()

using Plots; gr()
using LaTeXStrings
using DataFrames
using CSV
using OrdinaryDiffEq
using LinearAlgebra
using DataInterpolations
using BenchmarkTools
using DiffEqFlux
using Optim
using Flux
using Zygote

# Plot path
figpath = "../figs/";
# Linewidths and styles
LW1 = 2.5
LW2 = 1.5
LS1 = :solid
LS2 = :dot
LS3 = :dash
LC1 = :blue
LC2 = :red
LC3 = :green
LC4 = :black
LA1 = 1.0
LA2 = 0.7
LA3 = 0.3
;

df=CSV.read("../data/Lab-Helicopter_Experimental-data.csv")
ENV["COLUMNS"]=100 # change the number of IJulia columns from default 80 to 100
df[1:5,:]

u_th = df[:,1]
u_psi = df[:,2]
theta = df[:,3]
psi = df[:,4]
dt = df[1,5]
tm = 0:dt:(length(psi)-1)*dt
;

timeseriesdata = [theta';psi']
Nd = length(tm);

U_th = ConstantInterpolation(u_th,tm)
U_psi = ConstantInterpolation(u_psi,tm)
Theta = ConstantInterpolation(theta,tm)
Psi = ConstantInterpolation(psi,tm);
u_t = (th=U_th, psi=U_psi)
rad2deg(t,u) = (t,u*180/pi)

include("../optimization_results/globaloptimizationresults.jl")

###############
## Now go neural, nonlinear K
###############

model_univ_ux = FastChain(FastDense(6, 16, tanh),
                       FastDense(16, 2))
pinit = initial_params(model_univ_ux)/1000

function helicopter_uode(dx,x,Ps,P0,u,p,t)
# P0 = (rc=rc,h=h,m=m,Jbc=Jbc,Js=Js,g=g,K_th_th=K_th_th,K_th_psi=K_th_psi,K_psi_th=K_psi_th,K_psi_psi=K_psi_psi,D_th=D_th,D_psi=D_psi);
    th = x[1]
    psi = x[2]
    w_th = x[3]
    w_psi = x[4]

    U = [u.th(t),u.psi(t)]
    K = model_univ_ux([U;x],p)
    #
    M11 = Ps[4]*P0.Jbc + Ps[3]*P0.m*((Ps[1]*P0.rc)^2+(Ps[2]*P0.h)^2)
    M22 = Ps[4]*P0.Jbc*cos(th)^2 + Ps[3]*P0.m*(Ps[1]*P0.rc*cos(th)-Ps[2]*P0.h*sin(th))^2 + Ps[5]*P0.Js
    #
    dLdth = -w_psi^2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th)-(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2)) - (Ps[3]*P0.m)*(Ps[6]*P0.g)*((Ps[1]*P0.rc)*cos(th) - (Ps[2]*P0.h)*sin(th))
    #
    dp_psi_dth = -2*(((Ps[4]*P0.Jbc)+(Ps[3]*P0.m)*((Ps[1]*P0.rc)^2-(Ps[2]*P0.h)^2))*cos(th)*sin(th) + (Ps[3]*P0.m)*(Ps[1]*P0.rc)*(Ps[2]*P0.h)*(cos(th)^2-sin(th)^2))*w_psi
    #
    F_th = Ps[7]*P0.K_th_th*U[1] - Ps[8]*P0.K_th_psi*U[2] - P0.D_th*w_th + K[1]
    F_psi = Ps[9]*P0.K_psi_th*U[1] - Ps[10]*P0.K_psi_psi*U[2] - P0.D_psi*w_psi + K[2]

    #
    dx[1] = w_th
    dx[2] = w_psi
    dx[3] = (dLdth + F_th)/M11
    dx[4] = (-dp_psi_dth*w_th + F_psi)/M22
end

uode_f! = (dx,x,p,t) -> helicopter_uode(dx,x,Ps_best,P0,u_t,p,t)

tspan = (tm[1],tm[end])

#
prob = ODEProblem(uode_f!,x0_best,tspan,pinit)
#
sol = solve(prob,AutoTsit5(TRBDF2(autodiff=false)),reltol=1e-6)

function uode_cost(p)
    #
    prob = ODEProblem(uode_f!,x0_best,tspan,p)
    sol = solve(prob, AutoTsit5(TRBDF2(autodiff=false)), saveat=dt)
    #
    return norm(sol[1:2,:]-timeseriesdata[:,1:size(sol,2)],2)^2/Nd
    #
end

iter = 0
callback = function (p, l)
    global iter
    @show iter
    @show l
    prob = ODEProblem(uode_f!,x0_best,tspan,p)
    sol = solve(prob, AutoTsit5(TRBDF2(autodiff=false)), saveat=dt)
    pl = plot(sol,vars=(rad2deg,0,2),lw=LW1,lc=LC1)
    plot!(pl,tm,psi*180/pi,lw=LW1,ls=LS2,lc=LC2)
    plot!(pl,xlim = tspan)
    plot!(pl)
    display(pl)
    iter += 1
    return false
end

callback(pinit,uode_cost(pinit))

result_univ1_ux = DiffEqFlux.sciml_train(uode_cost, pinit,
                                     ADAM(0.001), maxiters = 100,
                                     cb = callback)

result_univ2_ux = DiffEqFlux.sciml_train(uode_cost, result_univ1_ux.minimizer,
                                     BFGS(initial_stepnorm = 0.01), maxiters = 100,
                                     cb = callback)

prob_ux = ODEProblem(uode_f!,x0_best,tspan,result_univ2_ux.minimizer)
#
sol_ux = solve(prob_ux,AutoTsit5(TRBDF2(autodiff=false)),reltol=1e-6);

plot(sol_ux,vars=(rad2deg,0,1),lw=LW1,lc=LC1,label=L"$\theta$")
plot!(tm,theta*180/pi,lw=LW1,ls=LS2,lc=LC2,label=L"\theta^\mathrm{d}")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel=L"$\theta$ [$^\circ$]")
plot!(title="Pitch angle: model+FNN(u,x) (blue) vs. data (red)",box=true)
figname="Helicopter_pitch-angle_model_fit_FNN_u-x.svg"
savefig(figpath*figname)

plot(sol_ux,vars=(rad2deg,0,2),lw=LW1,lc=LC1,label=L"$\psi$")
plot!(tm,psi*180/pi,lw=LW1,ls=LS2,lc=LC2,label=L"\psi^\mathrm{d}")
plot!(xlim = tspan,xlabel=L"time $t$ [s]", ylabel=L"$\psi$ [$^\circ$]")
plot!(title="Yaw angle: model+FNN(u,x) (blue) vs. data (red)",box=true)
figname="Helicopter_yaw-angle_model_fit_FNN_u-x.svg"
savefig(figpath*figname)

# Summarizing results to file
pnn = result_univ2_ux.minimizer
prob = ODEProblem(uode_f!,x0_best,tspan,pnn)
sol = solve(prob, AutoTsit5(TRBDF2(autodiff=false)), saveat=dt)
U(t) = [u_t.th(t);u_t.psi(t);sol(t)]
u_ux_ts = U.(sol.t)
K_ux_ts = model_univ_ux.(u_ux_ts,(pnn,))

open("../optimization_results/neural_augmentation_results_ux.jl", "w") do f
    write(f,"pnn = $(result_univ2_ux.minimizer) \n")
    write(f,"utimeseries = $(u_ux_ts) \n")
    write(f,"Ktimeseries = $(K_ux_ts) \n")
end
